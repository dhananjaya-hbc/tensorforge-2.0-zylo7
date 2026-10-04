"""Fine-tune Model B (multilingual encoder + multi-task heads). Runs on CUDA (Kaggle/Colab), Apple MPS or CPU.

Model selection uses train -> validation, because the two splits use different phrasing templates
(see notebooks/01_data_exploration.ipynb and docs): validation is the honest estimate for unseen phrasings.

Examples:
  uv run python scripts/train_transformer.py --encoder intfloat/multilingual-e5-small --out runs/e5s
  uv run python scripts/train_transformer.py --train-on train+validation --epochs 4 --save --out runs/e5s-final
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.data import load_split  # noqa: E402
from ml.decide import Thresholds, decide  # noqa: E402
from ml.evaluate import score, score_by  # noqa: E402
from ml.labels import CAT2ID, CATEGORIES, LANGUAGES  # noqa: E402
from ml.preprocess import Preprocessor  # noqa: E402
from ml.transformer.model import MultiTaskClassifier  # noqa: E402

LANG2ID = {lang: i for i, lang in enumerate(LANGUAGES)}


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def texts(rows, pre: Preprocessor, view: str, prefix: str) -> list[str]:
    fn = pre.denoised_view if view == "denoised" else pre.full_view
    return [prefix + fn(r) for r in rows]


def targets(rows):
    n = len(rows)
    sec = np.zeros((n, len(CATEGORIES)), dtype=np.float32)
    sec_w = np.ones((n, len(CATEGORIES)), dtype=np.float32)
    for i, r in enumerate(rows):
        sec_w[i, CAT2ID[r["category"]]] = 0.0  # the primary label is never a valid secondary
        if r["secondary_category"]:
            sec[i, CAT2ID[r["secondary_category"]]] = 1.0
    return {
        "primary": torch.tensor([CAT2ID[r["category"]] for r in rows]),
        "secondary": torch.tensor(sec),
        "secondary_w": torch.tensor(sec_w),
        "urgent": torch.tensor([float(r["is_urgent"]) for r in rows]),
        "language": torch.tensor([LANG2ID.get(r.get("language"), 0) for r in rows]),
    }


@torch.no_grad()
def predict(model, tok, xs, args, dev):
    model.eval()
    out = {"primary": [], "secondary": [], "urgent": []}
    for i in range(0, len(xs), args.batch * 2):
        enc = tok(xs[i:i + args.batch * 2], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt").to(dev)
        o = model(enc["input_ids"], enc["attention_mask"])
        out["primary"].append(torch.softmax(o["primary"], -1).float().cpu().numpy())
        out["secondary"].append(torch.sigmoid(o["secondary"]).float().cpu().numpy())
        out["urgent"].append(torch.sigmoid(o["urgent"]).float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="intfloat/multilingual-e5-small")
    ap.add_argument("--train-on", default="train", choices=["train", "train+validation"])
    ap.add_argument("--view", default="denoised", choices=["denoised", "full"])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--w-secondary", type=float, default=0.5)
    ap.add_argument("--w-urgent", type=float, default=0.5)
    ap.add_argument("--w-language", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/model_b")
    ap.add_argument("--save", action="store_true", help="save weights + tokenizer of the last epoch")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N tickets of each split (smoke tests)")
    args = ap.parse_args()

    seed_all(args.seed)
    dev = device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = load_split("train") + (load_split("validation") if args.train_on == "train+validation" else [])
    val = load_split("validation")
    if args.limit:
        train, val = train[:args.limit], val[:args.limit]
    pre = Preprocessor().fit(train)
    prefix = "query: " if "e5" in args.encoder else ""
    x_tr, x_va = texts(train, pre, args.view, prefix), texts(val, pre, args.view, prefix)
    y = targets(train)

    tok = AutoTokenizer.from_pretrained(args.encoder)
    model = MultiTaskClassifier(args.encoder).to(dev)
    counts = np.bincount(y["primary"].numpy(), minlength=len(CATEGORIES))
    ce_primary = nn.CrossEntropyLoss(weight=torch.tensor(len(train) / (len(CATEGORIES) * np.maximum(counts, 1)), dtype=torch.float32).to(dev))
    bce = nn.BCEWithLogitsLoss(reduction="none")
    bce_urgent = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0).to(dev))
    ce_lang = nn.CrossEntropyLoss()

    heads = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.lr},
        {"params": heads, "lr": args.head_lr},
    ], weight_decay=0.01)
    steps = args.epochs * ((len(train) + args.batch - 1) // args.batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    use_amp = dev.type == "cuda"  # fp16 mixed precision on GPU only
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"device={dev} amp={use_amp} train={len(train)} encoder={args.encoder} view={args.view}", flush=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(train))
        t0, running = time.time(), 0.0
        for b, i in enumerate(range(0, len(order), args.batch)):
            idx = order[i:i + args.batch]
            enc = tok([x_tr[j] for j in idx], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt").to(dev)
            t = {k: v[idx].to(dev) for k, v in y.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                o = model(enc["input_ids"], enc["attention_mask"])
            o = {k: v.float() for k, v in o.items()}
            sec_loss = (bce(o["secondary"], t["secondary"]) * t["secondary_w"]).sum() / t["secondary_w"].sum()
            loss = (ce_primary(o["primary"], t["primary"]) + args.w_secondary * sec_loss
                    + args.w_urgent * bce_urgent(o["urgent"], t["urgent"]) + args.w_language * ce_lang(o["language"], t["language"]))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
        probs = predict(model, tok, x_va, args, dev)
        m = score(val, decide(**probs, th=Thresholds(0.5, 0.5)))
        m.update(epoch=epoch, train_loss=round(running / (b + 1), 4), seconds=round(time.time() - t0))
        history.append(m)
        print(json.dumps({k: m[k] for k in ("epoch", "train_loss", "primary_accuracy", "primary_macro_f1", "secondary_f1", "urgent_f1", "ece", "seconds")}), flush=True)
        np.savez(out / f"val_probs_e{epoch}.npz", **probs)

    preds = decide(**probs, th=Thresholds(0.5, 0.5))
    report = {"args": vars(args), "history": history, "validation_by_language": score_by(val, preds, "language")}
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    for lang, s in report["validation_by_language"].items():
        print(f"  {lang:9s} acc={s['primary_accuracy']:.3f} macroF1={s['primary_macro_f1']:.3f}")
    if args.save:
        model.cpu()
        torch.save(model.state_dict(), out / "model.pt")
        tok.save_pretrained(out / "tokenizer")
        joblib.dump(pre, out / "preprocessor.joblib")
        meta = {"encoder": args.encoder, "prefix": prefix, "view": args.view, "max_len": args.max_len,
                "trained_on": args.train_on, "epochs": args.epochs}
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"saved weights to {out}")


if __name__ == "__main__":
    main()
