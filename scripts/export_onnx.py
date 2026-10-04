"""Export a trained Model B run to an int8 ONNX bundle for CPU serving (no torch at runtime).

Writes <out>/: model_b.onnx, tokenizer/, preprocessor.joblib, meta.json, parity.json.
The bundle is only written if the quantised model agrees with PyTorch on >= 98% of validation tickets.

Pipeline (each heavy stage runs in its own process, because Python does not reliably return freed memory
to the OS and LaBSE tracing alone peaks near 9 GB on free Colab's ~12.7 GB):
  1. export      torch -> fp32 ONNX, traced with a *padded* batch and eager attention so the padding mask
                 is part of the graph (an unpadded trace can bake in "no mask" and break batched inference)
  2. check fp32  fp32 ONNX vs PyTorch; a mismatch here is an export bug, so stop
  3. quantize    try int8 everywhere (per-channel); if parity fails, keep the embedding table in fp32 and
                 quantise only the MatMul layers (large vocabularies are the part most sensitive to int8).
                 reduce_range=True (7-bit weights) avoids the u8s8 saturation error on x86 CPUs without VNNI.
  4. finalize    pack the first passing candidate into one file and write the bundle

Usage: python scripts/export_onnx.py --run runs/final --out bundle_b
"""

import argparse
import json
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

T0 = time.time()
MIN_AGREEMENT = 0.98
QUANT_MODES = ("int8_all", "int8_matmul_only")


def step(msg: str) -> None:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = peak / 1e9 if sys.platform == "darwin" else peak / 1e6  # bytes on macOS, KB on Linux
    print(f"[{time.time() - T0:6.0f}s | peak RAM {peak_gb:4.1f} GB] {msg}", flush=True)


def load_texts(run: Path, meta: dict) -> list[str]:
    import joblib

    from ml.data import load_split

    pre = joblib.load(run / "preprocessor.joblib")
    fn = pre.denoised_view if meta["view"] == "denoised" else pre.full_view
    return [meta["prefix"] + fn(r) for r in load_split("validation")]


def batches(tok, texts, max_len, size=32):
    for i in range(0, len(texts), size):
        enc = tok(texts[i:i + size], truncation=True, max_length=max_len, padding=True, return_tensors="np")
        yield enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)


def stage_export(run: Path, scratch: Path, meta: dict) -> None:
    import torch
    from transformers import AutoTokenizer

    from ml.transformer.model import InferenceWrapper, MultiTaskClassifier

    tok = AutoTokenizer.from_pretrained(run / "tokenizer")
    step("export: loading PyTorch weights (eager attention)")
    model = MultiTaskClassifier(meta["encoder"], attn_implementation="eager")
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    wrapper = InferenceWrapper(model).eval()

    step("export: PyTorch reference probabilities for the parity checks")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper.to(dev)
    ref = []
    with torch.no_grad():
        for ids, mask in batches(tok, load_texts(run, meta), meta["max_len"]):
            ref.append(wrapper(torch.from_numpy(ids).to(dev), torch.from_numpy(mask).to(dev))[0].float().cpu().numpy())
    np.save(scratch / "reference_primary.npy", np.concatenate(ref))
    wrapper.to("cpu")

    step("export: tracing fp32 ONNX graph with a padded batch")
    dummy = tok([meta["prefix"] + "hello", meta["prefix"] + "my order never arrived and the rider is not answering"],
                padding=True, return_tensors="pt")
    assert int(dummy["attention_mask"].min()) == 0, "trace batch must contain padding"
    torch.onnx.export(
        wrapper, (dummy["input_ids"], dummy["attention_mask"]), str(scratch / "fp32.onnx"),
        input_names=["input_ids", "attention_mask"], output_names=["primary", "secondary", "urgent"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "primary": {0: "batch"}, "secondary": {0: "batch"}, "urgent": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    step("export: done")


def stage_check(run: Path, scratch: Path, meta: dict, model_file: str, tag: str) -> None:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    step(f"check[{tag}]: scoring validation with ONNX Runtime")
    tok = AutoTokenizer.from_pretrained(run / "tokenizer")
    sess = ort.InferenceSession(str(scratch / model_file), providers=["CPUExecutionProvider"])
    got = np.concatenate([sess.run(["primary"], {"input_ids": ids, "attention_mask": mask})[0]
                          for ids, mask in batches(tok, load_texts(run, meta), meta["max_len"])])
    ref = np.load(scratch / "reference_primary.npy")
    parity = {"argmax_agreement": float((got.argmax(1) == ref.argmax(1)).mean()),
              "max_abs_prob_diff": float(np.abs(got - ref).max())}
    (scratch / f"parity_{tag}.json").write_text(json.dumps(parity))
    step(f"check[{tag}]: {parity}")


def stage_quantize(scratch: Path, mode: str) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    step(f"quantize[{mode}]: int8 dynamic quantisation (a few minutes)")
    extra = {"op_types_to_quantize": ["MatMul"]} if mode == "int8_matmul_only" else {}
    # External-data mode lets the quantiser handle graphs near/over protobuf's 2 GB limit.
    quantize_dynamic(str(scratch / "fp32.onnx"), str(scratch / f"{mode}.onnx"), weight_type=QuantType.QInt8,
                     per_channel=True, reduce_range=True, use_external_data_format=True, **extra)
    step(f"quantize[{mode}]: done")


def stage_finalize(run: Path, out: Path, scratch: Path, meta: dict, mode: str) -> None:
    import onnx
    from transformers import AutoTokenizer

    step(f"finalize: packing {mode} model into a single file")
    packed = onnx.load(str(scratch / f"{mode}.onnx"))
    onnx.save_model(packed, str(out / "model_b.onnx"))
    del packed
    parity = {"quantization": mode, **json.loads((scratch / f"parity_{mode}.json").read_text()),
              "fp32_export": json.loads((scratch / "parity_fp32.json").read_text())}
    AutoTokenizer.from_pretrained(run / "tokenizer").save_pretrained(out / "tokenizer")
    shutil.copy(run / "preprocessor.joblib", out / "preprocessor.joblib")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "parity.json").write_text(json.dumps(parity, indent=2))
    size_mb = (out / "model_b.onnx").stat().st_size / 1e6
    print("parity:", parity, flush=True)
    step(f"finalize: bundle written to {out} (model_b.onnx {size_mb:.0f} MB)")


def child(args_list: list[str], what: str) -> None:
    result = subprocess.run([sys.executable, __file__, *args_list])
    if result.returncode != 0:
        reason = "killed (out of memory?)" if result.returncode < 0 or result.returncode == 137 else f"exit {result.returncode}"
        sys.exit(f"EXPORT FAILED during {what}: {reason}")


def agreement(scratch: Path, tag: str) -> float:
    return json.loads((scratch / f"parity_{tag}.json").read_text())["argmax_agreement"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="output dir of train_transformer.py --save")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["export", "check", "quantize", "finalize"], help=argparse.SUPPRESS)
    ap.add_argument("--file", help=argparse.SUPPRESS)
    ap.add_argument("--tag", help=argparse.SUPPRESS)
    ap.add_argument("--min-agreement", type=float, default=MIN_AGREEMENT, help=argparse.SUPPRESS)  # smoke tests
    args = ap.parse_args()

    run, out = Path(args.run), Path(args.out)
    scratch = out / "_export_tmp"
    meta = json.loads((run / "meta.json").read_text())

    if args.stage == "export":
        return stage_export(run, scratch, meta)
    if args.stage == "check":
        return stage_check(run, scratch, meta, args.file, args.tag)
    if args.stage == "quantize":
        return stage_quantize(scratch, args.tag)
    if args.stage == "finalize":
        return stage_finalize(run, out, scratch, meta, args.tag)

    # Orchestrator: every heavy stage in a fresh process, so its memory is fully released afterwards.
    base = ["--run", str(run), "--out", str(out)]
    shutil.rmtree(out, ignore_errors=True)
    scratch.mkdir(parents=True)
    child([*base, "--stage", "export"], "export")
    child([*base, "--stage", "check", "--file", "fp32.onnx", "--tag", "fp32"], "fp32 check")
    if agreement(scratch, "fp32") < 0.995:
        sys.exit("EXPORT FAILED: the fp32 ONNX graph already disagrees with PyTorch, so this is an export bug, "
                 "not quantisation. See parity_fp32.json in the scratch folder.")
    for mode in QUANT_MODES:
        child([*base, "--stage", "quantize", "--tag", mode], f"quantize {mode}")
        child([*base, "--stage", "check", "--file", f"{mode}.onnx", "--tag", mode], f"{mode} check")
        if agreement(scratch, mode) >= args.min_agreement:
            child([*base, "--stage", "finalize", "--tag", mode], "finalize")
            shutil.rmtree(scratch)
            return
        step(f"{mode} agreement below {args.min_agreement:.0%}, trying the next option")
    sys.exit(f"EXPORT FAILED: no quantisation option reached {args.min_agreement:.0%} agreement with PyTorch")


if __name__ == "__main__":
    main()
