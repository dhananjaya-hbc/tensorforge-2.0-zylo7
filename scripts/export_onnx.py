"""Export a trained Model B run to an int8 ONNX bundle for CPU serving (no torch at runtime).

Writes <out>/: model_b.onnx (int8), tokenizer/, preprocessor.joblib, meta.json, parity.json.
The parity check compares the quantised ONNX model against the PyTorch model on the validation set.

Large encoders need care with RAM (LaBSE: ~1.9 GB fp32; tracing peaks near 9 GB and free Colab has ~12.7 GB).
Python does not reliably return freed memory to the OS, so each heavy stage runs in its own child process:
  1. export    torch -> fp32 ONNX (+ PyTorch reference probabilities for the parity check)
  2. quantize  fp32 ONNX -> int8 ONNX (fresh process, so the export peak is gone)
  3. finalize  pack into one file, parity check, copy tokenizer/preprocessor/meta
Each stage prints its elapsed time and peak memory.

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
    step("export: loading PyTorch weights")
    model = MultiTaskClassifier(meta["encoder"])
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    wrapper = InferenceWrapper(model).eval()

    step("export: PyTorch reference probabilities for the parity check")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper.to(dev)
    ref = []
    with torch.no_grad():
        for ids, mask in batches(tok, load_texts(run, meta), meta["max_len"]):
            ref.append(wrapper(torch.from_numpy(ids).to(dev), torch.from_numpy(mask).to(dev))[0].float().cpu().numpy())
    np.save(scratch / "reference_primary.npy", np.concatenate(ref))
    wrapper.to("cpu")

    step("export: tracing fp32 ONNX graph")
    dummy = tok(["query: hello"], return_tensors="pt")
    torch.onnx.export(
        wrapper, (dummy["input_ids"], dummy["attention_mask"]), str(scratch / "fp32.onnx"),
        input_names=["input_ids", "attention_mask"], output_names=["primary", "secondary", "urgent"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "primary": {0: "batch"}, "secondary": {0: "batch"}, "urgent": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    step("export: done")


def stage_quantize(scratch: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    step("quantize: int8 dynamic quantisation (slowest step for large vocabularies; a few minutes)")
    # External-data mode lets the quantiser handle graphs near/over protobuf's 2 GB limit.
    quantize_dynamic(str(scratch / "fp32.onnx"), str(scratch / "int8.onnx"), weight_type=QuantType.QInt8,
                     use_external_data_format=True)
    step("quantize: done")


def stage_finalize(run: Path, out: Path, scratch: Path, meta: dict) -> None:
    import onnx
    import onnxruntime as ort
    from transformers import AutoTokenizer

    step("finalize: packing int8 model into a single file")
    packed = onnx.load(str(scratch / "int8.onnx"))
    onnx.save_model(packed, str(out / "model_b.onnx"))
    del packed

    step("finalize: parity check, int8 ONNX vs PyTorch on validation")
    tok = AutoTokenizer.from_pretrained(run / "tokenizer")
    sess = ort.InferenceSession(str(out / "model_b.onnx"), providers=["CPUExecutionProvider"])
    got = np.concatenate([sess.run(["primary"], {"input_ids": ids, "attention_mask": mask})[0]
                          for ids, mask in batches(tok, load_texts(run, meta), meta["max_len"])])
    ref = np.load(scratch / "reference_primary.npy")
    parity = {"argmax_agreement": float((got.argmax(1) == ref.argmax(1)).mean()),
              "max_abs_prob_diff": float(np.abs(got - ref).max())}
    print("parity:", parity, flush=True)

    tok.save_pretrained(out / "tokenizer")
    shutil.copy(run / "preprocessor.joblib", out / "preprocessor.joblib")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "parity.json").write_text(json.dumps(parity, indent=2))
    size_mb = (out / "model_b.onnx").stat().st_size / 1e6
    step(f"finalize: bundle written to {out} (model_b.onnx {size_mb:.0f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="output dir of train_transformer.py --save")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["export", "quantize", "finalize"], help=argparse.SUPPRESS)
    args = ap.parse_args()

    run, out = Path(args.run), Path(args.out)
    scratch = out / "_export_tmp"
    meta = json.loads((run / "meta.json").read_text())

    if args.stage == "export":
        return stage_export(run, scratch, meta)
    if args.stage == "quantize":
        return stage_quantize(scratch)
    if args.stage == "finalize":
        return stage_finalize(run, out, scratch, meta)

    # Orchestrator: run each stage in a fresh process so its memory is fully released afterwards.
    out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    for stage in ("export", "quantize", "finalize"):
        result = subprocess.run([sys.executable, __file__, "--run", str(run), "--out", str(out), "--stage", stage])
        if result.returncode != 0:
            reason = "killed (out of memory?)" if result.returncode < 0 or result.returncode == 137 else f"exit {result.returncode}"
            sys.exit(f"stage '{stage}' failed: {reason}")
    shutil.rmtree(scratch)


if __name__ == "__main__":
    main()
