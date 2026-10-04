"""Export a trained Model B run to an int8 ONNX bundle for CPU serving (no torch at runtime).

Writes <out>/: model_b.onnx (int8), tokenizer/, preprocessor.joblib, meta.json, parity.json.
The parity check compares the quantised ONNX model against the PyTorch model on the validation set.

Memory matters for large encoders (LaBSE is ~1.9 GB in fp32; free Colab has ~12.7 GB RAM), so the
PyTorch reference probabilities are computed first and the PyTorch model is freed before quantisation.
Each step prints its elapsed time and peak memory.

Usage: python scripts/export_onnx.py --run runs/final --out bundle_b
"""

import argparse
import gc
import json
import resource
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.data import load_split  # noqa: E402
from ml.transformer.model import InferenceWrapper, MultiTaskClassifier  # noqa: E402

T0 = time.time()


def step(msg: str) -> None:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = peak / 1e9 if sys.platform == "darwin" else peak / 1e6  # bytes on macOS, KB on Linux
    print(f"[{time.time() - T0:6.0f}s | peak RAM {peak_gb:4.1f} GB] {msg}", flush=True)


def batches(tok, texts, max_len, size=32):
    for i in range(0, len(texts), size):
        enc = tok(texts[i:i + size], truncation=True, max_length=max_len, padding=True, return_tensors="np")
        yield enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="output dir of train_transformer.py --save")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoTokenizer

    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((run / "meta.json").read_text())
    tok = AutoTokenizer.from_pretrained(run / "tokenizer")
    pre = joblib.load(run / "preprocessor.joblib")
    fn = pre.denoised_view if meta["view"] == "denoised" else pre.full_view
    texts = [meta["prefix"] + fn(r) for r in load_split("validation")]

    step("loading PyTorch weights")
    model = MultiTaskClassifier(meta["encoder"])
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    wrapper = InferenceWrapper(model).eval()

    step("PyTorch reference probabilities for the parity check")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper.to(dev)
    ref = []
    with torch.no_grad():
        for ids, mask in batches(tok, texts, meta["max_len"]):
            ref.append(wrapper(torch.from_numpy(ids).to(dev), torch.from_numpy(mask).to(dev))[0].float().cpu().numpy())
    ref = np.concatenate(ref)
    wrapper.to("cpu")

    scratch = out / "_export_tmp"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    fp32 = scratch / "fp32.onnx"
    step("exporting fp32 ONNX graph")
    dummy = tok(["query: hello"], return_tensors="pt")
    torch.onnx.export(
        wrapper, (dummy["input_ids"], dummy["attention_mask"]), str(fp32),
        input_names=["input_ids", "attention_mask"], output_names=["primary", "secondary", "urgent"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "primary": {0: "batch"}, "secondary": {0: "batch"}, "urgent": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    del wrapper, model, dummy
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    step("quantising to int8 (the slowest step for large vocabularies; a few minutes)")
    int8 = scratch / "int8.onnx"
    # External-data mode lets the quantiser handle graphs near/over protobuf's 2 GB limit.
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8, use_external_data_format=True)

    step("packing int8 model into a single file")
    packed = onnx.load(str(int8))
    onnx.save_model(packed, str(out / "model_b.onnx"))
    del packed
    shutil.rmtree(scratch)
    gc.collect()

    step("parity check: int8 ONNX vs PyTorch on validation")
    sess = ort.InferenceSession(str(out / "model_b.onnx"), providers=["CPUExecutionProvider"])
    got = np.concatenate([sess.run(["primary"], {"input_ids": ids, "attention_mask": mask})[0]
                          for ids, mask in batches(tok, texts, meta["max_len"])])
    parity = {"argmax_agreement": float((got.argmax(1) == ref.argmax(1)).mean()),
              "max_abs_prob_diff": float(np.abs(got - ref).max())}
    print("parity:", parity, flush=True)

    tok.save_pretrained(out / "tokenizer")
    shutil.copy(run / "preprocessor.joblib", out / "preprocessor.joblib")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "parity.json").write_text(json.dumps(parity, indent=2))
    size_mb = (out / "model_b.onnx").stat().st_size / 1e6
    step(f"bundle written to {out} (model_b.onnx {size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
