"""Export a trained Model B run to an int8 ONNX bundle for CPU serving (no torch at runtime).

Writes <out>/: model_b.onnx (int8), tokenizer/, preprocessor.joblib, meta.json, parity.json.
The parity check compares the quantised ONNX model against the PyTorch model on the validation set.

Usage: python scripts/export_onnx.py --run runs/final --out bundle_b
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.data import load_split  # noqa: E402
from ml.transformer.model import InferenceWrapper, MultiTaskClassifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="output dir of train_transformer.py --save")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoTokenizer

    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((run / "meta.json").read_text())
    model = MultiTaskClassifier(meta["encoder"])
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    wrapper = InferenceWrapper(model).eval()
    tok = AutoTokenizer.from_pretrained(run / "tokenizer")

    dummy = tok(["query: hello"], return_tensors="pt")
    fp32 = out / "model_b.fp32.onnx"
    torch.onnx.export(
        wrapper, (dummy["input_ids"], dummy["attention_mask"]), fp32,
        input_names=["input_ids", "attention_mask"], output_names=["primary", "secondary", "urgent"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "primary": {0: "batch"}, "secondary": {0: "batch"}, "urgent": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    quantize_dynamic(str(fp32), str(out / "model_b.onnx"), weight_type=QuantType.QInt8)
    fp32.unlink()

    # Parity: int8 ONNX vs PyTorch on the validation set.
    pre = joblib.load(run / "preprocessor.joblib")
    fn = pre.denoised_view if meta["view"] == "denoised" else pre.full_view
    texts = [meta["prefix"] + fn(r) for r in load_split("validation")]
    sess = ort.InferenceSession(str(out / "model_b.onnx"), providers=["CPUExecutionProvider"])
    agree, max_diff = 0, 0.0
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i + 32], truncation=True, max_length=meta["max_len"], padding=True, return_tensors="np")
        feeds = {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)}
        p_onnx = sess.run(["primary"], feeds)[0]
        with torch.no_grad():
            p_torch = wrapper(torch.from_numpy(feeds["input_ids"]), torch.from_numpy(feeds["attention_mask"]))[0].numpy()
        agree += int((p_onnx.argmax(1) == p_torch.argmax(1)).sum())
        max_diff = max(max_diff, float(np.abs(p_onnx - p_torch).max()))
    parity = {"argmax_agreement": agree / len(texts), "max_abs_prob_diff": max_diff}
    print("parity:", parity)

    tok.save_pretrained(out / "tokenizer")
    shutil.copy(run / "preprocessor.joblib", out / "preprocessor.joblib")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "parity.json").write_text(json.dumps(parity, indent=2))
    size_mb = (out / "model_b.onnx").stat().st_size / 1e6
    print(f"bundle written to {out} (model_b.onnx {size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
