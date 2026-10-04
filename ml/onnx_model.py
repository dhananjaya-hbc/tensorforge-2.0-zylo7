"""Serving-side Model B: the exported ONNX bundle, run with ONNX Runtime and HF `tokenizers` (no torch).

Bundle layout (scripts/export_onnx.py + notebooks/03_final_model_training.ipynb):
    model_b.onnx, tokenizer/tokenizer.json, preprocessor.joblib, meta.json, model_card.json

Length bucketing: tickets are only batched with tickets of the same token length, so no padding is ever
added. A ticket's prediction is therefore identical whether it is sent alone, in a batch or in a job,
at any position (the spec requires predictions to be independent of batch neighbours and deterministic).
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

BATCH = 32


class OnnxModel:
    def __init__(self, bundle: Path, threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        bundle = Path(bundle)
        self.meta = json.loads((bundle / "meta.json").read_text())
        self.pre = joblib.load(bundle / "preprocessor.joblib")
        self.tok = Tokenizer.from_file(str(bundle / "tokenizer" / "tokenizer.json"))
        self.tok.no_padding()
        self.tok.enable_truncation(max_length=self.meta["max_len"])
        opts = ort.SessionOptions()
        # One thread per call by default: the session is shared by the job worker and live requests, which then
        # run side by side (2 vCPU = 2 hyperthreads) instead of competing for both, keeping /predict fast while a
        # job runs. ORT_THREADS=0 lets one call use every core.
        opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "1")) if threads is None else threads
        self.session = ort.InferenceSession(str(bundle / "model_b.onnx"), opts, providers=["CPUExecutionProvider"])

    def text(self, row: dict) -> str:
        view = self.pre.denoised_view if self.meta["view"] == "denoised" else self.pre.full_view
        return self.meta["prefix"] + view(row)

    def predict_proba(self, rows: list[dict]) -> dict[str, np.ndarray]:
        encodings = self.tok.encode_batch([self.text(r) for r in rows])
        buckets: dict[int, list[int]] = defaultdict(list)
        for i, enc in enumerate(encodings):
            buckets[len(enc.ids)].append(i)
        out = {"primary": None, "secondary": None, "urgent": None}
        for idx in buckets.values():
            for start in range(0, len(idx), BATCH):
                chunk = idx[start:start + BATCH]
                ids = np.array([encodings[i].ids for i in chunk], dtype=np.int64)
                feeds = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
                primary, secondary, urgent = self.session.run(["primary", "secondary", "urgent"], feeds)
                for name, value in (("primary", primary), ("secondary", secondary), ("urgent", urgent)):
                    if out[name] is None:
                        out[name] = np.zeros((len(rows),) + value.shape[1:], dtype=np.float32)
                    out[name][chunk] = value
        return out
