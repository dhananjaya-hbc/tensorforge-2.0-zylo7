"""Serving Model B from an ONNX bundle, using a tiny fake bundle built in code (no downloads)."""

import json

import joblib
import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from app.config import Settings
from app.main import create_app
from app.predictor import Predictor
from ml.onnx_model import OnnxModel
from ml.preprocess import Preprocessor
from tests.conftest import KEY, schema_errors

WORDS = ["query", "chat", "email", "body", "my", "bag", "food", "late", "refund", "driver", "scared", "app", "crash"]


def fake_graph(path):
    """input_ids/attention_mask -> primary(11, softmax), secondary(11, sigmoid), urgent(sigmoid), driven by token ids."""
    rng = np.random.default_rng(0)
    init = [numpy_helper.from_array(a.astype(np.float32), n) for n, a in (
        ("w1", rng.normal(0, 0.05, (1, 11))), ("b1", rng.normal(0, 1, 11)),
        ("w2", rng.normal(0, 0.05, (1, 11))), ("b2", rng.normal(0, 1, 11)),
        ("w3", rng.normal(0, 0.05, (1, 1))), ("b3", rng.normal(0, 1, 1)))]
    init.append(numpy_helper.from_array(np.array([1], dtype=np.int64), "axis1"))
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["x"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["attention_mask"], ["m"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["x", "m"], ["xm"]),
        helper.make_node("ReduceSum", ["xm", "axis1"], ["s"], keepdims=1),
        helper.make_node("MatMul", ["s", "w1"], ["l1"]), helper.make_node("Add", ["l1", "b1"], ["z1"]),
        helper.make_node("Softmax", ["z1"], ["primary"], axis=1),
        helper.make_node("MatMul", ["s", "w2"], ["l2"]), helper.make_node("Add", ["l2", "b2"], ["z2"]),
        helper.make_node("Sigmoid", ["z2"], ["secondary"]),
        helper.make_node("MatMul", ["s", "w3"], ["l3"]), helper.make_node("Add", ["l3", "b3"], ["z3"]),
        helper.make_node("Squeeze", ["z3", "axis1"], ["z3s"]), helper.make_node("Sigmoid", ["z3s"], ["urgent"]),
    ]
    ints = [helper.make_tensor_value_info(n, TensorProto.INT64, ["batch", "seq"]) for n in ("input_ids", "attention_mask")]
    outs = [helper.make_tensor_value_info("primary", TensorProto.FLOAT, ["batch", 11]),
            helper.make_tensor_value_info("secondary", TensorProto.FLOAT, ["batch", 11]),
            helper.make_tensor_value_info("urgent", TensorProto.FLOAT, ["batch"])]
    model = helper.make_model(helper.make_graph(nodes, "fake", ints, outs, init), opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8  # what torch's exporter writes; newer onnx defaults exceed what onnxruntime reads
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    d = tmp_path_factory.mktemp("bundle")
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, **{w: i + 4 for i, w in enumerate(WORDS)}}
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    (d / "tokenizer").mkdir()
    tok.save(str(d / "tokenizer" / "tokenizer.json"))
    fake_graph(d / "model_b.onnx")
    joblib.dump(Preprocessor(), d / "preprocessor.joblib")
    (d / "meta.json").write_text(json.dumps({"encoder": "fake", "prefix": "query: ", "view": "denoised", "max_len": 16}))
    (d / "model_card.json").write_text(json.dumps({
        "model_version": "v-test-onnx", "kind": "onnx_multitask",
        "thresholds": {"secondary": 0.5, "urgent": 0.5}, "review_below_confidence": 0.8}))
    return d


ROWS = [{"channel": "chat", "subject": "", "text": t} for t in
        ("my bag", "food late refund", "driver scared", "app crash app crash app crash", "😡", "my food")]


def test_probabilities_have_the_right_shape(bundle):
    p = OnnxModel(bundle).predict_proba(ROWS)
    assert p["primary"].shape == (6, 11) and p["secondary"].shape == (6, 11) and p["urgent"].shape == (6,)
    assert np.allclose(p["primary"].sum(1), 1.0, atol=1e-5)


def test_predictions_do_not_depend_on_batch_neighbours_or_position(bundle):
    m = OnnxModel(bundle)
    together = m.predict_proba(ROWS)
    reversed_ = m.predict_proba(ROWS[::-1])
    for i, row in enumerate(ROWS):
        alone = m.predict_proba([row])
        for head in ("primary", "secondary", "urgent"):
            assert np.array_equal(alone[head][0], together[head][i])
            assert np.array_equal(reversed_[head][len(ROWS) - 1 - i], together[head][i])


def test_long_text_is_truncated_not_rejected(bundle):
    p = OnnxModel(bundle).predict_proba([{"channel": "email", "subject": "s", "text": "food " * 5000}])
    assert p["primary"].shape == (1, 11)


def test_predictor_serves_onnx_bundle_with_review_flag(bundle):
    pred = Predictor(bundle)
    pred._load()
    assert pred.ready and pred.version == "v-test-onnx" and pred.error is None
    out = pred.predict([{"ticket_id": "T-1", "channel": "chat", "text": "my bag"}])[0]
    assert schema_errors("PredictResponse", out) == []
    assert out["needs_human_review"] is (out["confidence"] < 0.8)


def test_api_end_to_end_with_onnx_bundle(bundle, tmp_path):
    settings = Settings(api_key=KEY, model_dir=bundle, data_dir=tmp_path)
    with TestClient(create_app(settings)) as c:
        for _ in range(200):
            if c.get("/health").status_code == 200:
                break
            import time
            time.sleep(0.02)
        health = c.get("/health").json()
        assert health == {"status": "ok", "model_version": "v-test-onnx", "model_loaded": True}
        r = c.post("/predict/batch", json={"tickets": [{"ticket_id": f"T-{i}", **r} for i, r in enumerate(ROWS)]},
                   headers={"X-API-Key": KEY})
        assert r.status_code == 200 and schema_errors("BatchResponse", r.json()) == []
        single = c.post("/predict", json={"ticket_id": "T-1", **ROWS[1]}, headers={"X-API-Key": KEY}).json()
        assert single == r.json()["predictions"][1]  # same answer alone or in a batch


def test_broken_bundle_keeps_health_503(tmp_path):
    (tmp_path / "model_card.json").write_text(json.dumps({"model_version": "x", "kind": "onnx_multitask"}))
    pred = Predictor(tmp_path)
    pred._load()
    assert pred.ready is False and "FileNotFoundError" in pred.error


def started_client(settings):
    import time
    c = TestClient(create_app(settings))
    c.__enter__()
    for _ in range(200):
        if c.get("/health").status_code == 200:
            break
        time.sleep(0.02)
    return c


def test_explain_and_model_card(bundle, tmp_path):
    c = started_client(Settings(api_key=KEY, model_dir=bundle, data_dir=tmp_path, web_dir=tmp_path / "no-site"))
    try:
        plain = c.post("/predict", json=ROWS[1], headers={"X-API-Key": KEY}).json()
        r = c.post("/predict?explain=true", json=ROWS[1], headers={"X-API-Key": KEY})
        body = r.json()
        assert r.status_code == 200 and schema_errors("PredictResponse", body) == []
        assert "".join(s["text"] for s in body["explain"]["segments"]) == ROWS[1]["text"]
        assert body["explain"]["model_input"].startswith("__chat__")
        assert {k: v for k, v in body.items() if k != "explain"} == plain  # explain never changes the prediction
        assert "explain" not in plain
        card = c.get("/model-card")
        assert card.status_code == 200 and card.json()["model_version"] == "v-test-onnx"
        assert c.get("/").status_code == 404 and c.get("/").json()["error"]["code"] == "not_found"  # no site built
    finally:
        c.__exit__(None, None, None)


def test_website_is_served_without_changing_the_api(bundle, tmp_path):
    site = tmp_path / "dist"
    (site / "assets").mkdir(parents=True)
    (site / "index.html").write_text("<!doctype html><title>Sentinel</title>")
    (site / "assets" / "app.js").write_text("console.log(1)")
    c = started_client(Settings(api_key=KEY, model_dir=bundle, data_dir=tmp_path, web_dir=site))
    try:
        assert c.get("/").status_code == 200 and "<title>Sentinel" in c.get("/").text
        assert c.get("/ui/assets/app.js").status_code == 200
        missing = c.get("/ui/assets/nope.js")
        assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"
        assert c.get("/totally-unknown").json()["error"]["code"] == "not_found"
        assert c.post("/predict", json=ROWS[0]).status_code == 401  # website does not open the API
    finally:
        c.__exit__(None, None, None)
