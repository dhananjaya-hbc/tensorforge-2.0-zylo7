"""Generates notebooks/03_final_model_training.ipynb: the production recipe for the shipped model.

Settings are fixed from the bake-off evidence in experiments/model_b_bakeoff (LaBSE, 5 epochs).
Runs on Colab or Kaggle (GPU). On Colab the bundle is saved to Google Drive so the cloud server can
download it directly (the large file never has to pass through a slow home connection).
"""

from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))  # noqa: E731

md("""
# 03 · Final model training: LaBSE (production recipe)

This notebook builds **the exact model that the API serves**. Its settings are not searched here; they are fixed
from the evidence in `notebooks/02_encoder_bakeoff.ipynb` / `experiments/model_b_bakeoff/`:

| Encoder | Best epoch (composite) | Primary macro-F1 | Secondary F1 | Urgent F1 | Composite |
|---|---|---|---|---|---|
| **LaBSE** | **5** | 0.908 | 0.933 | 0.956 | **0.927** |
| e5-base | 5 | 0.839 | 0.868 | 0.962 | 0.882 |
| XLM-R base | 5 | 0.846 | 0.819 | 0.856 | 0.843 |
| e5-small | 5 | 0.810 | 0.324 | 0.917 | 0.745 |

Steps: **(1)** train on `train` only and score `validation` (the honest estimate recorded in the model card),
**(2)** refit the same recipe on train + validation, **(3)** export int8 ONNX with a parity check,
**(4)** write the model card and save the bundle (Colab: Google Drive).

**Run (about 15 minutes):** Colab: Runtime → Change runtime type → **T4 GPU** → Run all (allow Drive access when asked).
Kaggle: Accelerator **GPU** + Internet **On** → Run All (download the bundle from the Output panel).
""")

code("""
import os, shutil, subprocess
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "/content"
ON_COLAB = WORK == "/content"
REPO = "github.com/dhananjaya-hbc/tensorforge-2.0-zylo7.git"
!nvidia-smi -L

def github_token():  # only needed if the repo is private
    try:
        from google.colab import userdata
        return userdata.get("GITHUB_TOKEN")
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception:
        return os.environ.get("GITHUB_TOKEN")

token = github_token()
shutil.rmtree(f"{WORK}/repo", ignore_errors=True)
url = f"https://x-access-token:{token}@{REPO}" if token else f"https://{REPO}"
clone = subprocess.run(["git", "clone", "--depth", "1", url, f"{WORK}/repo"], capture_output=True, text=True,
                       env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
if clone.returncode != 0:
    msg = clone.stderr.replace(token, "***") if token else clone.stderr
    raise RuntimeError("git clone failed. If the repo is private, add the GITHUB_TOKEN secret.\\n" + msg)
os.chdir(f"{WORK}/repo")
subprocess.run(["git", "remote", "set-url", "origin", f"https://{REPO}"], check=True)
del token, url
COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
print("code commit:", COMMIT)
!pip install -q onnx onnxruntime
""")

md("## Recipe (fixed from the bake-off)")
code("""
ENCODER = "sentence-transformers/LaBSE"
EPOCHS = 5
VERSION = "v1.0"
REVIEW_BELOW = 0.80   # needs_human_review when confidence < this (validation: ~16% reviewed, ~97% accurate on the rest)
THRESHOLDS = {"secondary": 0.5, "urgent": 0.5}  # tuned thresholds overfit in cross-fitting, defaults kept
RECIPE = f"--encoder {ENCODER} --epochs {EPOCHS}"
""")

md("## 1 · Honest estimate: train only → validation")
code("""
get_ipython().system(f"python scripts/train_transformer.py {RECIPE} --out runs/check")
import json
check = json.load(open("runs/check/metrics.json"))
last = check["history"][-1]
print({k: last[k] for k in ("primary_accuracy", "primary_macro_f1", "secondary_f1", "urgent_f1", "ece", "composite")})
""")

md("## 2 · Refit on train + validation (the shipped weights)")
code("""
get_ipython().system(f"python scripts/train_transformer.py {RECIPE} --train-on train+validation --save --out runs/final")
""")

md("## 3 · Export int8 ONNX + parity check")
code("""
BUNDLE = f"bundle_labse_{VERSION}"
shutil.rmtree(BUNDLE, ignore_errors=True)
get_ipython().system(f"python scripts/export_onnx.py --run runs/final --out {BUNDLE}")
""")

md("## 4 · Model card")
code("""
import hashlib, datetime
digest = hashlib.sha256(open(f"{BUNDLE}/model_b.onnx", "rb").read()).hexdigest()[:8]
card = {
    "model_version": f"{VERSION}-{digest}",
    "kind": "onnx_multitask",
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "code_commit": COMMIT,
    "encoder": ENCODER,
    "epochs": EPOCHS,
    "trained_on": "train + validation (4,800 tickets)",
    "thresholds": THRESHOLDS,
    "review_below_confidence": REVIEW_BELOW,
    "validation_estimate": {"protocol": "same recipe trained on train only, scored on validation", **last,
                            "by_language": check["validation_by_language"]},
    "onnx_parity": json.load(open(f"{BUNDLE}/parity.json")),
    "files": {"model_b.onnx": digest},
}
json.dump(card, open(f"{BUNDLE}/model_card.json", "w"), indent=2)
print(json.dumps({k: card[k] for k in ("model_version", "encoder", "epochs", "onnx_parity")}, indent=2))
!ls -lh {BUNDLE}
""")

md("## 5 · Save the bundle (Colab → Google Drive · Kaggle → Output panel)")
code("""
archive = shutil.make_archive(f"{WORK}/{BUNDLE}", "zip", BUNDLE)
print("archive:", archive, f"{os.path.getsize(archive) / 1e6:.0f} MB")
if ON_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    dest = "/content/drive/MyDrive/tensorforge"
    os.makedirs(dest, exist_ok=True)
    shutil.copy(archive, dest)
    shutil.copy(f"{BUNDLE}/model_card.json", f"{dest}/{BUNDLE}_model_card.json")
    print(f"saved to Google Drive: MyDrive/tensorforge/{os.path.basename(archive)}")
    print("Next: in Drive, right-click the zip -> Share -> 'Anyone with the link' -> Copy link.")
else:
    print("Kaggle: download the zip from the Output panel.")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, Path(__file__).resolve().parent.parent / "notebooks" / "03_final_model_training.ipynb")
print("written notebooks/03_final_model_training.ipynb")
