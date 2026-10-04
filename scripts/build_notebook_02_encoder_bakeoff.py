"""Generates notebooks/02_encoder_bakeoff.ipynb, the GPU training notebook for Model B.

Works on Kaggle and Google Colab (the output folder is detected automatically):
- Kaggle: New Notebook -> File -> Import Notebook, Settings -> Accelerator GPU T4 + Internet On, Run All.
  Outputs: /kaggle/working/results.zip and bundle_b.zip (Output panel).
- Colab: File -> Upload notebook, Runtime -> Change runtime type -> T4 GPU, Runtime -> Run all.
  Outputs: /content/results.zip and bundle_b.zip (downloaded automatically at the end).
"""

from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))  # noqa: E731

md("""
# 02 · Encoder bake-off: choosing Model B (Kaggle / Colab GPU)

**Why this model exists.** Validation tickets describe the same issues with *different phrasings* than train
(only ~20% of their phrases occur in train). Word-matching models (Model A, TF-IDF) reach ~0.65 macro-F1 on
validation, so we fine-tune multilingual encoders that represent *meaning* across English, Sinhala, Tamil,
Singlish and Tanglish.

**Protocol.** Train on `train`, select on `validation` (train → validation is the honest estimate for unseen
phrasings). The chosen setup is then refit on train + validation and exported to int8 ONNX for CPU serving.

**Run (about 1 hour):**
- **Kaggle:** Settings → Accelerator **GPU T4** and Internet **On**, then *Run All*.
- **Colab:** Runtime → Change runtime type → **T4 GPU**, then Runtime → *Run all*. Keep the tab open.

**Private repo:** add a read-only GitHub token as a secret named `GITHUB_TOKEN`
(Colab: 🔑 *Secrets* in the left sidebar, enable *Notebook access* · Kaggle: *Add-ons → Secrets*).
The token is never printed and is removed from the clone's git config.
""")

code("""
import os, shutil, subprocess
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "/content"  # Kaggle or Colab
REPO = "github.com/dhananjaya-hbc/tensorforge-2.0-zylo7.git"
print("output folder:", WORK)
!nvidia-smi -L

def github_token():
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
print("GitHub token found:", bool(token))
shutil.rmtree(f"{WORK}/repo", ignore_errors=True)
url = f"https://x-access-token:{token}@{REPO}" if token else f"https://{REPO}"
clone = subprocess.run(["git", "clone", "--depth", "1", url, f"{WORK}/repo"], capture_output=True, text=True,
                       env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
if clone.returncode != 0:
    msg = clone.stderr.replace(token, "***") if token else clone.stderr
    raise RuntimeError("git clone failed. If the repo is private, add the GITHUB_TOKEN secret.\\n" + msg)
os.chdir(f"{WORK}/repo")
subprocess.run(["git", "remote", "set-url", "origin", f"https://{REPO}"], check=True)  # drop the token
del token, url
!pip install -q onnx onnxruntime
!git log --oneline -1

import sys, json, os, numpy as np, pandas as pd
sys.path.insert(0, ".")
from ml.data import load_split
from ml.decide import decide, Thresholds
from ml.evaluate import score
os.makedirs("results", exist_ok=True)
train, val = load_split("train"), load_split("validation")
print(len(train), len(val))
""")

md("## 1 · Reference: Model A (TF-IDF) on validation")
code("""
from ml.linear import LinearModel
A = LinearModel().fit(train).predict_proba(val)
np.savez("results/model_a_val.npz", **A)
ref = score(val, decide(**A, th=Thresholds(0.5, 0.5)))
print({k: ref[k] for k in ("primary_accuracy", "primary_macro_f1", "secondary_f1", "urgent_f1")})
""")

md("""
## 2 · Encoder bake-off (train → validation)
Each encoder is fine-tuned with four heads: primary, secondary, urgency and an auxiliary *language* head
(training only; the API never sends the language).
""")
code("""
ENCODERS = {
    "e5-small": "intfloat/multilingual-e5-small",
    "e5-base": "intfloat/multilingual-e5-base",
    "labse": "sentence-transformers/LaBSE",
    "xlmr-base": "FacebookAI/xlm-roberta-base",
}
EPOCHS = 5
for name, enc in ENCODERS.items():
    print(f"===== {name}")
    get_ipython().system(f"python scripts/train_transformer.py --encoder {enc} --epochs {EPOCHS} --out runs/{name}")
""")

md("## 3 · Results: each encoder alone and blended with Model A")
code("""
rows, best_epoch = [], {}
for name in ENCODERS:
    hist = json.load(open(f"runs/{name}/metrics.json"))["history"]
    # composite = 0.5 primary macro-F1 + 0.2 secondary F1 + 0.3 urgent F1 (primary alone favours epoch 1,
    # where the secondary head has not converged yet)
    e = max(hist, key=lambda h: h["composite"])["epoch"]
    best_epoch[name] = e
    B = dict(np.load(f"runs/{name}/val_probs_e{e}.npz"))
    blends = []
    for w in np.round(np.linspace(0, 1, 11), 1):
        P = {h: w * B[h] + (1 - w) * A[h] for h in A}
        blends.append((w, score(val, decide(**P, th=Thresholds(0.5, 0.5)))))
    w, s = max(blends, key=lambda x: x[1]["composite"])
    alone = score(val, decide(**B, th=Thresholds(0.5, 0.5)))
    rows.append({"encoder": name, "best_epoch": e, "alone_composite": alone["composite"],
                 "alone_macro_f1": alone["primary_macro_f1"], "alone_acc": alone["primary_accuracy"],
                 "blend_weight_B": w, "blend_composite": s["composite"], "blend_macro_f1": s["primary_macro_f1"],
                 "blend_secondary_f1": s["secondary_f1"], "blend_urgent_f1": s["urgent_f1"]})
table = pd.DataFrame(rows).sort_values("blend_composite", ascending=False)
table.to_csv("results/bakeoff.csv", index=False)
table
""")

md("""
## 4 · Final Model B: refit on train + validation, export int8 ONNX
The best encoder by blended validation composite score is chosen automatically (set `CHOICE` to override).
""")
code("""
CHOICE = None  # e.g. "e5-base" to override
best = CHOICE or table.iloc[0]["encoder"]
print("final encoder:", best, "| epochs:", best_epoch[best])
get_ipython().system(f"python scripts/train_transformer.py --encoder {ENCODERS[best]} --epochs {best_epoch[best]} "
                     f"--train-on train+validation --save --out runs/final")
get_ipython().system("python scripts/export_onnx.py --run runs/final --out bundle_b")
""")

md("## 5 · Package outputs (Kaggle: Output panel · Colab: downloads start automatically)")
code("""
import shutil, glob
for name in ENCODERS:
    os.makedirs(f"results/{name}", exist_ok=True)
    for f in glob.glob(f"runs/{name}/*.json") + glob.glob(f"runs/{name}/*.npz"):
        shutil.copy(f, f"results/{name}/")
shutil.copy("runs/final/metrics.json", "bundle_b/final_train_metrics.json")
shutil.make_archive(f"{WORK}/results", "zip", "results")
shutil.make_archive(f"{WORK}/bundle_b", "zip", "bundle_b")
!ls -lh {WORK}/*.zip
try:  # Colab: trigger browser downloads; on Kaggle use the Output panel instead
    from google.colab import files
    files.download(f"{WORK}/results.zip")
    files.download(f"{WORK}/bundle_b.zip")
except ImportError:
    print("Kaggle: download results.zip and bundle_b.zip from the Output panel")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, Path(__file__).resolve().parent.parent / "notebooks" / "02_encoder_bakeoff.ipynb")
print("written notebooks/02_encoder_bakeoff.ipynb")
