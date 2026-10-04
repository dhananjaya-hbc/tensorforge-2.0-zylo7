#!/usr/bin/env bash
# Download the model bundle zip into models/active/ and verify it against its model card.
#
#   MODEL_URL="https://drive.google.com/file/d/<id>/view?usp=sharing" scripts/fetch_model.sh
#
# Google Drive links are fetched with gdown inside a throwaway Docker container (nothing installed on the
# host); any other http(s) URL is fetched with curl. The bundle is too large for git (> 100 MB).
set -euo pipefail

: "${MODEL_URL:?set MODEL_URL to the bundle zip link}"
cd "$(dirname "$0")/.."
mkdir -p models
rm -f models/bundle.zip

if [[ "$MODEL_URL" == *drive.google.com* ]]; then
  docker run --rm -v "$PWD/models:/out" python:3.12-slim \
    sh -c 'pip install -q gdown && gdown --fuzzy "$0" -O /out/bundle.zip' "$MODEL_URL"
else
  curl -fL --retry 5 -o models/bundle.zip "$MODEL_URL"
fi

rm -rf models/active
mkdir -p models/active
unzip -q models/bundle.zip -d models/active
rm -f models/bundle.zip

expected=$(python3 -c 'import json; print(json.load(open("models/active/model_card.json"))["files"]["model_b.onnx"])')
actual=$(sha256sum models/active/model_b.onnx | cut -c1-8)
if [[ "$expected" != "$actual" ]]; then
  echo "checksum mismatch: model card says $expected, file is $actual" >&2
  exit 1
fi
echo "model bundle OK: $(python3 -c 'import json; print(json.load(open("models/active/model_card.json"))["model_version"])')"
