# Zylo7 Sentinel: production image (CPU only, no torch, no internet needed at runtime).
#
# The model bundle must be present in models/active/ before building (scripts/fetch_model.sh downloads it).
#   docker build -t sentinel .
#   docker run -p 8000:8000 -e API_KEY=<key> sentinel

# ---------- web stage: build the React demo site (only its static output reaches the final image) ----------
FROM node:24-slim AS web
RUN npm install -g pnpm@12.4.2
WORKDIR /web
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm build

# ---------- build stage: resolve runtime dependencies from the lockfile ----------
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
# Runtime group only: no dev/test/training packages in the image.
RUN uv export --frozen --no-dev --no-hashes --no-emit-project -o requirements.txt \
 && python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---------- runtime stage ----------
FROM python:3.12-slim
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models/active \
    WEB_DIR=/app/web \
    DATA_DIR=/app/var \
    HF_HUB_OFFLINE=1
RUN useradd --create-home --uid 10001 sentinel
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY app/ app/
COPY ml/ ml/
COPY specs/ specs/
COPY models/active/ models/active/
COPY --from=web /web/dist /app/web
RUN mkdir -p /app/var && chown -R sentinel:sentinel /app/var
USER sentinel
EXPOSE 8000
# /health returns 200 once the model is loaded (503 while loading); the spec allows up to 120 s.
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "75"]
