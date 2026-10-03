"""Pure ASGI gatekeeper, applied before any routing or body parsing.

Order mandated by the spec for prediction and job endpoints:
    auth (401) -> content type (415) -> size (413) -> [handler: JSON parse (400) -> validation (422)]
An unauthorized request is rejected before its body is read, so it can never reach the model.
"""

import hmac
import json

from app.config import Settings

PROTECTED_PREFIXES = ("/predict", "/batch/")


def is_protected(path: str) -> bool:
    return path == "/predict" or path.startswith("/predict/") or path == "/batch/jobs" or path.startswith("/batch/")


def body_limit(path: str, settings: Settings) -> int:
    if path == "/predict":
        return settings.max_body_predict
    if path == "/predict/batch":
        return settings.max_body_batch
    return settings.max_body_jobs


async def _send_error(send, status: int, code: str, message: str, extra_headers=(), request_id: str | None = None):
    body = json.dumps({"error": {"code": code, "message": message}}).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()), *extra_headers]
    if request_id:
        headers.append((b"x-request-id", request_id.encode("latin-1", "replace")))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


def _header(scope, name: bytes) -> str | None:
    for k, v in scope["headers"]:
        if k.lower() == name:
            return v.decode("latin-1")
    return None


class Gatekeeper:
    def __init__(self, app, settings: Settings):
        self.app = app
        self.settings = settings
        self._key = settings.api_key.encode() if settings.api_key else None

    def _authorized(self, scope) -> tuple[bool, str]:
        if self._key is None:
            return False, "Service has no API key configured; prediction endpoints are disabled."
        candidates = []
        if (k := _header(scope, b"x-api-key")) is not None:
            candidates.append(k.strip())
        if (auth := _header(scope, b"authorization")) is not None:
            scheme, _, token = auth.strip().partition(" ")
            if scheme.lower() == "bearer":
                candidates.append(token.strip())
        if not candidates:
            return False, "Missing API key. Send X-API-Key or Authorization: Bearer."
        # Compare every candidate in constant time; never short-circuit on a partial match.
        ok = False
        for c in candidates:
            ok |= hmac.compare_digest(c.encode("utf-8", "replace"), self._key)
        return ok, "Invalid API key."

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        request_id = _header(scope, b"x-request-id")
        if request_id is not None:
            request_id = request_id[:128]

            async def send_with_id(message, _send=send):
                if message["type"] == "http.response.start":
                    headers = [h for h in message.get("headers", []) if h[0].lower() != b"x-request-id"]
                    headers.append((b"x-request-id", request_id.encode("latin-1", "replace")))
                    message = {**message, "headers": headers}
                await _send(message)

            send = send_with_id

        path, method = scope["path"], scope["method"]
        if not is_protected(path):
            return await self.app(scope, receive, send)

        ok, message = self._authorized(scope)
        if not ok:
            return await _send_error(send, 401, "unauthorized", message, [(b"www-authenticate", b"Bearer")])

        if method != "POST":
            return await self.app(scope, receive, send)

        ctype = (_header(scope, b"content-type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            return await _send_error(send, 415, "unsupported_media_type", "Content-Type must be application/json.")

        limit = body_limit(path, self.settings)
        too_large = (413, "payload_too_large", "Request body exceeds the maximum allowed size.")
        declared = _header(scope, b"content-length")
        if declared and declared.strip().isdigit() and int(declared) > limit:
            return await _send_error(send, *too_large)

        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > limit:
                return await _send_error(send, *too_large)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
