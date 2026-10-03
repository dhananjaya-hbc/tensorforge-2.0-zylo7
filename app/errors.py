"""Every error leaves the service in the spec's ErrorResponse shape: {"error": {"code", "message", "details?"}}."""

from fastapi.responses import JSONResponse


def error(status: int, code: str, message: str, details: list[dict] | None = None, headers: dict | None = None) -> JSONResponse:
    body = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body, headers=headers)


def unauthorized(message: str) -> JSONResponse:
    return error(401, "unauthorized", message, headers={"WWW-Authenticate": "Bearer"})


def validation_failed(details: list[dict]) -> JSONResponse:
    return error(422, "validation_error", "Request failed validation.", details)


STATUS_CODES = {
    400: ("bad_request", "Bad request."),
    401: ("unauthorized", "Missing or invalid API key."),
    404: ("not_found", "Resource not found."),
    405: ("method_not_allowed", "Method not allowed for this resource."),
    413: ("payload_too_large", "Request body exceeds the maximum allowed size."),
    415: ("unsupported_media_type", "Content-Type must be application/json."),
    422: ("validation_error", "Request failed validation."),
    429: ("too_many_requests", "Too many requests."),
    500: ("internal_error", "Internal server error."),
    503: ("service_unavailable", "Service unavailable."),
}


def from_status(status: int, headers: dict | None = None) -> JSONResponse:
    code, message = STATUS_CODES.get(status, ("error", "Request failed."))
    return error(status, code, message, headers=headers)
