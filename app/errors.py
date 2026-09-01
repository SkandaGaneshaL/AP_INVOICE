from .model_output import ModelOutputError


def status_code_for_error(exc: Exception) -> int:
    if isinstance(exc, ModelOutputError):
        return 502
    status = getattr(exc, "status", None)
    if status in {400, 401, 403, 404, 408, 429, 500, 502, 503}:
        return 502 if status >= 500 or status == 408 else status
    if isinstance(exc, (ValueError, KeyError)):
        return 422
    return 500

