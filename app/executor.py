from __future__ import annotations

import importlib
import os
from typing import Any


class CallableExtractionExecutor:
    def __init__(self, function):
        self.function = function

    def extract(self, *, document_bytes: bytes, mime_type: str, field_key: str,
                field_path: str, instruction: str, rules: list[str]) -> Any:
        return self.function(document_bytes=document_bytes, mime_type=mime_type, field_key=field_key,
                             field_path=field_path, instruction=instruction, rules=rules)


def load_extraction_executor():
    """Load the real extractor as MODULE:FUNCTION when configured."""
    reference = os.getenv("EXTRACTION_EXECUTOR", "").strip()
    if not reference or ":" not in reference:
        return None
    module_name, function_name = reference.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    return CallableExtractionExecutor(function)
