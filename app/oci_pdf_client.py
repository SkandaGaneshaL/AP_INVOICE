from __future__ import annotations

import base64
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from .models import TokenUsage
from .usage import normalize_provider_usage

load_dotenv()


@dataclass(frozen=True)
class PdfModelResponse:
    text: str
    request_id: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    model: str | None = None


class ExtractionConfigurationError(RuntimeError):
    """Safe, typed configuration failure for PDF extraction."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _project_region(project_id: str | None) -> str | None:
    if not project_id:
        return None
    match = re.search(r"\.oc1\.([a-z0-9-]+)\.", project_id)
    return match.group(1) if match else None


def _httpx2_oci_auth(signer: Any) -> Any:
    """Build an actual httpx2.Auth subclass around an OCI request signer."""
    try:
        import httpx2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("The installed OpenAI client requires httpx2") from exc

    class Httpx2OciAuth(httpx2.Auth):
        def auth_flow(self, request):
            import requests

            content = request.content
            prepared = requests.Request(
                method=request.method,
                url=str(request.url),
                headers=dict(request.headers),
                data=content,
            ).prepare()
            signer.do_request_sign(prepared)
            request.headers.update(prepared.headers)
            yield request

    return Httpx2OciAuth()


class OciPdfClient:
    """Small OCI OpenAI-compatible client used only for PDF extraction."""

    def __init__(self, client: Any = None, *, model_id: str | None = None):
        self.region = os.getenv("OCI_EXTRACTION_REGION") or os.getenv("OCI_EXC_REGION") or os.getenv("OCI_REGION")
        if not self.region:
            raise ExtractionConfigurationError("OCI_EXTRACTION_REGION_MISSING", "No OCI region is configured for PDF extraction")
        self.compartment_id = os.getenv("OCI_EXC_COMPARTMENT_ID") or os.getenv("OCI_COMPARTMENT_ID")
        if not self.compartment_id:
            raise ExtractionConfigurationError("OCI_EXTRACTION_COMPARTMENT_MISSING", "OCI_EXC_COMPARTMENT_ID or OCI_COMPARTMENT_ID must be configured")
        self.model_id = model_id or os.getenv("OCI_EXTRACTION_MODEL_ID", "google.gemini-2.5-flash")
        self.project_id = os.getenv("OCI_EXTRACTION_PROJECT_ID") or os.getenv("PROJECT_ID")
        if client is None:
            self._validate_project()
        self.client = client or self._build_client()

    def _validate_project(self) -> None:
        if not self.project_id:
            raise ExtractionConfigurationError("OCI_EXTRACTION_PROJECT_MISSING", "No Generative AI project is configured for PDF extraction")
        project_region = _project_region(self.project_id)
        if project_region and project_region != self.region:
            raise ExtractionConfigurationError("OCI_EXTRACTION_PROJECT_REGION_MISMATCH", "The extraction project region does not match the extraction region")

    def _build_client(self) -> Any:
        try:
            from oci_openai import OciOpenAI, OciUserPrincipalAuth
        except ImportError as exc:  # pragma: no cover - exercised in deployment environments
            raise RuntimeError("Install the `oci-openai` package to enable PDF extraction") from exc

        auth = OciUserPrincipalAuth(
            config_file=os.getenv("OCI_CONFIG_FILE", ".oci/config"),
            profile_name=os.getenv("OCI_PROFILE", "DEFAULT"),
        )
        kwargs = {
            "region": self.region,
            "auth": auth,
            "compartment_id": self.compartment_id,
            "project": self.project_id,
        }
        try:
            return OciOpenAI(**kwargs)
        except TypeError as exc:
            if "Invalid \"auth\" argument" not in str(exc):
                raise
            # Some oci-openai/openai combinations use different httpx module
            # namespaces. Reuse the OCI signer, but adapt its auth protocol.
            kwargs["auth"] = _httpx2_oci_auth(auth.signer)
            return OciOpenAI(**kwargs)

    @staticmethod
    def _request_id(response: Any) -> str | None:
        headers = getattr(response, "headers", None)
        if isinstance(headers, dict):
            return headers.get("opc-request-id") or headers.get("x-request-id")
        return getattr(response, "request_id", None)

    @staticmethod
    def _output_text(response: Any) -> str:
        text = getattr(response, "output_text", None)
        if text:
            return str(text).strip()
        output = getattr(response, "output", None) or []
        parts: list[str] = []
        for item in output:
            for content in getattr(item, "content", None) or []:
                value = getattr(content, "text", None)
                if value:
                    parts.append(str(value))
        return "\n".join(parts).strip()

    def extract(self, *, document_bytes: bytes, filename: str, prompt: str) -> PdfModelResponse:
        encoded = base64.b64encode(document_bytes).decode("ascii")
        request = {
            "model": self.model_id,
            "store": False,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": filename,
                            "file_data": encoded,
                            "mime_type": "application/pdf",
                        },
                        {"type": "input_text", "text": prompt},
                    ],
                }
            ],
        }

        for attempt in range(3):
            try:
                response = self.client.responses.create(**request)
                text = self._output_text(response)
                if not text:
                    raise RuntimeError("OCI returned an empty extraction response")
                return PdfModelResponse(
                    text=text,
                    request_id=self._request_id(response),
                    finish_reason=getattr(response, "finish_reason", None),
                    usage=normalize_provider_usage(
                        getattr(response, "usage", None), model=self.model_id,
                        request_id=self._request_id(response), call_type="pdf_extraction", attempt=attempt + 1,
                    ),
                    model=self.model_id,
                )
            except Exception as exc:
                status = getattr(exc, "status", None)
                retryable = status in {408, 429, 500, 502, 503} or status is None
                if attempt == 2 or not retryable:
                    raise
                time.sleep((2**attempt) + random.uniform(0, 0.25))

        raise RuntimeError("OCI extraction retries exhausted")
