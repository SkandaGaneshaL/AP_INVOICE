from __future__ import annotations

import time
import uuid
from dataclasses import dataclass


@dataclass
class StoredDocument:
    document_id: str
    filename: str
    mime_type: str
    document_bytes: bytes
    expires_at: float


class InMemoryDocumentStore:
    def __init__(self, ttl_seconds: int = 1800):
        self.ttl_seconds = ttl_seconds
        self._documents: dict[str, StoredDocument] = {}

    def put(self, document_bytes: bytes, filename: str, mime_type: str = "application/pdf") -> str:
        self.cleanup()
        document_id = str(uuid.uuid4())
        self._documents[document_id] = StoredDocument(document_id, filename, mime_type, document_bytes,
                                                       time.time() + self.ttl_seconds)
        return document_id

    def get(self, document_id: str) -> StoredDocument:
        self.cleanup()
        try:
            return self._documents[document_id]
        except KeyError as exc:
            raise KeyError(f"document {document_id} was not found or expired") from exc

    def delete(self, document_id: str) -> None:
        self._documents.pop(document_id, None)

    def cleanup(self) -> None:
        now = time.time()
        for key, value in list(self._documents.items()):
            if value.expires_at <= now:
                del self._documents[key]
