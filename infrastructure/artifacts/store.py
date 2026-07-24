"""Content-addressed filesystem storage for large immutable bodies."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    artifact_id: str
    content_hash: str
    artifact_kind: str
    storage_path: str
    byte_size: int
    media_type: str | None = None


class FilesystemArtifactStore:
    """Store bytes once while SQLite retains all application metadata."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        content: bytes,
        *,
        artifact_kind: str,
        media_type: str | None = None,
    ) -> StoredArtifact:
        content_hash = hashlib.sha256(content).hexdigest()
        artifact_id = hashlib.sha256(
            f"{artifact_kind}\0{content_hash}".encode()
        ).hexdigest()
        destination = self.root / content_hash[:2] / content_hash[2:4] / content_hash
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
            try:
                temporary.write_bytes(content)
                try:
                    os.replace(temporary, destination)
                except FileExistsError:
                    temporary.unlink(missing_ok=True)
            finally:
                temporary.unlink(missing_ok=True)
        return StoredArtifact(
            artifact_id=artifact_id,
            content_hash=content_hash,
            artifact_kind=artifact_kind,
            storage_path=str(destination),
            byte_size=len(content),
            media_type=media_type,
        )

    def read(self, artifact: StoredArtifact) -> bytes:
        content = Path(artifact.storage_path).read_bytes()
        if hashlib.sha256(content).hexdigest() != artifact.content_hash:
            raise ValueError(f"artifact integrity check failed: {artifact.artifact_id}")
        return content
