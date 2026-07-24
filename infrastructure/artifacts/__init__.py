"""Immutable content-addressed analysis artifacts."""

from .store import FilesystemArtifactStore, StoredArtifact

__all__ = ["FilesystemArtifactStore", "StoredArtifact"]
