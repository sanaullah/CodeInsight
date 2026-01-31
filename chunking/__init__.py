"""
Chunking system for grouping files into token-based chunks.

Provides ChunkManager for managing file chunking with different strategies.
"""

from .chunk_manager import ChunkManager, Chunk, FileChunk, ChunkingStrategy
from .token_counter import TokenCounter

__all__ = [
    "ChunkManager",
    "Chunk",
    "FileChunk",
    "ChunkingStrategy",
    "TokenCounter",
]

