"""
Chunk manager for grouping files into token-based chunks.
"""

import logging
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from scanners.project_scanner import FileInfo
from .token_counter import TokenCounter

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Chunking strategy enumeration."""
    NONE = "none"  # No chunking, process all files together
    STANDARD = "standard"  # Standard chunking (100K tokens)
    AGGRESSIVE = "aggressive"  # Aggressive chunking (50K tokens)


@dataclass
class FileChunk:
    """Represents a file or part of a file in a chunk."""
    file_info: FileInfo
    start_line: Optional[int] = None  # For split files
    end_line: Optional[int] = None  # For split files
    content: Optional[str] = None  # Extracted content if file is split


@dataclass
class Chunk:
    """A chunk of files grouped together."""
    chunk_index: int
    files: List[FileChunk]
    total_tokens: int
    total_files: int
    total_lines: int


class ChunkManager:
    """
    Manages chunking of files based on token limits.
    
    Groups files into chunks, ensuring each chunk stays within the token limit.
    Handles large files by splitting them if necessary.
    """
    
    def __init__(
        self, 
        max_tokens: int = 100000, 
        model_name: Optional[str] = None,
        chunking_strategy: Optional[ChunkingStrategy] = None
    ):
        """
        Initialize chunk manager.
        
        Args:
            max_tokens: Maximum tokens per chunk (default: 100K)
            model_name: Model name for token counting
            chunking_strategy: Optional chunking strategy (defaults to STANDARD)
        """
        self.max_tokens = max_tokens
        self.token_counter = TokenCounter(model_name=model_name)
        self.chunking_strategy = chunking_strategy or ChunkingStrategy.STANDARD
        
        # Adjust max_tokens based on strategy if not explicitly set
        if chunking_strategy == ChunkingStrategy.AGGRESSIVE and max_tokens == 100000:
            self.max_tokens = 50000  # 50K for aggressive chunking
        elif chunking_strategy == ChunkingStrategy.NONE:
            # For NONE strategy, set a very large max_tokens
            self.max_tokens = max_tokens if max_tokens > 1000000 else 10000000
        
        logger.info(
            f"ChunkManager initialized with max_tokens={self.max_tokens}, "
            f"strategy={self.chunking_strategy.value}"
        )
    
    def create_chunks(self, files: List[FileInfo]) -> List[Chunk]:
        """
        Create chunks from a list of files.
        
        Uses chunking strategy to determine how to chunk files.
        
        Args:
            files: List of FileInfo objects to chunk
        
        Returns:
            List of Chunk objects
        """
        if not files:
            return []
        
        # Handle NONE strategy - create single chunk
        if self.chunking_strategy == ChunkingStrategy.NONE:
            return self._create_single_chunk(files)
        
        # Handle STANDARD and AGGRESSIVE strategies
        chunks: List[Chunk] = []
        current_chunk_files: List[FileChunk] = []
        current_tokens = 0
        chunk_index = 0
        
        for file_info in files:
            # Count tokens for this file
            file_tokens = self.token_counter.count_tokens(file_info.content)
            
            # Check if file fits in current chunk
            if current_tokens + file_tokens <= self.max_tokens:
                # Add to current chunk
                current_chunk_files.append(FileChunk(file_info=file_info))
                current_tokens += file_tokens
            else:
                # Current chunk is full, save it and start new one
                if current_chunk_files:
                    chunk = self._create_chunk(chunk_index, current_chunk_files, current_tokens)
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Handle large files that exceed chunk size
                if file_tokens > self.max_tokens:
                    # Split the file
                    file_chunks = self._split_file(file_info, chunk_index)
                    chunks.extend(file_chunks)
                    chunk_index += len(file_chunks)
                    current_chunk_files = []
                    current_tokens = 0
                else:
                    # Start new chunk with this file
                    current_chunk_files = [FileChunk(file_info=file_info)]
                    current_tokens = file_tokens
        
        # Add final chunk if it has files
        if current_chunk_files:
            chunk = self._create_chunk(chunk_index, current_chunk_files, current_tokens)
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from {len(files)} files using {self.chunking_strategy.value} strategy")
        return chunks
    
    def _create_single_chunk(self, files: List[FileInfo]) -> List[Chunk]:
        """
        Create a single chunk with all files (for small projects).
        
        Args:
            files: List of FileInfo objects
        
        Returns:
            List with single Chunk containing all files
        """
        logger.info(f"Creating single chunk with {len(files)} files (NONE strategy)")
        
        # Create FileChunk objects
        file_chunks = [FileChunk(file_info=f) for f in files]
        
        # Count total tokens
        total_tokens = sum(self.token_counter.count_tokens(f.content) for f in files)
        
        # Create single chunk
        chunk = Chunk(
            chunk_index=0,
            files=file_chunks,
            total_tokens=total_tokens,
            total_files=len(files),
            total_lines=sum(f.line_count for f in files)
        )
        
        return [chunk]
    
    def _create_chunk(self, chunk_index: int, file_chunks: List[FileChunk], 
                     total_tokens: int) -> Chunk:
        """Create a Chunk object from file chunks."""
        total_files = len(file_chunks)
        total_lines = sum(fc.file_info.line_count for fc in file_chunks)
        
        return Chunk(
            chunk_index=chunk_index,
            files=file_chunks,
            total_tokens=total_tokens,
            total_files=total_files,
            total_lines=total_lines
        )
    
    def _split_file(self, file_info: FileInfo, start_chunk_index: int) -> List[Chunk]:
        """
        Split a large file into multiple chunks.
        
        Args:
            file_info: File to split
            start_chunk_index: Starting chunk index
        
        Returns:
            List of Chunk objects
        """
        lines = file_info.content.splitlines(keepends=True)
        chunks: List[Chunk] = []
        current_lines: List[str] = []
        current_tokens = 0
        chunk_index = start_chunk_index
        start_line = 0
        
        for i, line in enumerate(lines):
            line_tokens = self.token_counter.count_tokens(line)
            
            if current_tokens + line_tokens <= self.max_tokens:
                current_lines.append(line)
                current_tokens += line_tokens
            else:
                # Current chunk is full, save it
                if current_lines:
                    content = "".join(current_lines)
                    file_chunk = FileChunk(
                        file_info=file_info,
                        start_line=start_line,
                        end_line=i - 1,
                        content=content
                    )
                    chunk = self._create_chunk(chunk_index, [file_chunk], current_tokens)
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with this line
                current_lines = [line]
                current_tokens = line_tokens
                start_line = i
        
        # Add final chunk if it has lines
        if current_lines:
            content = "".join(current_lines)
            file_chunk = FileChunk(
                file_info=file_info,
                start_line=start_line,
                end_line=len(lines) - 1,
                content=content
            )
            chunk = self._create_chunk(chunk_index, [file_chunk], current_tokens)
            chunks.append(chunk)
        
        logger.info(f"Split file {file_info.relative_path} into {len(chunks)} chunks")
        return chunks

