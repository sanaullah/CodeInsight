"""
Project scanning utilities for CodeLumen v3.
"""

from .project_scanner import ProjectScanner, FileInfo
from .file_reader import read_file_with_encoding
from .language_config import (
    Language,
    LanguageMetadata,
    get_extensions_for_languages,
    get_language_for_extension,
    get_supported_languages,
    get_language_metadata,
    get_all_dependency_file_patterns,
)
from .language_detector import LanguageDetector
from .scanner_factory import create_project_scanner

__all__ = [
    "ProjectScanner",
    "FileInfo",
    "read_file_with_encoding",
    "Language",
    "LanguageMetadata",
    "get_extensions_for_languages",
    "get_language_for_extension",
    "get_supported_languages",
    "get_language_metadata",
    "get_all_dependency_file_patterns",
    "LanguageDetector",
    "create_project_scanner",
]

