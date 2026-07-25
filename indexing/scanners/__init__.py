"""Language capability metadata used by the native index and API."""

from .language_config import (
    LANGUAGE_EXTENSIONS,
    Language,
    LanguageMetadata,
    get_all_dependency_file_patterns,
    get_extensions_for_languages,
    get_language_for_extension,
    get_language_metadata,
    get_supported_languages,
)

__all__ = [
    "LANGUAGE_EXTENSIONS",
    "Language",
    "LanguageMetadata",
    "get_all_dependency_file_patterns",
    "get_extensions_for_languages",
    "get_language_for_extension",
    "get_language_metadata",
    "get_supported_languages",
]
