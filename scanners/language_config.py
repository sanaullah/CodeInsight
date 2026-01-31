"""
Language configuration and definitions for multi-language support.

Provides centralized language definitions, file extension mappings, and language-specific metadata.
"""

from enum import Enum
from typing import Dict, List, Set, Optional
from dataclasses import dataclass


class Language(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    C = "c"
    CPP = "cpp"
    RUBY = "ruby"
    PHP = "php"
    CSHARP = "csharp"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"
    DART = "dart"
    LUA = "lua"
    PERL = "perl"
    R = "r"
    MATLAB = "matlab"
    SHELL = "shell"
    POWERSHELL = "powershell"
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    YAML = "yaml"
    JSON = "json"
    XML = "xml"
    MARKDOWN = "markdown"
    BOXLANG = "boxlang"
    COLDFUSION = "coldfusion"


@dataclass
class LanguageMetadata:
    """Metadata for a programming language."""
    name: str
    extensions: List[str]
    parser_support: bool
    tree_sitter_grammar: Optional[str] = None
    ignore_patterns: Optional[List[str]] = None
    dependency_files: Optional[List[str]] = None
    comment_style: Optional[str] = None  # "//", "#", "/*", etc.


# Language to file extension mapping
LANGUAGE_EXTENSIONS: Dict[Language, List[str]] = {
    Language.PYTHON: [".py", ".pyw", ".pyi"],
    Language.JAVASCRIPT: [".js", ".jsx", ".mjs", ".cjs"],
    Language.TYPESCRIPT: [".ts", ".tsx"],
    Language.JAVA: [".java"],
    Language.GO: [".go"],
    Language.RUST: [".rs"],
    Language.C: [".c", ".h"],
    Language.CPP: [".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".h++"],
    Language.RUBY: [".rb", ".rake"],
    Language.PHP: [".php", ".php3", ".php4", ".php5", ".phtml"],
    Language.CSHARP: [".cs"],
    Language.SWIFT: [".swift"],
    Language.KOTLIN: [".kt", ".kts"],
    Language.SCALA: [".scala", ".sc"],
    Language.DART: [".dart"],
    Language.LUA: [".lua"],
    Language.PERL: [".pl", ".pm"],
    Language.R: [".r", ".R"],
    Language.MATLAB: [".m"],
    Language.SHELL: [".sh", ".bash", ".zsh"],
    Language.POWERSHELL: [".ps1", ".psm1", ".psd1"],
    Language.HTML: [".html", ".htm", ".xhtml"],
    Language.CSS: [".css", ".scss", ".sass", ".less"],
    Language.SQL: [".sql"],
    Language.YAML: [".yaml", ".yml"],
    Language.JSON: [".json"],
    Language.XML: [".xml"],
    Language.MARKDOWN: [".md", ".markdown"],
    Language.BOXLANG: [".bx", ".boxlang"],
    Language.COLDFUSION: [".cfm", ".cfc"],
}

# Reverse mapping: extension to language
EXTENSION_TO_LANGUAGE: Dict[str, Language] = {}
for lang, exts in LANGUAGE_EXTENSIONS.items():
    for ext in exts:
        EXTENSION_TO_LANGUAGE[ext.lower()] = lang


# Language metadata
LANGUAGE_METADATA: Dict[Language, LanguageMetadata] = {
    Language.PYTHON: LanguageMetadata(
        name="Python",
        extensions=[".py", ".pyw", ".pyi"],
        parser_support=True,
        tree_sitter_grammar="tree-sitter-python",
        ignore_patterns=["__pycache__", "*.pyc", "*.pyo", ".pytest_cache", ".mypy_cache", ".ruff_cache"],
        dependency_files=["requirements.txt", "pyproject.toml", "setup.py", "setup.cfg", "poetry.lock", "Pipfile"],
        comment_style="#",
    ),
    Language.JAVASCRIPT: LanguageMetadata(
        name="JavaScript",
        extensions=[".js", ".jsx", ".mjs", ".cjs"],
        parser_support=True,
        tree_sitter_grammar="tree-sitter-javascript",
        ignore_patterns=["node_modules", "*.min.js", ".next", ".nuxt"],
        dependency_files=["package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml"],
        comment_style="//",
    ),
    Language.TYPESCRIPT: LanguageMetadata(
        name="TypeScript",
        extensions=[".ts", ".tsx"],
        parser_support=True,
        tree_sitter_grammar="tree-sitter-typescript",
        ignore_patterns=["node_modules", "*.min.js", ".next", ".nuxt", "dist"],
        dependency_files=["package.json", "package-lock.json", "yarn.lock", "tsconfig.json"],
        comment_style="//",
    ),
    Language.JAVA: LanguageMetadata(
        name="Java",
        extensions=[".java"],
        parser_support=True,
        tree_sitter_grammar="tree-sitter-java",
        ignore_patterns=["target", "build", "*.class", ".gradle"],
        dependency_files=["pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle"],
        comment_style="//",
    ),
    Language.GO: LanguageMetadata(
        name="Go",
        extensions=[".go"],
        parser_support=True,
        tree_sitter_grammar="tree-sitter-go",
        ignore_patterns=["vendor", "*.test"],
        dependency_files=["go.mod", "go.sum"],
        comment_style="//",
    ),
    Language.RUST: LanguageMetadata(
        name="Rust",
        extensions=[".rs"],
        parser_support=True,
        tree_sitter_grammar="tree-sitter-rust",
        ignore_patterns=["target", "Cargo.lock"],
        dependency_files=["Cargo.toml", "Cargo.lock"],
        comment_style="//",
    ),
    Language.C: LanguageMetadata(
        name="C",
        extensions=[".c", ".h"],
        parser_support=True,
        tree_sitter_grammar="tree-sitter-c",
        ignore_patterns=["*.o", "*.a", "*.so", "*.dylib"],
        dependency_files=["CMakeLists.txt", "Makefile", "configure.ac"],
        comment_style="//",
    ),
    Language.CPP: LanguageMetadata(
        name="C++",
        extensions=[".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".h++"],
        parser_support=True,
        tree_sitter_grammar="tree-sitter-cpp",
        ignore_patterns=["*.o", "*.a", "*.so", "*.dylib", "build"],
        dependency_files=["CMakeLists.txt", "Makefile", "conanfile.txt"],
        comment_style="//",
    ),
    Language.RUBY: LanguageMetadata(
        name="Ruby",
        extensions=[".rb", ".rake"],
        parser_support=True,
        tree_sitter_grammar="tree-sitter-ruby",
        ignore_patterns=["vendor", "*.gem"],
        dependency_files=["Gemfile", "Gemfile.lock", "Rakefile"],
        comment_style="#",
    ),
    Language.PHP: LanguageMetadata(
        name="PHP",
        extensions=[".php", ".php3", ".php4", ".php5", ".phtml"],
        parser_support=True,
        tree_sitter_grammar="tree-sitter-php",
        ignore_patterns=["vendor", "*.cache"],
        dependency_files=["composer.json", "composer.lock"],
        comment_style="//",
    ),
    Language.BOXLANG: LanguageMetadata(
        name="BoxLang",
        extensions=[".bx", ".boxlang"],
        parser_support=True,
        tree_sitter_grammar=None,
        ignore_patterns=["box", ".boxlang"],
        dependency_files=["box.json"],
        comment_style="//",
    ),
    Language.COLDFUSION: LanguageMetadata(
        name="ColdFusion",
        extensions=[".cfm", ".cfc"],
        parser_support=True,
        tree_sitter_grammar=None,
        ignore_patterns=["WEB-INF", "cfclasses"],
        dependency_files=["Application.cfc", "Application.cfm", "Application.bx", "box.json"],
        comment_style="<!--",
    ),
}


def get_extensions_for_languages(languages: List[str]) -> List[str]:
    """
    Get file extensions for a list of language names.
    
    Args:
        languages: List of language names (e.g., ["python", "javascript"])
    
    Returns:
        List of file extensions (e.g., [".py", ".js", ".jsx"])
    """
    extensions = []
    for lang_name in languages:
        try:
            lang = Language(lang_name.lower())
            if lang in LANGUAGE_EXTENSIONS:
                extensions.extend(LANGUAGE_EXTENSIONS[lang])
        except ValueError:
            # Unknown language, skip
            continue
    return list(set(extensions))  # Remove duplicates


def get_language_for_extension(extension: str) -> Optional[Language]:
    """
    Get language for a file extension.
    
    Args:
        extension: File extension (e.g., ".py", ".js")
    
    Returns:
        Language enum or None if not found
    """
    return EXTENSION_TO_LANGUAGE.get(extension.lower())


def get_supported_languages() -> List[str]:
    """
    Get list of all supported language names.
    
    Returns:
        List of language names
    """
    return [lang.value for lang in Language]


def get_language_metadata(language: str) -> Optional[LanguageMetadata]:
    """
    Get metadata for a language.
    
    Args:
        language: Language name (e.g., "python")
    
    Returns:
        LanguageMetadata or None if not found
    """
    try:
        lang = Language(language.lower())
        return LANGUAGE_METADATA.get(lang)
    except ValueError:
        return None


def get_all_dependency_file_patterns() -> List[str]:
    """
    Get all dependency file patterns across all languages.
    
    Returns:
        List of dependency file patterns
    """
    patterns = set()
    for metadata in LANGUAGE_METADATA.values():
        if metadata.dependency_files:
            patterns.update(metadata.dependency_files)
    return sorted(list(patterns))

