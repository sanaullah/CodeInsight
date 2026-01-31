"""
Language detection for automatically identifying project languages.

Detects programming languages in a project by analyzing:
- File extensions
- Configuration files (package.json, pom.xml, Cargo.toml, etc.)
- Project structure
"""

import logging
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import defaultdict

from .language_config import (
    Language,
    LANGUAGE_EXTENSIONS,
    EXTENSION_TO_LANGUAGE,
    get_language_metadata,
)

logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Detects programming languages in a project.
    
    Uses multiple heuristics:
    1. File extension analysis
    2. Configuration file detection
    3. Project structure analysis
    """
    
    # Configuration files that indicate specific languages
    CONFIG_FILE_INDICATORS: Dict[str, List[Language]] = {
        # Python
        "requirements.txt": [Language.PYTHON],
        "pyproject.toml": [Language.PYTHON],
        "setup.py": [Language.PYTHON],
        "setup.cfg": [Language.PYTHON],
        "Pipfile": [Language.PYTHON],
        "poetry.lock": [Language.PYTHON],
        "environment.yml": [Language.PYTHON],
        "conda.yml": [Language.PYTHON],
        # JavaScript/TypeScript
        "package.json": [Language.JAVASCRIPT, Language.TYPESCRIPT],
        "package-lock.json": [Language.JAVASCRIPT, Language.TYPESCRIPT],
        "yarn.lock": [Language.JAVASCRIPT, Language.TYPESCRIPT],
        "pnpm-lock.yaml": [Language.JAVASCRIPT, Language.TYPESCRIPT],
        "tsconfig.json": [Language.TYPESCRIPT],
        "jsconfig.json": [Language.JAVASCRIPT],
        # Java
        "pom.xml": [Language.JAVA],
        "build.gradle": [Language.JAVA],
        "build.gradle.kts": [Language.JAVA],
        "settings.gradle": [Language.JAVA],
        ".classpath": [Language.JAVA],
        ".project": [Language.JAVA],
        # Go
        "go.mod": [Language.GO],
        "go.sum": [Language.GO],
        "Gopkg.toml": [Language.GO],
        # Rust
        "Cargo.toml": [Language.RUST],
        "Cargo.lock": [Language.RUST],
        # Ruby
        "Gemfile": [Language.RUBY],
        "Gemfile.lock": [Language.RUBY],
        "Rakefile": [Language.RUBY],
        # PHP
        "composer.json": [Language.PHP],
        "composer.lock": [Language.PHP],
        # C/C++
        "CMakeLists.txt": [Language.C, Language.CPP],
        "Makefile": [Language.C, Language.CPP],
        "conanfile.txt": [Language.CPP],
        "configure.ac": [Language.C, Language.CPP],
        # C#
        "*.csproj": [Language.CSHARP],
        "*.sln": [Language.CSHARP],
        # Swift
        "Package.swift": [Language.SWIFT],
        "*.xcodeproj": [Language.SWIFT],
        # Kotlin
        "build.gradle.kts": [Language.KOTLIN],
        # Scala
        "build.sbt": [Language.SCALA],
        # Dart
        "pubspec.yaml": [Language.DART],
        "pubspec.lock": [Language.DART],
        # BoxLang/ColdFusion
        "box.json": [Language.BOXLANG, Language.COLDFUSION],
        "Application.cfc": [Language.COLDFUSION],
        "Application.cfm": [Language.COLDFUSION],
    }
    
    def __init__(self, min_confidence: float = 0.3):
        """
        Initialize language detector.
        
        Args:
            min_confidence: Minimum confidence threshold (0.0-1.0) for language detection
        """
        self.min_confidence = min_confidence
    
    def detect_languages(self, project_path: str) -> Dict[str, float]:
        """
        Detect languages in a project.
        
        Args:
            project_path: Path to project directory
        
        Returns:
            Dictionary mapping language names to confidence scores (0.0-1.0)
        """
        project_root = Path(project_path).resolve()
        
        if not project_root.exists() or not project_root.is_dir():
            logger.warning(f"Invalid project path: {project_path}")
            return {}
        
        logger.info(f"Detecting languages in: {project_root}")
        
        # Collect evidence from multiple sources
        extension_evidence = self._analyze_file_extensions(project_root)
        config_evidence = self._analyze_config_files(project_root)
        
        # Combine evidence
        language_scores: Dict[Language, float] = defaultdict(float)
        
        # File extension evidence (weight: 0.6)
        for lang, score in extension_evidence.items():
            language_scores[lang] += score * 0.6
        
        # Config file evidence (weight: 0.4, but higher confidence)
        for lang, score in config_evidence.items():
            language_scores[lang] += score * 0.4
        
        # Normalize scores to 0.0-1.0 range
        if language_scores:
            max_score = max(language_scores.values())
            if max_score > 0:
                for lang in language_scores:
                    language_scores[lang] = min(1.0, language_scores[lang] / max_score)
        
        # Filter by minimum confidence
        filtered_scores = {
            lang.value: score
            for lang, score in language_scores.items()
            if score >= self.min_confidence
        }
        
        # Sort by confidence (descending)
        sorted_scores = dict(sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True))
        
        logger.info(f"Detected languages: {sorted_scores}")
        return sorted_scores
    
    def _analyze_file_extensions(self, project_root: Path) -> Dict[Language, float]:
        """
        Analyze file extensions to detect languages.
        
        Args:
            project_root: Project root directory
        
        Returns:
            Dictionary mapping languages to confidence scores
        """
        extension_counts: Dict[str, int] = defaultdict(int)
        total_files = 0
        
        # Count files by extension (limit depth to avoid scanning too much)
        try:
            for file_path in project_root.rglob("*"):
                if file_path.is_file() and not self._should_skip_file(file_path, project_root):
                    ext = file_path.suffix.lower()
                    if ext:
                        extension_counts[ext] += 1
                        total_files += 1
        except Exception as e:
            logger.warning(f"Error scanning files: {e}")
        
        if total_files == 0:
            return {}
        
        # Map extensions to languages and calculate scores
        language_scores: Dict[Language, float] = defaultdict(float)
        
        for ext, count in extension_counts.items():
            lang = EXTENSION_TO_LANGUAGE.get(ext)
            if lang:
                # Score based on file count (normalized)
                score = count / total_files
                language_scores[lang] += score
        
        return language_scores
    
    def _analyze_config_files(self, project_root: Path) -> Dict[Language, float]:
        """
        Analyze configuration files to detect languages.
        
        Args:
            project_root: Project root directory
        
        Returns:
            Dictionary mapping languages to confidence scores
        """
        language_scores: Dict[Language, float] = defaultdict(float)
        
        # Check for config files
        for config_file, languages in self.CONFIG_FILE_INDICATORS.items():
            if "*" in config_file:
                # Handle glob patterns
                pattern = config_file.replace("*", "")
                for file_path in project_root.rglob(f"*{pattern}"):
                    if file_path.is_file():
                        # High confidence for config files
                        for lang in languages:
                            language_scores[lang] += 0.8
                        break
            else:
                # Direct file path
                file_path = project_root / config_file
                if file_path.exists() and file_path.is_file():
                    # High confidence for config files
                    for lang in languages:
                        language_scores[lang] += 0.8
        
        return language_scores
    
    def _should_skip_file(self, file_path: Path, project_root: Path) -> bool:
        """
        Check if a file should be skipped during language detection.
        
        Args:
            file_path: File path to check
            project_root: Project root directory
        
        Returns:
            True if file should be skipped
        """
        # Skip common ignore patterns
        ignore_patterns = {
            ".git", "__pycache__", "node_modules", "venv", ".venv",
            "target", "build", "dist", ".idea", ".vscode", ".DS_Store",
            "vendor", ".next", ".nuxt", "*.min.js", "*.min.css"
        }
        
        try:
            relative_path = file_path.relative_to(project_root)
            for part in relative_path.parts:
                if part in ignore_patterns:
                    return True
                # Check for patterns
                if any(pattern.replace("*", "") in part for pattern in ignore_patterns if "*" in pattern):
                    return True
        except ValueError:
            return True
        
        return False
    
    def get_file_extensions_for_languages(self, languages: List[str]) -> List[str]:
        """
        Get file extensions for a list of languages.
        
        Args:
            languages: List of language names
        
        Returns:
            List of file extensions
        """
        from .language_config import get_extensions_for_languages
        return get_extensions_for_languages(languages)
    
    def is_multi_language_project(self, project_path: str) -> bool:
        """
        Check if project contains multiple languages.
        
        Args:
            project_path: Path to project directory
        
        Returns:
            True if multiple languages detected
        """
        detected = self.detect_languages(project_path)
        return len(detected) > 1

