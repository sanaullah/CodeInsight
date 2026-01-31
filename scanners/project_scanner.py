"""
Project scanner for discovering and reading code files in a directory.

Supports multiple programming languages through configurable file extensions.
"""

import logging
from pathlib import Path
from typing import List, Optional, Set, Dict, Any
from dataclasses import dataclass

from .file_reader import read_file_with_encoding
from .language_config import get_all_dependency_file_patterns, get_language_for_extension

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """Information about a scanned file."""
    path: Path  # Absolute path
    relative_path: str  # Relative path from project root
    content: str  # File content
    size: int  # File size in bytes
    line_count: int  # Number of lines
    encoding: Optional[str] = None  # Detected encoding


class ProjectScanner:
    """
    Scans a directory for code files and reads their contents.
    
    Supports multiple programming languages through configurable file extensions.
    Filters out common ignore patterns like .git/, __pycache__/, venv/, etc.
    """
    
    # Common ignore patterns
    DEFAULT_IGNORE_PATTERNS: Set[str] = {
        ".git",
        "__pycache__",
        "venv",
        "env",
        ".venv",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
        "*.egg-info",
        ".idea",
        ".vscode",
        ".DS_Store",
        "target",  # Rust, Java
        "vendor",  # Go, Ruby, PHP
        ".next",  # Next.js
        ".nuxt",  # Nuxt.js
        "*.log",  # Log files
        "*.tmp",  # Temporary files
        "temp",  # Temp directories
        "tmp",  # Tmp directories
    }
    
    # Dependency file patterns - dynamically loaded from language_config
    # This ensures consistency across the codebase
    @classmethod
    def _get_dependency_patterns(cls) -> List[str]:
        """Get dependency file patterns from language configuration."""
        try:
            return get_all_dependency_file_patterns()
        except Exception:
            # Fallback to Python-only patterns if language_config not available
            return [
                "requirements.txt",
                "requirements*.txt",
                "pyproject.toml",
                "setup.py",
                "setup.cfg",
                "poetry.lock",
                "Pipfile",
                "Pipfile.lock",
                "environment.yml",
                "conda.yml",
            ]
    
    def _get_dependency_file_patterns(self) -> List[str]:
        """Get dependency file patterns for all supported languages."""
        return self._get_dependency_patterns()
    
    # Documentation file patterns
    DOCUMENTATION_FILE_PATTERNS = [
        "README.md",
        "README.rst",
        "README.txt",
        "readme.md",
        "readme.rst",
        "readme.txt",
        "CHANGELOG.md",
        "CHANGELOG.rst",
        "LICENSE",
        "LICENSE.txt",
        "docs/**/*.md",
        "docs/**/*.rst",
    ]
    
    def __init__(self, ignore_patterns: Optional[Set[str]] = None, 
                 file_extensions: Optional[List[str]] = None,
                 auto_detect_languages: bool = False):
        """
        Initialize the project scanner.
        
        Args:
            ignore_patterns: Additional patterns to ignore (merged with defaults)
            file_extensions: List of file extensions to scan (default: [".py"] for backward compatibility)
            auto_detect_languages: If True, automatically detect languages from project structure
        """
        self.ignore_patterns = self.DEFAULT_IGNORE_PATTERNS.copy()
        if ignore_patterns:
            self.ignore_patterns.update(ignore_patterns)
        
        # Set file extensions (default to Python for backward compatibility)
        self.file_extensions = file_extensions if file_extensions is not None else [".py"]
        self.auto_detect_languages = auto_detect_languages
        
        # Hook registry for cache invalidation (will be set by register_invalidation_hook)
        self._invalidation_hook = None
    
    def scan_directory(self, directory_path: str, 
                       file_extensions: Optional[List[str]] = None) -> List[FileInfo]:
        """
        Scan a directory for code files.
        
        Args:
            directory_path: Path to the directory to scan (absolute or relative)
            file_extensions: Optional list of file extensions to scan (overrides instance default)
        
        Returns:
            List of FileInfo objects for all discovered code files
        """
        project_root = Path(directory_path).resolve()
        
        if not project_root.exists():
            logger.error(f"Directory does not exist: {project_root}")
            return []
        
        if not project_root.is_dir():
            logger.error(f"Path is not a directory: {project_root}")
            return []
        
        # Use provided extensions or instance default
        extensions = file_extensions if file_extensions is not None else self.file_extensions
        
        # Auto-detect languages if enabled and no extensions provided
        if self.auto_detect_languages and not file_extensions:
            try:
                from .language_detector import LanguageDetector
                detector = LanguageDetector()
                detected_languages = detector.detect_languages(str(project_root))
                if detected_languages:
                    from .language_config import get_extensions_for_languages
                    extensions = get_extensions_for_languages(list(detected_languages.keys()))
                    logger.info(f"Auto-detected languages: {list(detected_languages.keys())}, extensions: {extensions}")
            except ImportError:
                logger.warning("LanguageDetector not available, using default extensions")
        
        logger.info(f"Scanning directory: {project_root} for extensions: {extensions}")
        
        files: List[FileInfo] = []
        
        # Recursively find all files matching the extensions
        for ext in extensions:
            # Ensure extension starts with dot
            if not ext.startswith("."):
                ext = "." + ext
            
            pattern = f"*{ext}"
            for code_file in project_root.rglob(pattern):
                # Check if file should be ignored
                if self._should_ignore(code_file, project_root):
                    continue
                
                # Read file
                file_info = self._read_file(code_file, project_root)
                if file_info:
                    files.append(file_info)
        
        # Sort files by relative path for deterministic ordering
        # This ensures consistent chunking across multiple runs
        files.sort(key=lambda f: f.relative_path)
        
        logger.info(f"Found {len(files)} code files")
        return files
    
    def scan_directories(
        self,
        directories: List[str],
        project_root: str,
        file_extensions: Optional[List[str]] = None
    ) -> List[FileInfo]:
        """
        Scan multiple directories and combine results.
        
        Args:
            directories: List of directory paths (relative or absolute to project_root)
            project_root: Root path of the project (for relative path calculation)
            file_extensions: Optional list of file extensions to scan
        
        Returns:
            Combined list of FileInfo objects from all directories
        """
        project_root_path = Path(project_root).resolve()
        all_files: List[FileInfo] = []
        seen_paths: Set[str] = set()
        
        for directory in directories:
            # Resolve directory path
            if Path(directory).is_absolute():
                dir_path = Path(directory).resolve()
            else:
                dir_path = project_root_path / directory
            
            if not dir_path.exists() or not dir_path.is_dir():
                logger.warning(f"Directory does not exist or is not a directory: {dir_path}")
                continue
            
            # Scan this directory
            files = self.scan_directory(str(dir_path), file_extensions=file_extensions)
            
            # Add files, filtering duplicates by relative path
            for file_info in files:
                if file_info.relative_path not in seen_paths:
                    seen_paths.add(file_info.relative_path)
                    all_files.append(file_info)
        
        # Sort files by relative path for deterministic ordering
        all_files.sort(key=lambda f: f.relative_path)
        
        logger.info(f"Scanned {len(directories)} directories, found {len(all_files)} unique files")
        return all_files
    
    def _should_ignore(self, file_path: Path, project_root: Path) -> bool:
        """Check if a file should be ignored based on ignore patterns."""
        # Get relative path from project root
        try:
            relative_path = file_path.relative_to(project_root)
        except ValueError:
            # File is outside project root, ignore it
            return True
        
        # Check each part of the path
        for part in relative_path.parts:
            if part in self.ignore_patterns:
                return True
            # Check for patterns like *.egg-info
            if any(pattern.replace("*", "") in part for pattern in self.ignore_patterns if "*" in pattern):
                return True
        
        return False
    
    def _read_file(self, file_path: Path, project_root: Path) -> Optional[FileInfo]:
        """Read a file and create FileInfo object."""
        try:
            # Get relative path
            relative_path = str(file_path.relative_to(project_root))
            
            # Read content with encoding detection
            content, encoding = read_file_with_encoding(file_path)
            
            if content is None:
                logger.warning(f"Could not read file: {file_path}")
                return None
            
            # Get file stats
            stat = file_path.stat()
            size = stat.st_size
            line_count = len(content.splitlines())
            
            return FileInfo(
                path=file_path,
                relative_path=relative_path,
                content=content,
                size=size,
                line_count=line_count,
                encoding=encoding
            )
        
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None
    
    def get_file_count(self, directory_path: str, 
                       file_extensions: Optional[List[str]] = None) -> int:
        """Get count of code files without reading them."""
        files = self.scan_directory(directory_path, file_extensions=file_extensions)
        return len(files)
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get currently configured file extensions.
        
        Returns:
            List of file extensions (e.g., [".py", ".js"])
        """
        return self.file_extensions.copy()
    
    def scan_dependency_files(self, directory_path: str) -> List[FileInfo]:
        """
        Scan a directory for dependency files.
        
        Args:
            directory_path: Path to the directory to scan (absolute or relative)
        
        Returns:
            List of FileInfo objects for all discovered dependency files
        """
        project_root = Path(directory_path).resolve()
        
        if not project_root.exists():
            logger.error(f"Directory does not exist: {project_root}")
            return []
        
        if not project_root.is_dir():
            logger.error(f"Path is not a directory: {project_root}")
            return []
        
        logger.info(f"Scanning for dependency files in: {project_root}")
        
        files: List[FileInfo] = []
        
        # Find all dependency files
        dependency_patterns = self._get_dependency_file_patterns()
        for pattern in dependency_patterns:
            if "*" in pattern:
                # Handle glob patterns
                for file_path in project_root.glob(pattern):
                    if file_path.is_file() and not self._should_ignore(file_path, project_root):
                        file_info = self._read_file(file_path, project_root)
                        if file_info:
                            files.append(file_info)
            else:
                # Direct file path
                file_path = project_root / pattern
                if file_path.exists() and file_path.is_file():
                    if not self._should_ignore(file_path, project_root):
                        file_info = self._read_file(file_path, project_root)
                        if file_info:
                            files.append(file_info)
        
        # Sort files by relative path
        files.sort(key=lambda f: f.relative_path)
        
        logger.info(f"Found {len(files)} dependency files")
        return files
    
    def get_documentation_files(self, directory_path: str) -> List[FileInfo]:
        """
        Get FileInfo objects for documentation files.
        
        Args:
            directory_path: Path to the directory to scan
        
        Returns:
            List of FileInfo objects for documentation files
        """
        return self.scan_documentation_files(directory_path)
    
    def scan_documentation_files(self, directory_path: str) -> List[FileInfo]:
        """
        Scan a directory for documentation files.
        
        Args:
            directory_path: Path to the directory to scan (absolute or relative)
        
        Returns:
            List of FileInfo objects for all discovered documentation files
        """
        project_root = Path(directory_path).resolve()
        
        if not project_root.exists():
            logger.error(f"Directory does not exist: {project_root}")
            return []
        
        if not project_root.is_dir():
            logger.error(f"Path is not a directory: {project_root}")
            return []
        
        logger.info(f"Scanning for documentation files in: {project_root}")
        
        files: List[FileInfo] = []
        
        # Find all documentation files
        for pattern in self.DOCUMENTATION_FILE_PATTERNS:
            if "**" in pattern:
                # Handle recursive glob patterns
                glob_pattern = pattern.replace("**", "*")
                for file_path in project_root.rglob(glob_pattern):
                    if file_path.is_file() and not self._should_ignore(file_path, project_root):
                        file_info = self._read_file(file_path, project_root)
                        if file_info:
                            files.append(file_info)
            elif "*" in pattern:
                # Handle glob patterns
                for file_path in project_root.glob(pattern):
                    if file_path.is_file() and not self._should_ignore(file_path, project_root):
                        file_info = self._read_file(file_path, project_root)
                        if file_info:
                            files.append(file_info)
            else:
                # Direct file path
                file_path = project_root / pattern
                if file_path.exists() and file_path.is_file():
                    if not self._should_ignore(file_path, project_root):
                        file_info = self._read_file(file_path, project_root)
                        if file_info:
                            files.append(file_info)
        
        # Sort files by relative path
        files.sort(key=lambda f: f.relative_path)
        
        logger.info(f"Found {len(files)} documentation files")
        return files
    
    def extract_documentation_metadata(
        self,
        file_paths: List[Path],
        project_root: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Extract documentation metadata from files for service consumption.
        
        Args:
            file_paths: List of file paths to analyze
            project_root: Optional project root for relative paths
        
        Returns:
            Dictionary with comprehensive documentation metadata including:
            - readme_files: List of README file paths
            - changelog_files: List of CHANGELOG file paths
            - license_files: List of LICENSE file paths
            - docstring_count: Total number of docstrings found
            - total_functions: Total number of functions/classes
            - documented_functions: Number of documented functions/classes
            - coverage_percentage: Documentation coverage (0-100)
            - documentation_files: List of all documentation file paths
        """
        metadata = {
            "readme_files": [],
            "changelog_files": [],
            "license_files": [],
            "documentation_files": [],
            "docstring_count": 0,
            "total_functions": 0,
            "total_classes": 0,
            "documented_functions": 0,
            "documented_classes": 0,
            "coverage_percentage": 0.0,
        }
        
        for file_path in file_paths:
            try:
                file_name_upper = file_path.name.upper()
                rel_path = str(file_path.relative_to(project_root)) if project_root and project_root in file_path.parents else str(file_path)
                
                # Check for documentation files
                if file_name_upper.startswith("README"):
                    metadata["readme_files"].append(rel_path)
                    metadata["documentation_files"].append(rel_path)
                elif file_name_upper.startswith("CHANGELOG"):
                    metadata["changelog_files"].append(rel_path)
                    metadata["documentation_files"].append(rel_path)
                elif file_name_upper.startswith("LICENSE") or file_name_upper == "LICENCE":
                    metadata["license_files"].append(rel_path)
                    metadata["documentation_files"].append(rel_path)
                elif file_path.parent.name.lower() == "docs" or "docs" in str(file_path.parent):
                    metadata["documentation_files"].append(rel_path)
                
                # Extract code metadata using multi-language parser
                # Check if file extension is in supported extensions
                if file_path.suffix in self.file_extensions:
                    language_obj = get_language_for_extension(file_path.suffix)
                    if language_obj:
                        language = language_obj.value
                        
                        # Use MultiLanguageParser for AST parsing
                        try:
                            from parsers.multi_language_parser import MultiLanguageParser
                            parser = MultiLanguageParser(fallback_to_text=True)
                            ast_metadata = parser.parse_file(file_path, language)
                            
                            if ast_metadata:
                                metadata["total_functions"] += ast_metadata.get("total_functions", 0)
                                metadata["total_classes"] += ast_metadata.get("total_classes", 0)
                                metadata["documented_functions"] += ast_metadata.get("documented_functions", 0)
                                metadata["documented_classes"] += ast_metadata.get("documented_classes", 0)
                                
                                # Count docstrings from documentation metadata
                                doc_metadata = ast_metadata.get("documentation", {})
                                metadata["docstring_count"] += doc_metadata.get("docstring_count", 0)
                        except ImportError:
                            # MultiLanguageParser not available, fallback to Python AST for .py files
                            if file_path.suffix == ".py":
                                try:
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        content = f.read()
                                    
                                    import ast
                                    try:
                                        tree = ast.parse(content, filename=str(file_path))
                                        
                                        for node in ast.walk(tree):
                                            if isinstance(node, ast.FunctionDef):
                                                metadata["total_functions"] += 1
                                                if ast.get_docstring(node):
                                                    metadata["documented_functions"] += 1
                                                    metadata["docstring_count"] += 1
                                            elif isinstance(node, ast.ClassDef):
                                                metadata["total_classes"] += 1
                                                if ast.get_docstring(node):
                                                    metadata["documented_classes"] += 1
                                                    metadata["docstring_count"] += 1
                                    except SyntaxError:
                                        logger.debug(f"Skipping file with syntax error: {file_path}")
                                        continue
                                except Exception as e:
                                    logger.debug(f"Error parsing Python file {file_path}: {e}")
                        except Exception as e:
                            logger.debug(f"Error parsing file {file_path} with MultiLanguageParser: {e}")
            
            except Exception as e:
                logger.warning(f"Error extracting metadata from {file_path}: {e}")
        
        # Calculate coverage percentage
        total_items = metadata["total_functions"] + metadata["total_classes"]
        documented_items = metadata["documented_functions"] + metadata["documented_classes"]
        if total_items > 0:
            metadata["coverage_percentage"] = (documented_items / total_items) * 100.0
        else:
            metadata["coverage_percentage"] = 0.0
        
        return metadata
    
    def get_dependency_file_info(self, directory_path: str) -> List[FileInfo]:
        """
        Get FileInfo objects for dependency files.
        
        Alias for scan_dependency_files() for consistency.
        
        Args:
            directory_path: Path to the directory to scan
        
        Returns:
            List of FileInfo objects for dependency files
        """
        return self.scan_dependency_files(directory_path)
    
    def extract_dependency_metadata(
        self,
        directory_path: str
    ) -> Dict[str, Any]:
        """
        Extract dependency file metadata for service consumption.
        
        Args:
            directory_path: Path to the directory to scan
        
        Returns:
            Dictionary with dependency metadata including:
            - dependency_files: List of dependency file paths
            - file_types: Dictionary mapping file paths to their types
            - total_files: Total number of dependency files found
        """
        dependency_files = self.scan_dependency_files(directory_path)
        
        metadata = {
            "dependency_files": [f.relative_path for f in dependency_files],
            "file_types": {},
            "total_files": len(dependency_files)
        }
        
        # Map file paths to their types
        for file_info in dependency_files:
            file_name = file_info.path.name.lower()
            if file_name == "requirements.txt" or file_name.startswith("requirements"):
                file_type = "requirements"
            elif file_name == "pyproject.toml":
                file_type = "pyproject"
            elif file_name == "setup.py":
                file_type = "setup_py"
            elif file_name == "setup.cfg":
                file_type = "setup_cfg"
            elif file_name == "poetry.lock":
                file_type = "poetry_lock"
            elif file_name == "pipfile" or file_name == "pipfile.lock":
                file_type = "pipfile"
            elif file_name.endswith(".yml") or file_name.endswith(".yaml"):
                file_type = "conda"
            else:
                file_type = "unknown"
            
            metadata["file_types"][file_info.relative_path] = file_type
        
        return metadata
    
    def register_invalidation_hook(self, hook) -> None:
        """
        Register a cache invalidation hook.
        
        Args:
            hook: FileChangeHook instance
        """
        from services.cache_invalidation_hooks import FileChangeHook
        if isinstance(hook, FileChangeHook):
            self._invalidation_hook = hook
            logger.info("Registered FileChangeHook for project scanner")
        else:
            logger.warning(f"Invalid hook type for ProjectScanner: {type(hook)}")
    
    def trigger_file_change_invalidation(self, project_path: str, file_hash: Optional[str] = None) -> None:
        """
        Trigger file change invalidation hook.
        
        This should be called when file changes are detected (e.g., via file hash comparison).
        
        Args:
            project_path: Project path
            file_hash: Optional file hash (if None, invalidates all for project)
        """
        if self._invalidation_hook:
            event_data = {
                "project_path": project_path,
                "file_hash": file_hash
            }
            self._invalidation_hook.on_invalidate(event_data)

