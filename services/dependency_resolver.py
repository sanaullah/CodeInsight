"""
Dependency Resolver Service.

Detects dependencies between directories by parsing imports from code files.
Supports multiple programming languages and maps imports to directory paths.
"""

import ast
import re
import logging
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass

from scanners.project_scanner import FileInfo
from utils.io.async_io import async_ast_parse

logger = logging.getLogger(__name__)


@dataclass
class DependencyInfo:
    """Information about a dependency."""
    source_dir: str
    target_dir: str
    import_statement: str
    file_path: str


class DependencyResolver:
    """
    Resolves dependencies between directories by parsing imports.
    
    Supports:
    - Python (AST-based parsing)
    - JavaScript/TypeScript (regex-based)
    - Other languages (basic regex fallback)
    """
    
    # Common ignore patterns for directories
    IGNORE_PATTERNS = {
        ".git", "__pycache__", "venv", "env", ".venv", "node_modules",
        ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "build",
        "*.egg-info", ".idea", ".vscode", ".DS_Store", "target", "vendor",
        ".next", ".nuxt"
    }
    
    def __init__(self, max_depth: int = 3):
        """
        Initialize dependency resolver.
        
        Args:
            max_depth: Maximum depth for recursive dependency resolution
        """
        self.max_depth = max_depth
    
    def detect_dependencies(
        self,
        selected_dirs: List[str],
        project_root: str,
        files: List[FileInfo]
    ) -> Dict[str, List[str]]:
        """
        Detect dependencies for selected directories.
        
        Args:
            selected_dirs: List of selected directory paths (relative to project_root)
            project_root: Root path of the project
            files: List of FileInfo objects from scanned files
        
        Returns:
            Dictionary mapping selected directories to their dependent directories
        """
        project_root_path = Path(project_root).resolve()
        dependencies: Dict[str, List[str]] = {dir_path: [] for dir_path in selected_dirs}
        
        # Get all available directories in project
        available_dirs = self._get_available_directories(project_root_path)
        
        # Parse imports from files in selected directories
        for file_info in files:
            file_path = Path(file_info.relative_path)
            
            # Check if file is in one of the selected directories
            source_dir = self._get_directory_for_file(file_path, selected_dirs)
            if not source_dir:
                continue
            
            # Parse imports based on file extension
            imports = self._parse_imports(file_info, project_root_path)
            
            # Map imports to directory paths
            for import_name in imports:
                target_dir = self._map_import_to_directory(
                    import_name,
                    file_path,
                    project_root_path,
                    available_dirs
                )
                
                if target_dir and target_dir not in dependencies[source_dir]:
                    dependencies[source_dir].append(target_dir)
        
        return dependencies
    
    def resolve_dependencies(
        self,
        selected_dirs: List[str],
        project_root: str,
        files: Optional[List[FileInfo]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Resolve all dependencies recursively.
        
        Args:
            selected_dirs: List of selected directory paths (relative to project_root)
            project_root: Root path of the project
            files: Optional list of FileInfo objects from selected directories
        
        Returns:
            Tuple of (all_directories, auto_included_directories)
        """
        project_root_path = Path(project_root).resolve()
        
        # Normalize selected directories (ensure they're relative paths)
        normalized_selected = []
        for d in selected_dirs:
            try:
                if Path(d).is_absolute():
                    normalized_selected.append(str(Path(d).relative_to(project_root_path)))
                else:
                    normalized_selected.append(d)
            except ValueError:
                # Directory is outside project root, skip it
                logger.warning(f"Directory {d} is outside project root, skipping")
                continue
        
        all_dirs = set(normalized_selected)
        auto_included = set()
        
        # If files not provided, we can't detect dependencies
        if files is None or not files:
            logger.warning("Files not provided to resolve_dependencies, cannot detect dependencies")
            return sorted(list(all_dirs)), sorted(list(auto_included))
        
        # Resolve dependencies recursively
        to_process = set(normalized_selected)
        processed = set()
        depth = 0
        
        while to_process and depth < self.max_depth:
            depth += 1
            current_level = to_process.copy()
            to_process.clear()
            
            for dir_path in current_level:
                if dir_path in processed:
                    continue
                processed.add(dir_path)
                
                # Find dependencies for this directory using files we have
                deps = self._find_dependencies_for_directory(
                    dir_path,
                    project_root_path,
                    files
                )
                
                for dep_dir in deps:
                    if dep_dir not in all_dirs:
                        all_dirs.add(dep_dir)
                        auto_included.add(dep_dir)
                        to_process.add(dep_dir)
        
        return sorted(list(all_dirs)), sorted(list(auto_included))
    
    def _parse_imports(self, file_info: FileInfo, project_root: Path) -> Set[str]:
        """
        Parse imports from a file based on its extension.
        
        Args:
            file_info: FileInfo object with file content
            project_root: Root path of the project
        
        Returns:
            Set of import module names
        """
        file_path = Path(file_info.relative_path)
        extension = file_path.suffix.lower()
        
        if extension == ".py":
            return self._parse_python_imports(file_info.content, file_path, project_root)
        elif extension in [".js", ".jsx", ".mjs", ".cjs"]:
            return self._parse_javascript_imports(file_info.content)
        elif extension in [".ts", ".tsx"]:
            return self._parse_typescript_imports(file_info.content)
        else:
            # Fallback to basic regex for other languages
            return self._parse_generic_imports(file_info.content)
    
    def _parse_python_imports(
        self,
        content: str,
        file_path: Path,
        project_root: Path
    ) -> Set[str]:
        """
        Parse Python imports using AST.
        
        Args:
            content: File content
            file_path: Path to the file
            project_root: Root path of the project
        
        Returns:
            Set of import module names
        """
        imports = set()
        
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Get first part of import (e.g., "utils" from "utils.langfuse")
                        module_name = alias.name.split('.')[0]
                        imports.add(module_name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Get first part of module (e.g., "utils" from "utils.langfuse")
                        module_name = node.module.split('.')[0]
                        imports.add(module_name)
                    elif node.level > 0:
                        # Relative import (e.g., "from .utils import ...")
                        # Try to resolve relative to current file's directory
                        current_dir = file_path.parent
                        for _ in range(node.level - 1):
                            current_dir = current_dir.parent
                        if current_dir != project_root:
                            dir_name = current_dir.name
                            if dir_name:
                                imports.add(dir_name)
        except SyntaxError as e:
            logger.debug(f"Syntax error parsing {file_path}: {e}, falling back to regex")
            # Fallback to regex for files with syntax errors
            imports.update(self._parse_generic_imports(content))
        except Exception as e:
            logger.warning(f"Error parsing Python imports from {file_path}: {e}")
        
        return imports
    
    async def async_parse_python_imports(
        self,
        content: str,
        file_path: Path,
        project_root: Path
    ) -> Set[str]:
        """
        Parse Python imports using AST asynchronously.
        
        Args:
            content: File content
            file_path: Path to the file
            project_root: Root path of the project
        
        Returns:
            Set of import module names
        """
        imports = set()
        
        try:
            tree = await async_ast_parse(content, filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Get first part of import (e.g., "utils" from "utils.langfuse")
                        module_name = alias.name.split('.')[0]
                        imports.add(module_name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Get first part of module (e.g., "utils" from "utils.langfuse")
                        module_name = node.module.split('.')[0]
                        imports.add(module_name)
                    elif node.level > 0:
                        # Relative import (e.g., "from .utils import ...")
                        # Try to resolve relative to current file's directory
                        current_dir = file_path.parent
                        for _ in range(node.level - 1):
                            current_dir = current_dir.parent
                        if current_dir != project_root:
                            dir_name = current_dir.name
                            if dir_name:
                                imports.add(dir_name)
        except SyntaxError as e:
            logger.debug(f"Syntax error parsing {file_path}: {e}, falling back to regex")
            # Fallback to regex for files with syntax errors
            imports.update(self._parse_generic_imports(content))
        except Exception as e:
            logger.warning(f"Error parsing Python imports from {file_path}: {e}")
        
        return imports
    
    def _parse_javascript_imports(self, content: str) -> Set[str]:
        """Parse JavaScript imports using regex."""
        imports = set()
        
        # Match: import ... from 'module' or import ... from "module"
        patterns = [
            r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]",
            r"import\s+['\"]([^'\"]+)['\"]",
            r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            for match in matches:
                # Get first part of import path
                module_name = match.split('/')[0].split('\\')[0]
                # Remove @scope/ if present
                if module_name.startswith('@'):
                    parts = module_name.split('/')
                    if len(parts) > 1:
                        module_name = parts[1]
                imports.add(module_name)
        
        return imports
    
    def _parse_typescript_imports(self, content: str) -> Set[str]:
        """Parse TypeScript imports (same as JavaScript)."""
        return self._parse_javascript_imports(content)
    
    def _parse_generic_imports(self, content: str) -> Set[str]:
        """Generic import parsing using basic regex patterns."""
        imports = set()
        
        # Try to match common import patterns
        patterns = [
            r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]",
            r"import\s+['\"]([^'\"]+)['\"]",
            r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
            r"^import\s+(\w+)",
            r"^from\s+(\w+)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ""
                module_name = match.split('.')[0].split('/')[0].split('\\')[0]
                if module_name and not module_name.startswith('.'):
                    imports.add(module_name)
        
        return imports
    
    def _map_import_to_directory(
        self,
        import_name: str,
        file_path: Path,
        project_root: Path,
        available_dirs: Set[str]
    ) -> Optional[str]:
        """
        Map an import name to a directory path.
        
        Args:
            import_name: Name of the imported module
            file_path: Path to the file making the import
            project_root: Root path of the project
            available_dirs: Set of available directory names in project
        
        Returns:
            Directory path (relative to project_root) or None
        """
        # Skip external packages (common patterns)
        if import_name.startswith('_') or import_name in ['sys', 'os', 'json', 're', 'pathlib']:
            return None
        
        # Check if import name matches a directory
        if import_name in available_dirs:
            return import_name
        
        # Check if it's a relative import that maps to a parent directory
        # This is handled in _parse_python_imports for relative imports
        
        return None
    
    def _get_available_directories(self, project_root: Path) -> Set[str]:
        """
        Get all available directory names in the project.
        
        Args:
            project_root: Root path of the project
        
        Returns:
            Set of directory names (relative to project_root)
        """
        directories = set()
        
        if not project_root.exists() or not project_root.is_dir():
            return directories
        
        try:
            for item in project_root.iterdir():
                if item.is_dir():
                    # Check if should be ignored
                    if self._should_ignore(item.name):
                        continue
                    directories.add(item.name)
        except Exception as e:
            logger.warning(f"Error scanning directories in {project_root}: {e}")
        
        return directories
    
    def _should_ignore(self, name: str) -> bool:
        """Check if a directory name should be ignored."""
        if name in self.IGNORE_PATTERNS:
            return True
        # Check for patterns like *.egg-info
        for pattern in self.IGNORE_PATTERNS:
            if '*' in pattern:
                pattern_base = pattern.replace('*', '')
                if pattern_base in name:
                    return True
        return False
    
    def _get_directory_for_file(
        self,
        file_path: Path,
        selected_dirs: List[str]
    ) -> Optional[str]:
        """
        Get the selected directory that contains this file.
        
        Args:
            file_path: Relative path to the file
            selected_dirs: List of selected directory paths
        
        Returns:
            Directory path if file is in a selected directory, None otherwise
        """
        file_parts = file_path.parts
        
        for selected_dir in selected_dirs:
            selected_parts = Path(selected_dir).parts
            # Check if file is in this directory
            if len(file_parts) >= len(selected_parts):
                if file_parts[:len(selected_parts)] == selected_parts:
                    return selected_dir
        
        return None
    
    def _find_dependencies_for_directory(
        self,
        dir_path: str,
        project_root: Path,
        files: List[FileInfo]
    ) -> List[str]:
        """
        Find all directories that the given directory depends on.
        
        Args:
            dir_path: Directory path (relative to project_root)
            project_root: Root path of the project
            files: List of FileInfo objects
        
        Returns:
            List of dependent directory paths
        """
        dependencies = set()
        available_dirs = self._get_available_directories(project_root)
        
        # Find all files in this directory
        dir_path_obj = Path(dir_path)
        for file_info in files:
            file_path = Path(file_info.relative_path)
            
            # Check if file is in the target directory
            if not self._file_in_directory(file_path, dir_path_obj):
                continue
            
            # Parse imports
            imports = self._parse_imports(file_info, project_root)
            
            # Map imports to directories
            for import_name in imports:
                target_dir = self._map_import_to_directory(
                    import_name,
                    file_path,
                    project_root,
                    available_dirs
                )
                
                if target_dir and target_dir != dir_path:
                    dependencies.add(target_dir)
        
        return list(dependencies)
    
    def _file_in_directory(self, file_path: Path, dir_path: Path) -> bool:
        """Check if a file is in a given directory."""
        file_parts = file_path.parts
        dir_parts = dir_path.parts
        
        if len(file_parts) < len(dir_parts):
            return False
        
        return file_parts[:len(dir_parts)] == dir_parts

