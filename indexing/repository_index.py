"""Immutable shared repository snapshot and lightweight structural index."""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from domain.contracts import RepositorySnapshot
from infrastructure.artifacts.store import FilesystemArtifactStore, StoredArtifact
from infrastructure.db.snapshot_repository import SqliteSnapshotRepository

SCANNER_VERSION = "native-index-v1"
DEFAULT_IGNORE_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "node_modules",
        "vendor",
        "target",
        "dist",
        "build",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".idea",
        ".vscode",
    }
)
LANGUAGES = {
    ".py": ("python", "parsed"),
    ".pyi": ("python", "parsed"),
    ".js": ("javascript", "dependency-aware"),
    ".jsx": ("javascript", "dependency-aware"),
    ".mjs": ("javascript", "dependency-aware"),
    ".cjs": ("javascript", "dependency-aware"),
    ".ts": ("typescript", "dependency-aware"),
    ".tsx": ("typescript", "dependency-aware"),
    ".java": ("java", "discovery"),
    ".kt": ("kotlin", "discovery"),
    ".go": ("go", "discovery"),
    ".rs": ("rust", "discovery"),
    ".cs": ("csharp", "discovery"),
    ".cpp": ("cpp", "discovery"),
    ".cc": ("cpp", "discovery"),
    ".c": ("c", "discovery"),
    ".h": ("c", "discovery"),
    ".hpp": ("cpp", "discovery"),
    ".rb": ("ruby", "discovery"),
    ".php": ("php", "discovery"),
    ".swift": ("swift", "discovery"),
    ".scala": ("scala", "discovery"),
    ".sh": ("shell", "discovery"),
    ".ps1": ("powershell", "discovery"),
    ".sql": ("sql", "discovery"),
    ".vue": ("vue", "dependency-aware"),
    ".svelte": ("svelte", "dependency-aware"),
    ".md": ("markdown", "discovery"),
    ".rst": ("rst", "discovery"),
    ".toml": ("toml", "discovery"),
    ".yaml": ("yaml", "discovery"),
    ".yml": ("yaml", "discovery"),
    ".json": ("json", "discovery"),
}
SPECIAL_FILES = {
    "Dockerfile",
    "Makefile",
    "CODEOWNERS",
    ".gitignore",
    ".dockerignore",
}
_JS_IMPORT = re.compile(
    r"""(?:import(?:[\s\S]*?\sfrom\s*)?|require\s*\()\s*['"]([^'"]+)['"]"""
)


@dataclass(frozen=True, slots=True)
class IndexResult:
    snapshot: RepositorySnapshot
    cached: bool
    file_count: int
    symbol_count: int
    edge_count: int
    changed_paths: tuple[str, ...]
    target_paths: tuple[str, ...]


@dataclass(slots=True)
class _ScannedFile:
    path: Path
    relative_path: str
    content: bytes
    text: str
    content_hash: str
    language: str
    support_tier: str
    classification: str
    line_count: int
    owners: tuple[str, ...]
    artifact: StoredArtifact
    file_id: str = ""


class RepositoryIndexer:
    """Build one reusable repository model for every specialist in a run."""

    def __init__(
        self,
        repository: SqliteSnapshotRepository,
        artifact_store: FilesystemArtifactStore,
        *,
        max_file_bytes: int = 2_000_000,
        max_files: int = 20_000,
    ) -> None:
        if max_file_bytes < 1 or max_files < 1:
            raise ValueError("index limits must be greater than zero")
        self.repository = repository
        self.artifact_store = artifact_store
        self.max_file_bytes = max_file_bytes
        self.max_files = max_files

    def build(
        self,
        project_path: str | Path,
        *,
        base_commit: str | None = None,
        selected_directories: tuple[str, ...] = (),
        include_extensions: tuple[str, ...] = (),
        neighborhood_depth: int = 1,
    ) -> IndexResult:
        root = Path(project_path).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"repository path is not a directory: {root}")
        extensions = {
            extension.lower()
            if extension.startswith(".")
            else f".{extension.lower()}"
            for extension in include_extensions
        }
        configuration = {
            "selected_directories": sorted(selected_directories),
            "include_extensions": sorted(extensions),
            "base_commit": base_commit,
            "max_file_bytes": self.max_file_bytes,
            "scanner_version": SCANNER_VERSION,
        }
        configuration_hash = _hash_json(configuration)
        git = _git_metadata(root, base_commit)
        owners = _load_codeowners(root)
        paths = _discover_paths(root, selected_directories)
        scanned: list[_ScannedFile] = []
        for path in paths:
            if len(scanned) >= self.max_files:
                break
            language = LANGUAGES.get(path.suffix.lower())
            if extensions and path.suffix.lower() not in extensions:
                continue
            if language is None and path.name not in SPECIAL_FILES:
                continue
            try:
                byte_size = path.stat().st_size
                if byte_size > self.max_file_bytes:
                    continue
                content = path.read_bytes()
            except OSError:
                continue
            relative_path = path.relative_to(root).as_posix()
            content_hash = hashlib.sha256(content).hexdigest()
            detected_language, support_tier = language or ("text", "discovery")
            artifact = self.artifact_store.put(
                content,
                artifact_kind="source",
                media_type="text/plain",
            )
            scanned.append(
                _ScannedFile(
                    path=path,
                    relative_path=relative_path,
                    content=content,
                    text=content.decode("utf-8", errors="replace"),
                    content_hash=content_hash,
                    language=detected_language,
                    support_tier=support_tier,
                    classification=_classify(relative_path),
                    line_count=len(content.splitlines()),
                    owners=_owners_for(relative_path, owners),
                    artifact=artifact,
                )
            )
        scanned.sort(key=lambda item: item.relative_path)
        identity_hash = _hash_json(
            {
                "canonical_path": os.path.normcase(str(root)),
                "configuration_hash": configuration_hash,
                "head_commit": git["head_commit"],
                "dirty": git["dirty"],
                "files": [
                    (item.relative_path, item.content_hash) for item in scanned
                ],
            }
        )
        cached = self.repository.get_by_identity(identity_hash)
        if cached is not None:
            snapshot = _snapshot_from_record(cached)
            _assign_file_ids(scanned, snapshot.snapshot_id)
            target_paths = self._target_paths(
                snapshot.snapshot_id,
                scanned,
                set(git["changed_paths"]),
                neighborhood_depth,
            )
            return IndexResult(
                snapshot=snapshot,
                cached=True,
                file_count=len(scanned),
                symbol_count=int(cached["metadata"]["symbol_count"]),
                edge_count=int(cached["metadata"]["edge_count"]),
                changed_paths=tuple(git["changed_paths"]),
                target_paths=target_paths,
            )

        snapshot_id = uuid4().hex
        project_id = hashlib.sha256(os.path.normcase(str(root)).encode()).hexdigest()
        previous_snapshot = self.repository.latest_for_project(project_id)
        _assign_file_ids(scanned, snapshot_id)
        symbols, calls_by_file, python_trees = _extract_symbols(
            scanned, snapshot_id
        )
        edges = _extract_edges(
            scanned, symbols, calls_by_file, python_trees, snapshot_id
        )
        created_at = datetime.now(UTC)
        snapshot = RepositorySnapshot(
            snapshot_id=snapshot_id,
            project_id=project_id,
            canonical_path=str(root),
            identity_hash=identity_hash,
            configuration_hash=configuration_hash,
            scanner_version=SCANNER_VERSION,
            created_at=created_at,
            git_repository=git["repository"],
            git_ref=git["git_ref"],
            base_commit=git["base_commit"],
            head_commit=git["head_commit"],
            parent_snapshot_id=(
                str(previous_snapshot["snapshot_id"]) if previous_snapshot else None
            ),
            dirty=git["dirty"],
            included_paths=tuple(item.relative_path for item in scanned),
            excluded_paths=(),
        )
        metadata = {
            "file_count": len(scanned),
            "symbol_count": len(symbols),
            "edge_count": len(edges),
            "changed_paths": git["changed_paths"],
            "included_paths": [item.relative_path for item in scanned],
            "configuration": configuration,
        }
        files = [
            {
                "file_id": item.file_id,
                "relative_path": item.relative_path,
                "content_hash": item.content_hash,
                "language": item.language,
                "classification": item.classification,
                "support_tier": item.support_tier,
                "byte_size": len(item.content),
                "line_count": item.line_count,
                "artifact_id": item.artifact.artifact_id,
                "metadata": {"owners": item.owners},
            }
            for item in scanned
        ]
        persisted = self.repository.persist(
            snapshot,
            display_name=root.name,
            artifacts=[item.artifact for item in scanned],
            files=files,
            symbols=symbols,
            edges=edges,
            metadata=metadata,
        )
        if not persisted:
            concurrent = self.repository.get_by_identity(identity_hash)
            if concurrent is None:
                raise RuntimeError("concurrent snapshot was not readable")
            snapshot = _snapshot_from_record(concurrent)
        target_paths = self._target_paths(
            snapshot.snapshot_id,
            scanned,
            set(git["changed_paths"]),
            neighborhood_depth,
        )
        return IndexResult(
            snapshot=snapshot,
            cached=not persisted,
            file_count=len(scanned),
            symbol_count=len(symbols),
            edge_count=len(edges),
            changed_paths=tuple(git["changed_paths"]),
            target_paths=target_paths,
        )

    def _target_paths(
        self,
        snapshot_id: str,
        scanned: list[_ScannedFile],
        changed_paths: set[str],
        depth: int,
    ) -> tuple[str, ...]:
        if not changed_paths:
            return tuple(item.relative_path for item in scanned)
        file_by_path = {item.relative_path: item for item in scanned}
        seed_ids = [
            file_by_path[path].file_id for path in changed_paths if path in file_by_path
        ]
        if not seed_ids:
            return tuple(item.relative_path for item in scanned)
        target_ids = set(
            self.repository.neighborhood(snapshot_id, seed_ids, depth=depth)
        )
        return tuple(
            item.relative_path for item in scanned if item.file_id in target_ids
        )


def _discover_paths(root: Path, selected_directories: tuple[str, ...]) -> list[Path]:
    roots = []
    for selected in selected_directories:
        candidate = (root / selected).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"selected directory escapes repository: {selected}") from exc
        if candidate.is_dir():
            roots.append(candidate)
    if not roots:
        roots = [root]
    found: list[Path] = []
    for scan_root in roots:
        for current, directories, filenames in os.walk(scan_root):
            directories[:] = sorted(
                directory
                for directory in directories
                if directory not in DEFAULT_IGNORE_DIRECTORIES
            )
            found.extend(Path(current) / filename for filename in sorted(filenames))
    return sorted(set(found), key=lambda path: path.relative_to(root).as_posix())


def _git(
    root: Path, *arguments: str, timeout: float = 5
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
        shell=False,
        timeout=timeout,
    )


def _git_metadata(root: Path, base_commit: str | None) -> dict[str, Any]:
    repository_check = _git(root, "rev-parse", "--show-toplevel")
    if repository_check.returncode != 0:
        return {
            "repository": None,
            "git_ref": None,
            "base_commit": None,
            "head_commit": None,
            "dirty": False,
            "changed_paths": [],
        }
    repository = str(Path(repository_check.stdout.strip()).resolve())
    head_result = _git(root, "rev-parse", "HEAD")
    head_commit = head_result.stdout.strip() if head_result.returncode == 0 else None
    ref_result = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    git_ref = ref_result.stdout.strip() if ref_result.returncode == 0 else None
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    dirty = bool(status.stdout.strip())
    changed: set[str] = set()
    diff_base = base_commit
    if diff_base:
        diff = _git(root, "diff", "--name-only", diff_base)
        if diff.returncode == 0:
            changed.update(
                PurePosixPath(line.strip()).as_posix()
                for line in diff.stdout.splitlines()
                if line.strip()
            )
    for line in status.stdout.splitlines():
        if len(line) > 3:
            path = line[3:].split(" -> ")[-1].strip()
            if path:
                changed.add(PurePosixPath(path).as_posix())
    return {
        "repository": repository,
        "git_ref": git_ref,
        "base_commit": base_commit,
        "head_commit": head_commit,
        "dirty": dirty,
        "changed_paths": sorted(changed),
    }


def _load_codeowners(root: Path) -> list[tuple[str, tuple[str, ...]]]:
    candidates = (
        root / ".github" / "CODEOWNERS",
        root / "docs" / "CODEOWNERS",
        root / "CODEOWNERS",
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        rules = []
        for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines():
            content = line.strip()
            if not content or content.startswith("#"):
                continue
            parts = content.split()
            if len(parts) >= 2:
                rules.append((parts[0].lstrip("/"), tuple(parts[1:])))
        return rules
    return []


def _owners_for(
    relative_path: str, rules: list[tuple[str, tuple[str, ...]]]
) -> tuple[str, ...]:
    owners: tuple[str, ...] = ()
    for pattern, candidate_owners in rules:
        normalized = pattern.rstrip("/")
        if fnmatch.fnmatch(relative_path, normalized) or relative_path.startswith(
            f"{normalized}/"
        ):
            owners = candidate_owners
    return owners


def _classify(relative_path: str) -> str:
    path = PurePosixPath(relative_path)
    lower_parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    if "test" in lower_parts or "tests" in lower_parts or name.startswith("test_"):
        return "test"
    if path.suffix.lower() in {".md", ".rst"} or "docs" in lower_parts:
        return "documentation"
    if name in {
        "pyproject.toml",
        "package.json",
        "cargo.toml",
        "go.mod",
        "dockerfile",
        "makefile",
    } or path.suffix.lower() in {".yaml", ".yml", ".toml", ".json"}:
        return "configuration"
    return "source"


class _PythonSymbolVisitor(ast.NodeVisitor):
    def __init__(self, file_id: str, snapshot_id: str, module: str) -> None:
        self.file_id = file_id
        self.snapshot_id = snapshot_id
        self.module = module
        self.scope: list[str] = []
        self.symbols: list[dict[str, Any]] = []
        self.calls: list[tuple[str, str]] = []

    def _visit_symbol(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, kind: str
    ) -> None:
        qualified = ".".join((self.module, *self.scope, node.name)).strip(".")
        symbol_id = hashlib.sha256(
            f"{self.snapshot_id}\0{self.file_id}\0{qualified}\0{node.lineno}".encode()
        ).hexdigest()
        self.symbols.append(
            {
                "symbol_id": symbol_id,
                "file_id": self.file_id,
                "qualified_name": qualified,
                "symbol_kind": kind,
                "start_line": node.lineno,
                "end_line": getattr(node, "end_lineno", node.lineno),
                "signature": node.name,
                "confidence": 1.0,
            }
        )
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_symbol(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_symbol(node, "async_function")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_symbol(node, "class")

    def visit_Call(self, node: ast.Call) -> None:
        if self.scope:
            target = _call_name(node.func)
            if target:
                source = ".".join((self.module, *self.scope)).strip(".")
                self.calls.append((source, target))
        self.generic_visit(node)


def _extract_symbols(
    scanned: list[_ScannedFile], snapshot_id: str
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[tuple[str, str]]],
    dict[str, ast.Module],
]:
    symbols: list[dict[str, Any]] = []
    calls: dict[str, list[tuple[str, str]]] = {}
    trees: dict[str, ast.Module] = {}
    for item in scanned:
        if item.language != "python":
            continue
        try:
            tree = ast.parse(item.text, filename=item.relative_path)
        except SyntaxError:
            continue
        module = item.relative_path.removesuffix(item.path.suffix).replace("/", ".")
        if module.endswith(".__init__"):
            module = module.removesuffix(".__init__")
        visitor = _PythonSymbolVisitor(item.file_id, snapshot_id, module)
        visitor.visit(tree)
        trees[item.file_id] = tree
        symbols.extend(visitor.symbols)
        calls[item.file_id] = visitor.calls
    return symbols, calls, trees


def _extract_edges(
    scanned: list[_ScannedFile],
    symbols: list[dict[str, Any]],
    calls_by_file: dict[str, list[tuple[str, str]]],
    python_trees: dict[str, ast.Module],
    snapshot_id: str,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    module_files: dict[str, str] = {}
    path_files = {item.relative_path: item.file_id for item in scanned}
    for item in scanned:
        if item.language == "python":
            module = item.relative_path.removesuffix(item.path.suffix).replace("/", ".")
            module_files[module.removesuffix(".__init__")] = item.file_id
    for item in scanned:
        targets: set[str] = set()
        if item.language == "python":
            tree = python_trees.get(item.file_id)
            if tree is not None:
                source_module = item.relative_path.removesuffix(
                    item.path.suffix
                ).replace("/", ".")
                source_package = source_module.removesuffix(".__init__").split(".")
                if not source_module.endswith(".__init__"):
                    source_package = source_package[:-1]
                for node in ast.walk(tree):
                    names: list[str] = []
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        if node.level:
                            keep = max(0, len(source_package) - node.level + 1)
                            module = ".".join((*source_package[:keep], module)).strip(".")
                        if module:
                            names = [module]
                    for name in names:
                        parts = name.split(".")
                        for length in range(len(parts), 0, -1):
                            target = module_files.get(".".join(parts[:length]))
                            if target:
                                targets.add(target)
                                break
        elif item.language in {"javascript", "typescript", "vue", "svelte"}:
            for imported in _JS_IMPORT.findall(item.text):
                if not imported.startswith("."):
                    continue
                base = (PurePosixPath(item.relative_path).parent / imported).as_posix()
                candidates = (
                    base,
                    *(f"{base}{suffix}" for suffix in (".js", ".jsx", ".ts", ".tsx")),
                    *(f"{base}/index{suffix}" for suffix in (".js", ".ts")),
                )
                target = next(
                    (path_files[candidate] for candidate in candidates if candidate in path_files),
                    None,
                )
                if target:
                    targets.add(target)
        for target in sorted(targets):
            if target != item.file_id:
                edges.append(_edge(snapshot_id, item.file_id, target, "imports", 0.95))

    symbol_by_qualified = {
        str(symbol["qualified_name"]): str(symbol["symbol_id"]) for symbol in symbols
    }
    symbols_by_suffix: dict[str, list[str]] = {}
    for qualified, symbol_id in symbol_by_qualified.items():
        symbols_by_suffix.setdefault(qualified.rsplit(".", 1)[-1], []).append(symbol_id)
    for calls in calls_by_file.values():
        for source_name, target_name in calls:
            source_id = symbol_by_qualified.get(source_name)
            candidates = symbols_by_suffix.get(target_name.rsplit(".", 1)[-1], [])
            if source_id and len(candidates) == 1 and candidates[0] != source_id:
                edges.append(
                    _edge(snapshot_id, source_id, candidates[0], "calls", 0.7)
                )
    # Repeated calls between the same symbols are one graph relationship.
    # The stable edge identity intentionally excludes source locations, so
    # collapse duplicates before the SQLite primary-key boundary.
    unique_edges = {str(edge["edge_id"]): edge for edge in edges}
    return [unique_edges[edge_id] for edge_id in sorted(unique_edges)]


def _edge(
    snapshot_id: str,
    source_id: str,
    target_id: str,
    edge_kind: str,
    confidence: float,
) -> dict[str, Any]:
    edge_id = hashlib.sha256(
        f"{snapshot_id}\0{source_id}\0{target_id}\0{edge_kind}".encode()
    ).hexdigest()
    return {
        "edge_id": edge_id,
        "source_id": source_id,
        "target_id": target_id,
        "edge_kind": edge_kind,
        "confidence": confidence,
    }


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _hash_json(value: Any) -> str:
    serialized = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def _snapshot_from_record(record: dict[str, Any]) -> RepositorySnapshot:
    metadata = record["metadata"]
    return RepositorySnapshot(
        snapshot_id=record["snapshot_id"],
        project_id=record["project_id"],
        canonical_path=record["canonical_path"],
        identity_hash=record["identity_hash"],
        configuration_hash=record["configuration_hash"],
        scanner_version=record["scanner_version"],
        created_at=datetime.fromisoformat(record["created_at"]),
        git_repository=record["git_repository"],
        git_ref=record.get("git_ref"),
        base_commit=record["base_commit"],
        head_commit=record["head_commit"],
        parent_snapshot_id=record.get("parent_snapshot_id"),
        dirty=record["dirty"],
        included_paths=tuple(metadata.get("included_paths", ())),
    )


def _assign_file_ids(scanned: list[_ScannedFile], snapshot_id: str) -> None:
    for item in scanned:
        item.file_id = hashlib.sha256(
            f"{snapshot_id}\0{item.relative_path}".encode()
        ).hexdigest()
