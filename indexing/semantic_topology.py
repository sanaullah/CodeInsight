"""Conservative, evidence-derived semantic topology extraction.

The extractor intentionally recognizes a small set of static conventions. It
never treats imports as runtime flow and never resolves dynamic route, resource,
queue, or URL expressions by guessing.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from domain.architecture import (
    ArchitectureCompleteness,
    ArchitectureSupportTier,
    ComponentKind,
    ProvenanceSpan,
    RelationKind,
    SemanticArchitectureProjection,
    SemanticBoundary,
    SemanticComponent,
    SemanticEndpoint,
    SemanticMembership,
    SemanticRelation,
    SemanticResource,
)

SEMANTIC_EXTRACTOR_VERSION = "semantic-topology-v1"

_JS_LANGUAGES = frozenset({"javascript", "typescript", "jsx", "tsx"})
_HTTP_METHODS = frozenset(
    {"delete", "get", "head", "options", "patch", "post", "put"}
)
_SQL_TABLE = re.compile(
    r"\b(?:FROM|JOIN|INTO|UPDATE|TABLE)\s+"
    r"([A-Za-z_][A-Za-z0-9_.-]*)",
    re.IGNORECASE,
)
_EXPRESS_ROUTE = re.compile(
    r"\b(?:app|router)\.(get|post|put|patch|delete|options|head|all)"
    r"\s*\(\s*(['\"])(/[^'\"]*)\2"
)
_JS_HTTP = re.compile(
    r"\b(?:fetch|axios\.(?:get|post|put|patch|delete)|"
    r"(?:http|https)\.(?:get|request))"
    r"\s*\(\s*(['\"])(https?://[^'\"]+)\1"
)
_JS_SQL = re.compile(
    r"\b(?:query|execute)\s*\(\s*(['\"])([^'\"]+)\1",
    re.IGNORECASE,
)
_JS_QUEUE = re.compile(
    r"\b(?:queue\.add|sendToQueue|publish)\s*\(\s*(['\"])([^'\"]+)\1"
)
_JS_KAFKA_TOPIC = re.compile(
    r"\btopic\s*:\s*(['\"])([^'\"]+)\1",
)


@dataclass(frozen=True, slots=True)
class SemanticFile:
    file_id: str
    relative_path: str
    language: str
    text: str
    line_count: int


@dataclass(frozen=True, slots=True)
class SemanticTopologyInput:
    snapshot_id: str
    project_name: str
    files: tuple[SemanticFile, ...]


def extract_semantic_topology(
    inputs: SemanticTopologyInput,
) -> SemanticArchitectureProjection:
    """Extract a deterministic projection from immutable source-file inputs."""

    builder = _ProjectionBuilder(inputs)
    builder.extract()
    return builder.projection()


class _ProjectionBuilder:
    def __init__(self, source: SemanticTopologyInput) -> None:
        self.source = source
        self.components: dict[str, SemanticComponent] = {}
        self.memberships: dict[str, SemanticMembership] = {}
        self.resources: dict[str, SemanticResource] = {}
        self.endpoints: dict[str, SemanticEndpoint] = {}
        self.relations: dict[str, SemanticRelation] = {}
        self.provenance: dict[str, ProvenanceSpan] = {}
        self._component_evidence: dict[str, tuple[SemanticFile, int, str]] = {}
        self._service_key = f"service:{_slug(source.project_name)}"
        self._service_id = _entity_id(
            source.snapshot_id, "component", self._service_key
        )
        self._service_reasons: set[str] = set()
        self._service_support = ArchitectureSupportTier.EXACT
        self._service_completeness = ArchitectureCompleteness.COMPLETE

    def extract(self) -> None:
        supported_seen = False
        for file in sorted(self.source.files, key=lambda item: item.relative_path):
            language = file.language.lower()
            if language == "python":
                supported_seen = True
                self._extract_python(file)
            elif language in _JS_LANGUAGES:
                supported_seen = True
                self._service_support = ArchitectureSupportTier.PARTIAL
                self._service_completeness = ArchitectureCompleteness.PARTIAL
                self._service_reasons.add("javascript-typescript-lexical-support")
                self._extract_javascript(file)
            else:
                self._service_completeness = ArchitectureCompleteness.PARTIAL
                self._service_reasons.add(f"unsupported-language:{language or 'unknown'}")
                self._add_unknown_language(file)

        if supported_seen:
            evidence_file = self._first_file()
            if evidence_file is not None:
                service_id = self._add_component(
                    self._service_key,
                    self.source.project_name,
                    ComponentKind.SERVICE,
                    self._service_support,
                    self._service_completeness,
                    1.0 if self._service_support is ArchitectureSupportTier.EXACT else 0.8,
                    evidence_file,
                    1,
                    "project-source-membership",
                    {"partial_reasons": sorted(self._service_reasons)},
                )
                for file in sorted(
                    self.source.files, key=lambda item: item.relative_path
                ):
                    if file.language.lower() in {"python", *_JS_LANGUAGES}:
                        self._add_membership(
                            service_id,
                            file,
                            1,
                            "project-source",
                            1.0,
                            "project-source-membership",
                        )

    def _extract_python(self, file: SemanticFile) -> None:
        try:
            tree = ast.parse(file.text, filename=file.relative_path)
        except SyntaxError as exc:
            self._service_support = ArchitectureSupportTier.PARTIAL
            self._service_completeness = ArchitectureCompleteness.PARTIAL
            self._service_reasons.add(
                f"python-parse-error:{file.relative_path}"
            )
            key = f"unknown-source:{_normalize_path(file.relative_path)}"
            component_id = self._add_component(
                key,
                PurePosixPath(file.relative_path).name,
                ComponentKind.UNKNOWN,
                ArchitectureSupportTier.PARTIAL,
                ArchitectureCompleteness.UNKNOWN,
                0.0,
                file,
                max(1, exc.lineno or 1),
                "python-parse-error",
                {"reason": "syntax-error"},
            )
            self._add_membership(
                component_id,
                file,
                max(1, exc.lineno or 1),
                "unparsed-source",
                0.0,
                "python-parse-error",
            )
            return

        imported = _python_imports(tree)
        framework = (
            "fastapi"
            if "fastapi" in imported
            else "flask"
            if "flask" in imported
            else None
        )
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and framework:
                for decorator in node.decorator_list:
                    self._extract_python_route(file, decorator, framework)
            if isinstance(node, ast.Call):
                self._extract_python_http(file, node, imported)
                self._extract_python_database(file, node, imported)
                self._extract_python_queue(file, node, imported)
            if isinstance(node, ast.ClassDef):
                self._extract_python_table(file, node)

    def _extract_python_route(
        self, file: SemanticFile, decorator: ast.expr, framework: str
    ) -> None:
        if not isinstance(decorator, ast.Call):
            return
        call_path = _call_path(decorator.func)
        action = call_path.rsplit(".", 1)[-1].lower()
        if action not in _HTTP_METHODS | {"api_route", "route"}:
            return
        route = _literal_string(decorator.args[0]) if decorator.args else None
        if route is None or not route.startswith("/"):
            self._service_completeness = ArchitectureCompleteness.PARTIAL
            self._service_reasons.add(
                f"dynamic-{framework}-route:{file.relative_path}"
            )
            return
        methods: list[str]
        if action in _HTTP_METHODS:
            methods = [action.upper()]
        else:
            methods = _literal_string_list(_keyword(decorator, "methods"))
            if not methods:
                methods = ["GET"] if framework == "flask" else ["*"]
        for method in sorted(set(methods)):
            self._add_endpoint(
                self._service_id,
                "http",
                method,
                route,
                "inbound",
                ArchitectureSupportTier.EXACT,
                ArchitectureCompleteness.COMPLETE,
                1.0,
                file,
                decorator.lineno,
                f"{framework}-route",
            )

    def _extract_python_http(
        self, file: SemanticFile, node: ast.Call, imported: set[str]
    ) -> None:
        if not imported.intersection({"requests", "httpx", "aiohttp"}):
            return
        path = _call_path(node.func)
        action = path.rsplit(".", 1)[-1].lower()
        if action not in _HTTP_METHODS | {"request"}:
            return
        url = _literal_string(node.args[0]) if node.args else None
        if action == "request" and len(node.args) >= 2:
            url = _literal_string(node.args[1])
        if not url or not url.startswith(("http://", "https://")):
            self._service_completeness = ArchitectureCompleteness.PARTIAL
            self._service_reasons.add(
                f"dynamic-outbound-http:{file.relative_path}"
            )
            return
        method = (
            (_literal_string(node.args[0]) or "*").upper()
            if action == "request"
            else action.upper()
        )
        self._add_external_request(file, node.lineno, method, url, ArchitectureSupportTier.EXACT)

    def _extract_python_database(
        self, file: SemanticFile, node: ast.Call, imported: set[str]
    ) -> None:
        database_imports = {
            "sqlite3": "sqlite",
            "sqlalchemy": "database",
            "psycopg": "postgresql",
            "psycopg2": "postgresql",
            "asyncpg": "postgresql",
        }
        engines = {
            engine
            for module, engine in database_imports.items()
            if module in imported
        }
        if not engines:
            return
        path = _call_path(node.func)
        action = path.rsplit(".", 1)[-1].lower()
        if action in {"connect", "create_engine", "create_async_engine"}:
            locator = _literal_string(node.args[0]) if node.args else None
            engine = _database_engine(locator) or sorted(engines)[0]
            self._add_datastore(file, node.lineno, engine, locator)
        if action in {"execute", "executemany", "text"} and node.args:
            sql = _literal_string(node.args[0])
            if sql:
                for table in sorted(set(_SQL_TABLE.findall(sql))):
                    self._add_table_resource(file, node.lineno, table)

    def _extract_python_table(self, file: SemanticFile, node: ast.ClassDef) -> None:
        for statement in node.body:
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            if not any(
                isinstance(target, ast.Name) and target.id == "__tablename__"
                for target in targets
            ):
                continue
            value = _literal_string(statement.value)
            if value:
                self._add_table_resource(file, statement.lineno, value)

    def _extract_python_queue(
        self, file: SemanticFile, node: ast.Call, imported: set[str]
    ) -> None:
        if not imported.intersection(
            {"celery", "kafka", "aiokafka", "pika", "boto3"}
        ):
            return
        action = _call_path(node.func).rsplit(".", 1)[-1]
        queue_name: str | None = None
        if action in {"send_task", "publish", "send", "send_message"}:
            queue_name = _literal_string(node.args[0]) if node.args else None
            queue_name = (
                queue_name
                or _literal_string(_keyword(node, "topic"))
                or _literal_string(_keyword(node, "QueueUrl"))
            )
        if queue_name:
            self._add_queue(file, node.lineno, queue_name, ArchitectureSupportTier.EXACT)
        elif action in {"send_task", "publish", "send", "send_message"}:
            self._service_completeness = ArchitectureCompleteness.PARTIAL
            self._service_reasons.add(f"dynamic-queue:{file.relative_path}")

    def _extract_javascript(self, file: SemanticFile) -> None:
        text = _mask_js_comments(file.text)
        if re.search(r"\b(?:from\s+['\"]express['\"]|require\(['\"]express['\"]\))", text):
            for match in _EXPRESS_ROUTE.finditer(text):
                action, _quote, route = match.groups()
                methods = sorted(_HTTP_METHODS) if action == "all" else [action]
                for method in methods:
                    self._add_endpoint(
                        self._service_id,
                        "http",
                        method.upper(),
                        route,
                        "inbound",
                        ArchitectureSupportTier.PARTIAL,
                        ArchitectureCompleteness.PARTIAL,
                        0.8,
                        file,
                        _line_number(text, match.start()),
                        "express-route-lexical",
                    )

        http_imports = bool(
            re.search(
                r"(?:from\s+['\"](?:axios|undici|node:https?|https?)['\"]|"
                r"require\(['\"](?:axios|undici|node:https?|https?)['\"]\))",
                text,
            )
        )
        for match in _JS_HTTP.finditer(text):
            if not match.group(0).lstrip().startswith("fetch") and not http_imports:
                continue
            _quote, url = match.groups()
            prefix = text[match.start() : match.start() + 32].lower()
            method_match = re.search(r"\.(get|post|put|patch|delete)", prefix)
            method = method_match.group(1).upper() if method_match else "*"
            self._add_external_request(
                file,
                _line_number(text, match.start()),
                method,
                url,
                ArchitectureSupportTier.PARTIAL,
            )

        database_modules = {
            "@prisma/client": "database",
            "pg": "postgresql",
            "sequelize": "database",
            "typeorm": "database",
        }
        database_support = False
        for module, engine in database_modules.items():
            if re.search(
                rf"(?:from\s+['\"]{re.escape(module)}['\"]|"
                rf"require\(['\"]{re.escape(module)}['\"]\))",
                text,
            ):
                database_support = True
                match = re.search(re.escape(module), text)
                assert match is not None
                self._add_datastore(
                    file,
                    _line_number(text, match.start()),
                    engine,
                    None,
                    support=ArchitectureSupportTier.PARTIAL,
                )
        if database_support:
            for match in _JS_SQL.finditer(text):
                sql = match.group(2)
                for table in sorted(set(_SQL_TABLE.findall(sql))):
                    self._add_table_resource(
                        file,
                        _line_number(text, match.start()),
                        table,
                        support=ArchitectureSupportTier.PARTIAL,
                    )

        queue_support = bool(
            re.search(
                r"(?:from\s+['\"](?:kafkajs|bullmq|amqplib)['\"]|"
                r"require\(['\"](?:kafkajs|bullmq|amqplib)['\"]\))",
                text,
            )
        )
        if queue_support:
            for pattern in (_JS_QUEUE, _JS_KAFKA_TOPIC):
                for match in pattern.finditer(text):
                    self._add_queue(
                        file,
                        _line_number(text, match.start()),
                        match.group(2),
                        ArchitectureSupportTier.PARTIAL,
                    )

    def _add_external_request(
        self,
        file: SemanticFile,
        line: int,
        method: str,
        url: str,
        support: ArchitectureSupportTier,
    ) -> None:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
        if not host:
            return
        component_key = f"external:{host}"
        external_id = self._add_component(
            component_key,
            host,
            ComponentKind.EXTERNAL_SYSTEM,
            support,
            ArchitectureCompleteness.COMPLETE
            if support is ArchitectureSupportTier.EXACT
            else ArchitectureCompleteness.PARTIAL,
            1.0 if support is ArchitectureSupportTier.EXACT else 0.8,
            file,
            line,
            "outbound-http-host",
            {"host": host},
        )
        self._add_membership(
            external_id, file, line, "client-reference", 1.0, "outbound-http-host"
        )
        self._add_endpoint(
            external_id,
            parsed.scheme,
            method,
            parsed.path or "/",
            "outbound",
            support,
            ArchitectureCompleteness.COMPLETE
            if support is ArchitectureSupportTier.EXACT
            else ArchitectureCompleteness.PARTIAL,
            1.0 if support is ArchitectureSupportTier.EXACT else 0.8,
            file,
            line,
            "outbound-http-literal",
            metadata={"host": host},
        )
        self._add_relation(
            self._service_id,
            external_id,
            RelationKind.REQUEST,
            parsed.scheme,
            False,
            support,
            1.0 if support is ArchitectureSupportTier.EXACT else 0.8,
            file,
            line,
            "outbound-http-literal",
        )

    def _add_datastore(
        self,
        file: SemanticFile,
        line: int,
        engine: str,
        locator: str | None,
        *,
        support: ArchitectureSupportTier = ArchitectureSupportTier.EXACT,
    ) -> None:
        safe_locator = _safe_database_locator(locator)
        stable_locator = safe_locator or engine
        key = f"datastore:{_slug(stable_locator)}"
        datastore_id = self._add_component(
            key,
            engine,
            ComponentKind.DATASTORE,
            support,
            ArchitectureCompleteness.COMPLETE
            if support is ArchitectureSupportTier.EXACT
            else ArchitectureCompleteness.PARTIAL,
            1.0 if support is ArchitectureSupportTier.EXACT else 0.75,
            file,
            line,
            "database-client",
            {"engine": engine},
        )
        self._add_membership(
            datastore_id, file, line, "client-reference", 1.0, "database-client"
        )
        self._add_resource(
            datastore_id,
            f"database:{_slug(stable_locator)}",
            "database",
            engine,
            safe_locator,
            support,
            file,
            line,
            "database-client",
        )
        self._add_relation(
            self._service_id,
            datastore_id,
            RelationKind.DATA_ACCESS,
            engine,
            False,
            support,
            1.0 if support is ArchitectureSupportTier.EXACT else 0.75,
            file,
            line,
            "database-client",
        )

    def _add_table_resource(
        self,
        file: SemanticFile,
        line: int,
        table: str,
        *,
        support: ArchitectureSupportTier = ArchitectureSupportTier.EXACT,
    ) -> None:
        normalized = table.strip("`\"[]").lower()
        if not re.fullmatch(r"[a-z_][a-z0-9_.-]*", normalized):
            return
        datastore_key = "datastore:database"
        datastore_id = self._add_component(
            datastore_key,
            "database",
            ComponentKind.DATASTORE,
            support,
            ArchitectureCompleteness.PARTIAL,
            0.9 if support is ArchitectureSupportTier.EXACT else 0.7,
            file,
            line,
            "sql-resource",
            {"engine": "unknown"},
        )
        self._add_membership(
            datastore_id, file, line, "resource-reference", 1.0, "sql-resource"
        )
        self._add_resource(
            datastore_id,
            f"table:{normalized}",
            "table",
            normalized,
            normalized,
            support,
            file,
            line,
            "sql-resource",
        )
        self._add_relation(
            self._service_id,
            datastore_id,
            RelationKind.DATA_ACCESS,
            None,
            False,
            support,
            0.9 if support is ArchitectureSupportTier.EXACT else 0.7,
            file,
            line,
            "sql-resource",
        )

    def _add_queue(
        self,
        file: SemanticFile,
        line: int,
        queue_name: str,
        support: ArchitectureSupportTier,
    ) -> None:
        key = f"queue:{_slug(queue_name)}"
        queue_id = self._add_component(
            key,
            queue_name,
            ComponentKind.QUEUE,
            support,
            ArchitectureCompleteness.COMPLETE
            if support is ArchitectureSupportTier.EXACT
            else ArchitectureCompleteness.PARTIAL,
            1.0 if support is ArchitectureSupportTier.EXACT else 0.75,
            file,
            line,
            "queue-literal",
        )
        self._add_membership(
            queue_id, file, line, "producer-reference", 1.0, "queue-literal"
        )
        self._add_resource(
            queue_id,
            f"queue-resource:{_slug(queue_name)}",
            "queue",
            queue_name,
            queue_name,
            support,
            file,
            line,
            "queue-literal",
        )
        self._add_relation(
            self._service_id,
            queue_id,
            RelationKind.EVENT,
            "queue",
            True,
            support,
            1.0 if support is ArchitectureSupportTier.EXACT else 0.75,
            file,
            line,
            "queue-literal",
        )

    def _add_unknown_language(self, file: SemanticFile) -> None:
        language = file.language.lower() or "unknown"
        key = f"unknown-language:{_slug(language)}"
        component_id = self._add_component(
            key,
            f"Unsupported {language}",
            ComponentKind.UNKNOWN,
            ArchitectureSupportTier.UNSUPPORTED,
            ArchitectureCompleteness.UNSUPPORTED,
            0.0,
            file,
            1,
            "unsupported-language",
            {"language": language, "reason": "no-semantic-extractor"},
        )
        self._add_membership(
            component_id,
            file,
            1,
            "unsupported-source",
            0.0,
            "unsupported-language",
        )

    def _add_component(
        self,
        key: str,
        name: str,
        kind: ComponentKind,
        support: ArchitectureSupportTier,
        completeness: ArchitectureCompleteness,
        confidence: float,
        file: SemanticFile,
        line: int,
        derivation: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        component_id = _entity_id(self.source.snapshot_id, "component", key)
        existing = self.components.get(key)
        if existing is None:
            self.components[key] = SemanticComponent(
                component_id=component_id,
                snapshot_id=self.source.snapshot_id,
                stable_key=key,
                name=name,
                component_kind=kind,
                support_tier=support,
                completeness=completeness,
                confidence=confidence,
                metadata=metadata or {},
            )
            self._component_evidence[component_id] = (file, line, derivation)
        else:
            self.components[key] = existing.model_copy(
                update={
                    "support_tier": _stronger_support(
                        existing.support_tier, support
                    ),
                    "completeness": _weaker_completeness(
                        existing.completeness, completeness
                    ),
                    "confidence": max(existing.confidence, confidence),
                }
            )
        self._add_provenance(
            "component", component_id, file, line, derivation, confidence
        )
        return component_id

    def _add_membership(
        self,
        component_id: str,
        file: SemanticFile,
        line: int,
        kind: str,
        confidence: float,
        derivation: str,
    ) -> None:
        key = f"{component_id}\0{file.file_id}\0{kind}"
        membership_id = _entity_id(self.source.snapshot_id, "membership", key)
        if key not in self.memberships:
            self.memberships[key] = SemanticMembership(
                membership_id=membership_id,
                snapshot_id=self.source.snapshot_id,
                component_id=component_id,
                file_id=file.file_id,
                membership_kind=kind,
                confidence=confidence,
                provenance={"derivation": derivation, "line": line},
            )
        self._add_provenance(
            "membership", membership_id, file, line, derivation, confidence
        )

    def _add_resource(
        self,
        component_id: str,
        key: str,
        kind: str,
        name: str,
        locator: str | None,
        support: ArchitectureSupportTier,
        file: SemanticFile,
        line: int,
        derivation: str,
    ) -> None:
        resource_id = _entity_id(self.source.snapshot_id, "resource", key)
        if key not in self.resources:
            self.resources[key] = SemanticResource(
                resource_id=resource_id,
                snapshot_id=self.source.snapshot_id,
                component_id=component_id,
                stable_key=key,
                resource_kind=kind,
                name=name,
                locator=locator,
                support_tier=support,
                completeness=ArchitectureCompleteness.COMPLETE
                if support is ArchitectureSupportTier.EXACT
                else ArchitectureCompleteness.PARTIAL,
                confidence=1.0 if support is ArchitectureSupportTier.EXACT else 0.75,
            )
        self._add_provenance(
            "resource", resource_id, file, line, derivation, 1.0
        )

    def _add_endpoint(
        self,
        component_id: str,
        protocol: str,
        method: str,
        route: str,
        direction: str,
        support: ArchitectureSupportTier,
        completeness: ArchitectureCompleteness,
        confidence: float,
        file: SemanticFile,
        line: int,
        derivation: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        component_key = self._component_stable_key(component_id)
        key = (
            f"endpoint:{direction}:{protocol.lower()}:{method.upper()}:"
            f"{_normalize_route(route)}:{component_key}"
        )
        endpoint_id = _entity_id(self.source.snapshot_id, "endpoint", key)
        if key not in self.endpoints:
            self.endpoints[key] = SemanticEndpoint(
                endpoint_id=endpoint_id,
                snapshot_id=self.source.snapshot_id,
                component_id=component_id,
                stable_key=key,
                protocol=protocol.lower(),
                method=method.upper(),
                route=_normalize_route(route),
                direction=direction,  # type: ignore[arg-type]
                support_tier=support,
                completeness=completeness,
                confidence=confidence,
                metadata=metadata or {},
            )
        else:
            existing = self.endpoints[key]
            self.endpoints[key] = existing.model_copy(
                update={
                    "support_tier": _stronger_support(
                        existing.support_tier, support
                    ),
                    "completeness": _weaker_completeness(
                        existing.completeness, completeness
                    ),
                    "confidence": max(existing.confidence, confidence),
                }
            )
        self._add_provenance(
            "endpoint", endpoint_id, file, line, derivation, confidence
        )

    def _add_relation(
        self,
        source_id: str,
        target_id: str,
        kind: RelationKind,
        transport: str | None,
        is_async: bool,
        support: ArchitectureSupportTier,
        confidence: float,
        file: SemanticFile,
        line: int,
        derivation: str,
    ) -> None:
        source_key = self._component_stable_key(source_id)
        target_key = self._component_stable_key(target_id)
        key = (
            f"relation:{kind.value}:{source_key}:{target_key}:"
            f"{transport or 'unknown'}:{int(is_async)}"
        )
        relation_id = _entity_id(self.source.snapshot_id, "relation", key)
        if key not in self.relations:
            self.relations[key] = SemanticRelation(
                relation_id=relation_id,
                snapshot_id=self.source.snapshot_id,
                source_component_id=source_id,
                target_component_id=target_id,
                stable_key=key,
                relation_kind=kind,
                transport=transport,
                is_async=is_async,
                support_tier=support,
                completeness=ArchitectureCompleteness.COMPLETE
                if support is ArchitectureSupportTier.EXACT
                else ArchitectureCompleteness.PARTIAL,
                confidence=confidence,
            )
        else:
            existing = self.relations[key]
            self.relations[key] = existing.model_copy(
                update={
                    "support_tier": _stronger_support(
                        existing.support_tier, support
                    ),
                    "completeness": _weaker_completeness(
                        existing.completeness,
                        ArchitectureCompleteness.COMPLETE
                        if support is ArchitectureSupportTier.EXACT
                        else ArchitectureCompleteness.PARTIAL,
                    ),
                    "confidence": max(existing.confidence, confidence),
                }
            )
        self._add_provenance(
            "relation", relation_id, file, line, derivation, confidence
        )

    def _add_provenance(
        self,
        entity_kind: str,
        entity_id: str,
        file: SemanticFile,
        line: int,
        derivation: str,
        confidence: float,
    ) -> None:
        line = max(1, min(line, max(1, file.line_count)))
        key = f"{entity_kind}\0{entity_id}\0{file.file_id}\0{line}\0{derivation}"
        if key in self.provenance:
            return
        provenance_id = _entity_id(self.source.snapshot_id, "provenance", key)
        self.provenance[key] = ProvenanceSpan(
            provenance_id=provenance_id,
            entity_kind=entity_kind,  # type: ignore[arg-type]
            entity_id=entity_id,
            file_id=file.file_id,
            relative_path=file.relative_path,
            start_line=line,
            end_line=line,
            derivation=derivation,
            extractor_version=SEMANTIC_EXTRACTOR_VERSION,
            confidence=confidence,
        )

    def projection(self) -> SemanticArchitectureProjection:
        boundaries: list[SemanticBoundary] = []
        groups = (
            ("application", "Application", {ComponentKind.SERVICE, ComponentKind.LIBRARY}),
            ("data", "Data layer", {ComponentKind.DATASTORE}),
            ("messaging", "Messaging", {ComponentKind.QUEUE}),
            ("external", "External systems", {ComponentKind.EXTERNAL_SYSTEM}),
        )
        for key, name, kinds in groups:
            component_ids = tuple(
                sorted(
                    component.component_id
                    for component in self.components.values()
                    if component.component_kind in kinds
                )
            )
            if not component_ids:
                continue
            boundary_key = f"boundary:{key}"
            boundary_id = _entity_id(
                self.source.snapshot_id, "boundary", boundary_key
            )
            boundaries.append(
                SemanticBoundary(
                    boundary_id=boundary_id,
                    snapshot_id=self.source.snapshot_id,
                    stable_key=boundary_key,
                    name=name,
                    boundary_kind=key,
                    support_tier=ArchitectureSupportTier.INFERRED,
                    completeness=ArchitectureCompleteness.PARTIAL,
                    confidence=0.8,
                    component_ids=component_ids,
                    metadata={"derivation": "typed-component-group"},
                )
            )
            evidence = self._component_evidence.get(component_ids[0])
            if evidence:
                file, line, _derivation = evidence
                self._add_provenance(
                    "boundary",
                    boundary_id,
                    file,
                    line,
                    "typed-component-group",
                    0.8,
                )
        return SemanticArchitectureProjection(
            snapshot_id=self.source.snapshot_id,
            extractor_version=SEMANTIC_EXTRACTOR_VERSION,
            components=tuple(
                sorted(self.components.values(), key=lambda item: item.stable_key)
            ),
            memberships=tuple(
                sorted(self.memberships.values(), key=lambda item: item.membership_id)
            ),
            boundaries=tuple(sorted(boundaries, key=lambda item: item.stable_key)),
            resources=tuple(
                sorted(self.resources.values(), key=lambda item: item.stable_key)
            ),
            endpoints=tuple(
                sorted(self.endpoints.values(), key=lambda item: item.stable_key)
            ),
            relations=tuple(
                sorted(self.relations.values(), key=lambda item: item.stable_key)
            ),
            provenance=tuple(
                sorted(
                    self.provenance.values(),
                    key=lambda item: (
                        item.relative_path or "",
                        item.start_line,
                        item.entity_kind,
                        item.entity_id,
                    ),
                )
            ),
        )

    def _first_file(self) -> SemanticFile | None:
        return min(
            (
                file
                for file in self.source.files
                if file.language.lower() in {"python", *_JS_LANGUAGES}
            ),
            key=lambda item: item.relative_path,
            default=None,
        )

    def _component_stable_key(self, component_id: str) -> str:
        if component_id == self._service_id:
            return self._service_key
        return next(
            component.stable_key
            for component in self.components.values()
            if component.component_id == component_id
        )


def _entity_id(snapshot_id: str, entity_kind: str, stable_key: str) -> str:
    return hashlib.sha256(
        f"{snapshot_id}\0{entity_kind}\0{stable_key}".encode()
    ).hexdigest()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9._-]+", "-", value.strip().lower()).strip("-")
    return normalized or "unknown"


def _normalize_path(path: str) -> str:
    return PurePosixPath(path.replace("\\", "/")).as_posix().lower()


def _normalize_route(route: str) -> str:
    normalized = "/" + route.strip().lstrip("/")
    return normalized if normalized == "/" else normalized.rstrip("/")


def _python_imports(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".", 1)[0])
    return modules


def _call_path(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_path(node.value)
        return f"{prefix}.{node.attr}".strip(".")
    return ""


def _literal_string(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _literal_string_list(node: ast.AST | None) -> list[str]:
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return []
    values = [_literal_string(item) for item in node.elts]
    return [value.upper() for value in values if value]


def _keyword(node: ast.Call, name: str) -> ast.AST | None:
    return next(
        (keyword.value for keyword in node.keywords if keyword.arg == name), None
    )


def _database_engine(locator: str | None) -> str | None:
    if not locator:
        return None
    scheme = locator.split(":", 1)[0].split("+", 1)[0].lower()
    return {
        "postgres": "postgresql",
        "postgresql": "postgresql",
        "sqlite": "sqlite",
        "mysql": "mysql",
        "mariadb": "mariadb",
    }.get(scheme)


def _safe_database_locator(locator: str | None) -> str | None:
    if not locator:
        return None
    if "://" not in locator:
        return _database_engine(locator) or locator
    parsed = urlsplit(locator)
    if not parsed.scheme:
        return None
    host = parsed.hostname
    path = parsed.path.strip("/")
    safe = f"{parsed.scheme.split('+', 1)[0]}://"
    if host:
        safe += host.lower()
    if path:
        safe += f"/{path}"
    return safe


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _stronger_support(
    left: ArchitectureSupportTier, right: ArchitectureSupportTier
) -> ArchitectureSupportTier:
    rank = {
        ArchitectureSupportTier.UNSUPPORTED: 0,
        ArchitectureSupportTier.PARTIAL: 1,
        ArchitectureSupportTier.INFERRED: 2,
        ArchitectureSupportTier.EXACT: 3,
    }
    return left if rank[left] >= rank[right] else right


def _weaker_completeness(
    left: ArchitectureCompleteness, right: ArchitectureCompleteness
) -> ArchitectureCompleteness:
    rank = {
        ArchitectureCompleteness.UNSUPPORTED: 0,
        ArchitectureCompleteness.UNKNOWN: 1,
        ArchitectureCompleteness.PARTIAL: 2,
        ArchitectureCompleteness.COMPLETE: 3,
    }
    return left if rank[left] <= rank[right] else right


def _mask_js_comments(text: str) -> str:
    """Replace comments with whitespace while preserving line offsets."""

    output = list(text)
    index = 0
    quote: str | None = None
    while index < len(text):
        character = text[index]
        if quote is not None:
            if character == "\\":
                index += 2
                continue
            if character == quote:
                quote = None
            index += 1
            continue
        if character in {"'", '"', "`"}:
            quote = character
            index += 1
            continue
        if text.startswith("//", index):
            end = text.find("\n", index)
            end = len(text) if end < 0 else end
            for position in range(index, end):
                output[position] = " "
            index = end
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            end = len(text) if end < 0 else end + 2
            for position in range(index, end):
                if output[position] != "\n":
                    output[position] = " "
            index = end
            continue
        index += 1
    return "".join(output)
