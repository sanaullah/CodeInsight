from __future__ import annotations

from indexing.semantic_topology import (
    SEMANTIC_EXTRACTOR_VERSION,
    SemanticFile,
    SemanticTopologyInput,
    extract_semantic_topology,
)


def _file(file_id: str, path: str, language: str, text: str) -> SemanticFile:
    return SemanticFile(
        file_id=file_id,
        relative_path=path,
        language=language,
        text=text,
        line_count=max(1, len(text.splitlines())),
    )


def _extract(*files: SemanticFile, snapshot_id: str = "snapshot-1"):
    return extract_semantic_topology(
        SemanticTopologyInput(
            snapshot_id=snapshot_id,
            project_name="Payments Platform",
            files=tuple(files),
        )
    )


def test_extracts_mixed_python_typescript_topology_with_exact_provenance() -> None:
    projection = _extract(
        _file(
            "py-api",
            "services/api.py",
            "python",
            "from fastapi import FastAPI\n"
            "import httpx\n"
            "import sqlite3\n"
            "from celery import Celery\n"
            "app = FastAPI()\n"
            "@app.post('/payments')\n"
            "async def pay():\n"
            "    sqlite3.connect('sqlite:///orders.db')\n"
            "    await httpx.post('https://api.stripe.com/v1/charges')\n"
            "    celery.send_task('payment.created')\n",
        ),
        _file(
            "ts-worker",
            "workers/consumer.ts",
            "typescript",
            "import express from 'express';\n"
            "import { Kafka } from 'kafkajs';\n"
            "import { PrismaClient } from '@prisma/client';\n"
            "router.get('/health', handler);\n"
            "producer.send({ topic: 'payment.created' });\n"
            "fetch('https://audit.example.com/events');\n",
        ),
    )

    assert projection.extractor_version == SEMANTIC_EXTRACTOR_VERSION
    kinds = {component.component_kind.value for component in projection.components}
    assert {"service", "datastore", "queue", "external_system"} <= kinds
    endpoint_contracts = {
        (endpoint.method, endpoint.route, endpoint.direction)
        for endpoint in projection.endpoints
    }
    assert endpoint_contracts >= {
        ("POST", "/payments", "inbound"),
        ("GET", "/health", "inbound"),
        ("POST", "/v1/charges", "outbound"),
        ("*", "/events", "outbound"),
    }
    relation_contracts = {
        (relation.relation_kind.value, relation.is_async)
        for relation in projection.relations
    }
    assert relation_contracts >= {
        ("request", False),
        ("data_access", False),
        ("event", True),
    }
    assert {boundary.boundary_kind for boundary in projection.boundaries} >= {
        "application",
        "data",
        "messaging",
        "external",
    }
    assert {membership.file_id for membership in projection.memberships} == {
        "py-api",
        "ts-worker",
    }
    assert projection.provenance
    assert all(
        span.start_line >= 1 and span.end_line == span.start_line
        for span in projection.provenance
    )
    assert any(
        span.derivation == "fastapi-route"
        and span.relative_path == "services/api.py"
        and span.start_line == 6
        for span in projection.provenance
    )
    service = next(
        component
        for component in projection.components
        if component.component_kind.value == "service"
    )
    assert service.completeness.value == "partial"
    assert "javascript-typescript-lexical-support" in service.metadata["partial_reasons"]


def test_flask_sql_resources_and_literal_http_are_conservative_and_deduplicated() -> None:
    projection = _extract(
        _file(
            "flask",
            "app.py",
            "python",
            "from flask import Flask\n"
            "import requests\n"
            "import sqlalchemy\n"
            "app = Flask(__name__)\n"
            "@app.route('/orders', methods=['GET', 'POST'])\n"
            "def orders():\n"
            "    db.execute('SELECT * FROM orders JOIN customers ON 1=1')\n"
            "    requests.get('https://inventory.example/api/stock')\n"
            "    requests.get('https://inventory.example/api/stock')\n",
        )
    )

    assert {(item.method, item.route) for item in projection.endpoints} >= {
        ("GET", "/orders"),
        ("POST", "/orders"),
        ("GET", "/api/stock"),
    }
    assert {resource.stable_key for resource in projection.resources} >= {
        "table:orders",
        "table:customers",
    }
    external = [
        item for item in projection.components if item.component_kind.value == "external_system"
    ]
    assert len(external) == 1
    requests = [
        item for item in projection.relations if item.relation_kind.value == "request"
    ]
    assert len(requests) == 1
    endpoint = next(item for item in projection.endpoints if item.direction == "outbound")
    endpoint_spans = [
        span
        for span in projection.provenance
        if span.entity_kind == "endpoint" and span.entity_id == endpoint.endpoint_id
    ]
    assert {span.start_line for span in endpoint_spans} == {8, 9}


def test_dynamic_malformed_and_unsupported_inputs_are_explicit_not_invented() -> None:
    projection = _extract(
        _file(
            "dynamic",
            "dynamic.py",
            "python",
            "from fastapi import FastAPI\n"
            "import httpx\n"
            "app = FastAPI()\n"
            "@app.get(route_name)\n"
            "def dynamic():\n"
            "    httpx.get(base_url + '/items')\n",
        ),
        _file("broken", "broken.py", "python", "def broken(:\n"),
        _file("java", "Service.java", "java", "class Service {}\n"),
    )

    assert projection.endpoints == ()
    assert projection.relations == ()
    by_key = {component.stable_key: component for component in projection.components}
    assert by_key["unknown-language:java"].support_tier.value == "unsupported"
    assert by_key["unknown-language:java"].completeness.value == "unsupported"
    assert by_key["unknown-source:broken.py"].completeness.value == "unknown"
    service = by_key["service:payments-platform"]
    assert service.completeness.value == "partial"
    assert {
        "dynamic-fastapi-route:dynamic.py",
        "dynamic-outbound-http:dynamic.py",
        "python-parse-error:broken.py",
        "unsupported-language:java",
    } <= set(service.metadata["partial_reasons"])
    assert not any(
        component.component_kind.value == "external_system"
        for component in projection.components
    )


def test_stable_keys_survive_snapshots_while_entity_ids_remain_snapshot_scoped() -> None:
    file = _file(
        "api-file",
        "api.py",
        "python",
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        "@app.get('/ready')\n"
        "def ready():\n"
        "    return True\n",
    )
    first = _extract(file, snapshot_id="snapshot-a")
    second = _extract(file, snapshot_id="snapshot-b")

    assert [item.stable_key for item in first.components] == [
        item.stable_key for item in second.components
    ]
    assert [item.stable_key for item in first.endpoints] == [
        item.stable_key for item in second.endpoints
    ]
    assert {item.route for item in first.endpoints} == {item.route for item in second.endpoints}
    assert {item.component_id for item in first.components}.isdisjoint(
        item.component_id for item in second.components
    )
    assert {item.endpoint_id for item in first.endpoints}.isdisjoint(
        item.endpoint_id for item in second.endpoints
    )


def test_javascript_comments_and_unrecognized_calls_do_not_create_claims() -> None:
    projection = _extract(
        _file(
            "ts",
            "app.ts",
            "typescript",
            "import express from 'express';\n"
            "// router.get('/fake', handler)\n"
            "/* fetch('https://fake.example/path') */\n"
            "router.get(dynamicRoute, handler);\n"
            "customClient.get('https://not-proven.example/path');\n",
        )
    )

    assert projection.endpoints == ()
    assert projection.relations == ()
    assert not any(
        item.component_kind.value == "external_system" for item in projection.components
    )
