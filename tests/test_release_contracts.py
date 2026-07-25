from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api.config
from api.app import create_app
from api.config import ApiSettings, load_environment
from indexing.scanners.language_config import (
    Language,
    get_all_dependency_file_patterns,
    get_extensions_for_languages,
    get_language_for_extension,
    get_language_metadata,
    get_supported_languages,
)
from infrastructure.db.database import SCHEMA_VERSION
from infrastructure.scripts import init_database


def test_local_environment_loads_without_overriding_process_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "CODEINSIGHT_DEFAULT_MODEL=from-file\n"
        "CODEINSIGHT_MAX_CONCURRENT_ANALYSES=7\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEINSIGHT_DEFAULT_MODEL", "from-process")

    assert load_environment(env_file)
    settings = ApiSettings.from_environment()

    assert settings.default_model == "from-process"
    assert settings.max_concurrent_analyses == 7
    assert not load_environment(tmp_path / "missing.env")


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("CODEINSIGHT_MAX_CONCURRENT_ANALYSES", "many", "must be an integer"),
        ("CODEINSIGHT_MAX_CONCURRENT_ANALYSES", "0", "greater than zero"),
    ],
)
def test_environment_rejects_invalid_concurrency(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, message: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        ApiSettings.from_environment()


def test_environment_uses_openai_compatible_fallbacks_and_blank_url_is_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_BASE", "http://127.0.0.1:8080/v1/")
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-key")
    monkeypatch.setenv("DEFAULT_MODEL", "fallback-model")
    settings = ApiSettings.from_environment()
    assert settings.model_base_url == "http://127.0.0.1:8080/v1"
    assert settings.model_api_key == "fallback-key"
    assert settings.default_model == "fallback-model"

    monkeypatch.setenv("CODEINSIGHT_MODEL_BASE_URL", "   ")
    monkeypatch.delenv("OPENAI_API_BASE")
    assert ApiSettings.from_environment().model_base_url is None


def test_environment_normalizes_optional_build_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEINSIGHT_BUILD_COMMIT", "  abc123  ")
    monkeypatch.setenv("CODEINSIGHT_BUILD_TIME", " 2026-07-25T12:00:00Z ")
    settings = ApiSettings.from_environment()
    assert settings.build_commit == "abc123"
    assert settings.build_time == "2026-07-25T12:00:00Z"

    monkeypatch.setenv("CODEINSIGHT_BUILD_COMMIT", " ")
    monkeypatch.setenv("CODEINSIGHT_BUILD_TIME", "")
    settings = ApiSettings.from_environment()
    assert settings.build_commit is None
    assert settings.build_time is None


def test_default_startup_uses_ignored_checkout_local_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CODEINSIGHT_DATA_DIR", raising=False)
    monkeypatch.delenv("CODEINSIGHT_DATABASE_PATH", raising=False)
    monkeypatch.setattr(api.config, "PROJECT_ROOT", tmp_path)

    with TestClient(create_app()) as client:
        assert client.get("/api/v1/health").status_code == 200

    assert (tmp_path / ".codeinsight" / "codeinsight.db").is_file()


def test_explicit_database_location_overrides_checkout_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "portable" / "ledger.db"
    monkeypatch.setenv("CODEINSIGHT_DATABASE_PATH", str(database_path))
    monkeypatch.setattr(api.config, "PROJECT_ROOT", tmp_path / "checkout")

    assert ApiSettings.from_environment().database_path == database_path


def test_language_capability_helpers_are_deterministic() -> None:
    assert get_language_for_extension(".PY") == Language.PYTHON
    assert get_language_for_extension(".unknown") is None
    assert get_extensions_for_languages(["python", "unknown"]) == [
        ".py",
        ".pyi",
        ".pyw",
    ]
    assert get_language_metadata("python").name == "Python"
    assert get_language_metadata("unknown") is None
    assert len(get_supported_languages()) == 30
    assert "pyproject.toml" in get_all_dependency_file_patterns()


def test_database_initializer_cli_uses_canonical_schema_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = tmp_path / "state" / "codeinsight.db"
    monkeypatch.setattr(sys, "argv", ["init_database", str(database_path)])

    assert init_database.main() == 0
    assert database_path.is_file()
    assert f"schema {SCHEMA_VERSION}" in capsys.readouterr().out


def test_production_frontend_assets_stay_within_release_budgets() -> None:
    web_root = Path(__file__).resolve().parents[1] / "web"
    javascript = list((web_root / "assets").glob("*.js"))
    stylesheets = list((web_root / "assets").glob("*.css"))

    assert len(javascript) == 1
    assert len(stylesheets) == 1
    assert len(gzip.compress(javascript[0].read_bytes())) < 100 * 1024
    assert len(gzip.compress(stylesheets[0].read_bytes())) < 20 * 1024
    assert not list((web_root / "assets").glob("*.map"))
    css = stylesheets[0].read_text(encoding="utf-8")
    index = (web_root / "index.html").read_text(encoding="utf-8")
    assert "@media (prefers-reduced-motion:reduce)" in css
    assert "@media (forced-colors:active)" in css
    assert "http://" not in index
    assert "https://" not in index
