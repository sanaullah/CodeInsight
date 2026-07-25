from __future__ import annotations

import sys
from pathlib import Path

import pytest

from api.config import ApiSettings, load_environment
from indexing.scanners.language_config import (
    Language,
    get_all_dependency_file_patterns,
    get_extensions_for_languages,
    get_language_for_extension,
    get_language_metadata,
    get_supported_languages,
)
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
    assert "schema 1" in capsys.readouterr().out
