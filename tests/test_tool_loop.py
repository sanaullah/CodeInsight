from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest


class _Logger:
    def debug(self, *_args, **_kwargs) -> None:
        pass

    info = debug
    warning = debug
    error = debug


def _load_legacy_loop(**overrides):
    """Execute the actual loop source without importing legacy frameworks."""

    source_path = (
        Path(__file__).resolve().parents[1]
        / "workflow"
        / "swarm_analysis"
        / "agent_execution"
        / "nodes.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    names = {"_normalize_tool_arguments", "_execute_with_tool_calling"}
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    for function in functions:
        function.decorator_list = []

    async def default_llm(_state, config=None):
        return {"last_response": "final", "tool_calls": []}

    async def default_tool(**_kwargs):
        return {"content": "ok"}

    namespace = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "asyncio": asyncio,
        "logger": _Logger(),
        "StateKeys": SimpleNamespace(
            FILE_CACHE="file_cache", TOOL_CALLS_HISTORY="tool_calls_history"
        ),
        "get_tool_definitions": lambda: [],
        "llm_node": default_llm,
        "execute_tool": default_tool,
        "parse_tool_call": lambda call: (
            call.get("function", {}).get("name"),
            {},
        ),
        "create_observation": lambda **_kwargs: None,
        **overrides,
    }
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_execute_with_tool_calling"], namespace["StateKeys"]


def _llm_state() -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": "review this code"}],
        "model": "fake-model",
    }


@pytest.mark.asyncio
async def test_tool_loop_returns_first_final_response() -> None:
    calls = 0

    async def fake_llm(_state, config=None):
        nonlocal calls
        calls += 1
        return {"last_response": "final analysis", "tool_calls": []}

    loop, _state_keys = _load_legacy_loop(llm_node=fake_llm)
    result = await loop(_llm_state(), ".", {}, max_iterations=5)

    assert result["last_response"] == "final analysis"
    assert calls == 1


@pytest.mark.asyncio
async def test_tool_loop_executes_tools_then_returns_final_response() -> None:
    responses = iter(
        [
            {
                "last_response": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"README.md"}',
                        },
                    }
                ],
            },
            {"last_response": "final analysis", "tool_calls": []},
        ]
    )

    async def fake_llm(_state, config=None):
        return next(responses)

    loop, state_keys = _load_legacy_loop(llm_node=fake_llm)
    state: dict[str, Any] = {}
    result = await loop(_llm_state(), ".", state, max_iterations=5)

    assert result["last_response"] == "final analysis"
    assert len(state[state_keys.TOOL_CALLS_HISTORY]) == 1


@pytest.mark.asyncio
async def test_tool_loop_reports_timeout_before_any_result() -> None:
    async def slow_llm(_state, config=None):
        await asyncio.sleep(1)

    loop, _state_keys = _load_legacy_loop(llm_node=slow_llm)
    result = await loop(
        _llm_state(), ".", {}, llm_call_timeout=0.001
    )

    assert "timed out" in result["error"]


@pytest.mark.asyncio
async def test_tool_loop_propagates_cancellation() -> None:
    async def cancelled_llm(_state, config=None):
        raise asyncio.CancelledError

    loop, _state_keys = _load_legacy_loop(llm_node=cancelled_llm)
    with pytest.raises(asyncio.CancelledError):
        await loop(_llm_state(), ".", {})
