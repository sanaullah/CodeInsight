from __future__ import annotations

import pytest

from analysis.native.review_graph import (
    ReviewGraphDependencies,
    ReviewGraphInput,
    ReviewGraphNode,
    ReviewGraphState,
    ReviewGraphTelemetry,
    build_review_graph,
    run_review_graph_contract,
)
from application.tracing import TraceEvent
from domain.contracts import RunStage


@pytest.mark.asyncio
async def test_graph_contract_emits_durable_event_shape_for_linear_review() -> None:
    events: list[tuple[str, dict[str, object]]] = []

    result = await run_review_graph_contract(
        run_id="run-graph-1",
        follow_up_requested=False,
        event_sink=lambda event_type, data: events.append((event_type, data)),
    )

    assert [transition.node for transition in result.transitions] == [
        ReviewGraphNode.PLAN_WAVE,
        ReviewGraphNode.DISPATCH_TASKS,
        ReviewGraphNode.VERIFY_EVIDENCE,
        ReviewGraphNode.DEDUPLICATE_AND_CORRELATE,
        ReviewGraphNode.ASSESS_COVERAGE,
        ReviewGraphNode.SYNTHESIZE,
        ReviewGraphNode.COMPLETE,
    ]
    assert [event_type for event_type, _ in events] == ["review_graph_node_entered"] * 7
    assert events[0][1] == {
        "run_id": "run-graph-1",
        "node": "plan_wave",
        "stage": "plan_wave",
        "transition_index": 1,
    }
    assert events[-1][1]["stage"] == "complete"


@pytest.mark.asyncio
async def test_graph_contract_routes_follow_up_branch_without_leasing_tasks() -> None:
    events: list[tuple[str, dict[str, object]]] = []

    result = await run_review_graph_contract(
        run_id="run-graph-2",
        follow_up_requested=True,
        event_sink=lambda event_type, data: events.append((event_type, data)),
    )

    assert ReviewGraphNode.PLAN_FOLLOW_UP_WAVE in [
        transition.node for transition in result.transitions
    ]
    assert [event[1]["transition_index"] for event in events] == list(
        range(1, len(events) + 1)
    )


def test_live_graph_transition_is_trace_exporter_compatible_and_run_correlated() -> None:
    trace_events: list[TraceEvent] = []

    def event_sink(event_type: str, data: dict[str, object]) -> None:
        trace_events.append(
            TraceEvent(
                name=event_type,
                run_id=str(data["run_id"]),
                stage=str(data["stage"]),
                attributes=dict(data),
            )
        )

    telemetry = ReviewGraphTelemetry(run_id="run-observed", event_sink=event_sink)
    transition = telemetry.record_stage(RunStage.PLAN_WAVE)

    assert transition is not None
    assert trace_events[0].name == "review_graph_node_entered"
    assert trace_events[0].run_id == "run-observed"
    assert trace_events[0].stage == "plan_wave"
    assert trace_events[0].attributes["graph_runtime"] == "pydantic-graph"


def test_graph_contract_renders_existing_coordinator_stage_topology() -> None:
    rendered = build_review_graph().render()

    assert "plan_wave" in rendered
    assert "dispatch_tasks" in rendered
    assert "follow_up_required" in rendered
    assert "synthesize" in rendered


@pytest.mark.asyncio
async def test_graph_contract_rejects_input_for_another_run() -> None:
    with pytest.raises(ValueError, match="does not match state run_id"):
        await build_review_graph().run(
            state=ReviewGraphState(run_id="run-graph-3"),
            deps=ReviewGraphDependencies(event_sink=lambda _event_type, _data: None),
            inputs=ReviewGraphInput(run_id="different-run"),
        )
