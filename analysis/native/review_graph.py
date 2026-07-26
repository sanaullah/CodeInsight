"""Typed Pydantic Graph contract and live telemetry for the review coordinator.

The graph keeps orchestration topology explicit while CodeInsight retains task
leasing, cancellation, recovery, and evidence persistence.  The coordinator
emits a graph-node transition at each real stage boundary through the existing
durable event sink, which also correlates it to the Langfuse run trace.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic_graph.graph_builder import Graph, GraphBuilder, StepContext

from domain.contracts import RunStage


class ReviewGraphNode(StrEnum):
    """Stable identifiers for coordinator stages represented by the graph."""

    PLAN_WAVE = "plan_wave"
    DISPATCH_TASKS = "dispatch_tasks"
    VERIFY_EVIDENCE = "verify_evidence"
    DEDUPLICATE_AND_CORRELATE = "deduplicate_and_correlate"
    ASSESS_COVERAGE = "assess_coverage"
    PLAN_FOLLOW_UP_WAVE = "plan_follow_up_wave"
    SYNTHESIZE = "synthesize"
    COMPLETE = "complete"


NODE_STAGE: dict[ReviewGraphNode, RunStage] = {
    ReviewGraphNode.PLAN_WAVE: RunStage.PLAN_WAVE,
    ReviewGraphNode.DISPATCH_TASKS: RunStage.DISPATCH_TASKS,
    ReviewGraphNode.VERIFY_EVIDENCE: RunStage.VERIFY_EVIDENCE,
    ReviewGraphNode.DEDUPLICATE_AND_CORRELATE: RunStage.DEDUPLICATE_AND_CORRELATE,
    ReviewGraphNode.ASSESS_COVERAGE: RunStage.ASSESS_COVERAGE,
    ReviewGraphNode.PLAN_FOLLOW_UP_WAVE: RunStage.PLAN_FOLLOW_UP_WAVE,
    ReviewGraphNode.SYNTHESIZE: RunStage.SYNTHESIZE,
    ReviewGraphNode.COMPLETE: RunStage.COMPLETE,
}

STAGE_NODE: dict[RunStage, ReviewGraphNode] = {
    stage: node for node, stage in NODE_STAGE.items()
}


class ReviewGraphTransition(BaseModel):
    """One durable-event-compatible graph-node transition."""

    model_config = ConfigDict(frozen=True)

    node: ReviewGraphNode
    stage: RunStage
    transition_index: int = Field(ge=1)


class ReviewGraphState(BaseModel):
    """Small typed state owned by the orchestration graph, never raw source."""

    run_id: str = Field(min_length=1)
    follow_up_requested: bool = False
    transitions: list[ReviewGraphTransition] = Field(default_factory=list)


class ReviewGraphInput(BaseModel):
    """Input needed to execute a deterministic graph-contract rehearsal."""

    run_id: str = Field(min_length=1)
    follow_up_requested: bool = False


class ReviewGraphResult(BaseModel):
    """Observable result of a graph-contract rehearsal."""

    run_id: str
    transitions: tuple[ReviewGraphTransition, ...]


EventSink = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class ReviewGraphDependencies:
    """Inject the existing durable event sink; the graph owns no storage."""

    event_sink: EventSink


class ReviewGraphTelemetry:
    """Emit live Pydantic-Graph node transitions without owning durable work.

    This bridge deliberately delegates persistence to the coordinator's
    existing event sink.  Therefore a node transition is written to SQLite and
    exported through the configured ``TraceExporter`` in the same way as every
    other run event, without a second telemetry pipeline or schema.
    """

    def __init__(self, *, run_id: str, event_sink: EventSink) -> None:
        self._run_id = run_id
        self._event_sink = event_sink
        self._transition_index = 0

    def record_stage(self, stage: RunStage) -> ReviewGraphTransition | None:
        node = STAGE_NODE.get(stage)
        if node is None:
            return None
        self._transition_index += 1
        transition = ReviewGraphTransition(
            node=node,
            stage=stage,
            transition_index=self._transition_index,
        )
        self._event_sink(
            "review_graph_node_entered",
            {
                "run_id": self._run_id,
                "node": node.value,
                "stage": stage.value,
                "transition_index": transition.transition_index,
                "graph_name": "durable_review_orchestration",
                "graph_runtime": "pydantic-graph",
            },
        )
        return transition


def _record_transition(
    ctx: StepContext[ReviewGraphState, ReviewGraphDependencies, Any],
    node: ReviewGraphNode,
) -> None:
    transition = ReviewGraphTransition(
        node=node,
        stage=NODE_STAGE[node],
        transition_index=len(ctx.state.transitions) + 1,
    )
    ctx.state.transitions.append(transition)
    ctx.deps.event_sink(
        "review_graph_node_entered",
        {
            "run_id": ctx.state.run_id,
            "node": transition.node.value,
            "stage": transition.stage.value,
            "transition_index": transition.transition_index,
        },
    )


def build_review_graph() -> Graph[
    ReviewGraphState,
    ReviewGraphDependencies,
    ReviewGraphInput,
    ReviewGraphResult,
]:
    """Build the current coordinator topology using Pydantic Graph's builder API.

    The graph deliberately stops at stage contracts.  A later milestone can
    attach the real architecture, planning, scheduler, verification, and
    synthesis handlers to these typed nodes without changing their event shape.
    """

    builder: GraphBuilder[
        ReviewGraphState,
        ReviewGraphDependencies,
        ReviewGraphInput,
        ReviewGraphResult,
    ] = GraphBuilder(name="durable_review_orchestration", auto_instrument=False)

    @builder.step(node_id="plan_wave", label="Plan wave")
    async def plan_wave(
        ctx: StepContext[ReviewGraphState, ReviewGraphDependencies, ReviewGraphInput],
    ) -> None:
        if ctx.inputs.run_id != ctx.state.run_id:
            raise ValueError("review graph input does not match state run_id")
        _record_transition(ctx, ReviewGraphNode.PLAN_WAVE)

    @builder.step(node_id="dispatch_tasks", label="Dispatch tasks")
    async def dispatch_tasks(
        ctx: StepContext[ReviewGraphState, ReviewGraphDependencies, None],
    ) -> None:
        _record_transition(ctx, ReviewGraphNode.DISPATCH_TASKS)

    @builder.step(node_id="verify_evidence", label="Verify evidence")
    async def verify_evidence(
        ctx: StepContext[ReviewGraphState, ReviewGraphDependencies, None],
    ) -> None:
        _record_transition(ctx, ReviewGraphNode.VERIFY_EVIDENCE)

    @builder.step(node_id="deduplicate_and_correlate", label="Deduplicate and correlate")
    async def deduplicate_and_correlate(
        ctx: StepContext[ReviewGraphState, ReviewGraphDependencies, None],
    ) -> None:
        _record_transition(ctx, ReviewGraphNode.DEDUPLICATE_AND_CORRELATE)

    @builder.step(node_id="assess_coverage", label="Assess coverage")
    async def assess_coverage(
        ctx: StepContext[ReviewGraphState, ReviewGraphDependencies, None],
    ) -> bool:
        _record_transition(ctx, ReviewGraphNode.ASSESS_COVERAGE)
        return ctx.state.follow_up_requested

    @builder.step(node_id="plan_follow_up_wave", label="Plan follow-up wave")
    async def plan_follow_up_wave(
        ctx: StepContext[ReviewGraphState, ReviewGraphDependencies, bool],
    ) -> None:
        _record_transition(ctx, ReviewGraphNode.PLAN_FOLLOW_UP_WAVE)

    @builder.step(node_id="synthesize", label="Synthesize")
    async def synthesize(
        ctx: StepContext[ReviewGraphState, ReviewGraphDependencies, object],
    ) -> None:
        _record_transition(ctx, ReviewGraphNode.SYNTHESIZE)

    @builder.step(node_id="complete", label="Complete")
    async def complete(
        ctx: StepContext[ReviewGraphState, ReviewGraphDependencies, None],
    ) -> ReviewGraphResult:
        _record_transition(ctx, ReviewGraphNode.COMPLETE)
        return ReviewGraphResult(
            run_id=ctx.state.run_id,
            transitions=tuple(ctx.state.transitions),
        )

    follow_up = builder.decision(node_id="follow_up_required", note="Follow-up evidence gap?")
    follow_up = follow_up.branch(
        builder.match(bool, matches=lambda requested: requested)
        .label("yes")
        .to(plan_follow_up_wave)
    )
    follow_up = follow_up.branch(
        builder.match(bool, matches=lambda requested: not requested).label("no").to(synthesize)
    )

    builder.add_edge(builder.start_node, plan_wave)
    builder.add_edge(plan_wave, dispatch_tasks)
    builder.add_edge(dispatch_tasks, verify_evidence)
    builder.add_edge(verify_evidence, deduplicate_and_correlate)
    builder.add_edge(deduplicate_and_correlate, assess_coverage)
    builder.add(builder.edge_from(assess_coverage).to(follow_up))
    builder.add_edge(plan_follow_up_wave, synthesize)
    builder.add_edge(synthesize, complete)
    builder.add_edge(complete, builder.end_node)
    return builder.build()


async def run_review_graph_contract(
    *,
    run_id: str,
    follow_up_requested: bool,
    event_sink: EventSink,
) -> ReviewGraphResult:
    """Run the non-production graph contract with the coordinator event sink."""

    state = ReviewGraphState(run_id=run_id, follow_up_requested=follow_up_requested)
    return await build_review_graph().run(
        state=state,
        deps=ReviewGraphDependencies(event_sink=event_sink),
        inputs=ReviewGraphInput(run_id=run_id, follow_up_requested=follow_up_requested),
    )
