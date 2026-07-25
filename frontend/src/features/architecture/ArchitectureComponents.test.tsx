import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type {
  FindingDetail,
  SemanticArchitectureGraph,
  SemanticComponent,
  SemanticComponentDetail,
  SemanticTrace,
} from "../../api/contracts";
import { ArchitectureTextAlternative } from "./ArchitectureTextAlternative";
import { ArchitectureWorkspace } from "./ArchitectureWorkspace";
import { ComponentInspector } from "./ComponentInspector";
import { FindingEvidenceDrawer } from "./FindingEvidenceDrawer";
import { LayerRail } from "./LayerRail";
import { TopologyViewport } from "./TopologyViewport";
import { TraceRail } from "./TraceRail";

const service: SemanticComponent = {
  component_id: "component-api",
  snapshot_id: "snapshot-1",
  stable_key: "service:api",
  name: "Review API",
  component_kind: "service",
  support_tier: "exact",
  completeness: "complete",
  confidence: 0.98,
  metadata: { framework: "FastAPI" },
};

const datastore: SemanticComponent = {
  component_id: "component-db",
  snapshot_id: "snapshot-1",
  stable_key: "datastore:ledger",
  name: "Durable ledger",
  component_kind: "datastore",
  support_tier: "exact",
  completeness: "complete",
  confidence: 0.96,
  metadata: {},
};

const library: SemanticComponent = {
  component_id: "component-lib",
  snapshot_id: "snapshot-1",
  stable_key: "library:helpers",
  name: "Static helpers",
  component_kind: "library",
  support_tier: "inferred",
  completeness: "partial",
  confidence: 0.74,
  metadata: {},
};

const graph: SemanticArchitectureGraph = {
  snapshot_id: "snapshot-1",
  display_name: "CodeInsightsV2",
  git: {
    ref: "refs/heads/refactor",
    head_commit: "abc123",
    dirty: false,
    parent_snapshot_id: null,
  },
  created_at: "2026-07-25T12:00:00Z",
  counts: {
    services: 1,
    datastores: 1,
    external_systems: 0,
    queues: 0,
    libraries: 1,
    unknown: 0,
    boundaries: 0,
  },
  totals: { files: 3, lines: 120, languages: { Python: 3 } },
  completeness: { status: "partial", extractor_version: "semantic-v1" },
  components: [service, datastore, library],
  boundaries: [
    {
      boundary_id: "boundary-core",
      snapshot_id: "snapshot-1",
      stable_key: "boundary:core",
      name: "Core services",
      boundary_kind: "logical",
      support_tier: "inferred",
      completeness: "partial",
      confidence: 0.8,
      component_ids: [service.component_id],
      metadata: {},
    },
  ],
  relations: [
    {
      relation_id: "relation-data",
      source_component_id: service.component_id,
      target_component_id: datastore.component_id,
      stable_key: "data:api:ledger",
      relation_kind: "data_access",
      transport: "SQLite",
      is_async: false,
      support_tier: "exact",
      completeness: "complete",
      confidence: 0.95,
      metadata: {},
    },
    {
      relation_id: "relation-call",
      source_component_id: service.component_id,
      target_component_id: library.component_id,
      stable_key: "call:api:helpers",
      relation_kind: "call",
      transport: null,
      is_async: false,
      support_tier: "inferred",
      completeness: "partial",
      confidence: 0.7,
      metadata: {},
    },
  ],
  findings: [
    {
      finding_id: "finding-1",
      title: "Unsafe query construction",
      severity: "high",
      confidence: 0.93,
      component_id: service.component_id,
    },
  ],
  truncated: true,
  findings_truncated: false,
  limits: { depth: 1, node_limit: 100, finding_limit: 500 },
};

const detail: SemanticComponentDetail = {
  ...service,
  memberships: [
    {
      membership_id: "membership-1",
      snapshot_id: "snapshot-1",
      component_id: service.component_id,
      file_id: "file-1",
      symbol_id: null,
      membership_kind: "contains",
      confidence: 1,
      provenance: {},
      relative_path: null,
      language: null,
      line_count: null,
    },
  ],
  resources: [
    {
      resource_id: "resource-1",
      snapshot_id: "snapshot-1",
      component_id: service.component_id,
      stable_key: "resource:reviews",
      resource_kind: "table",
      name: "reviews",
      locator: "codeinsight.sqlite",
      support_tier: "exact",
      completeness: "complete",
      confidence: 0.95,
      metadata: {},
    },
  ],
  endpoints: [
    {
      endpoint_id: "endpoint-1",
      snapshot_id: "snapshot-1",
      component_id: service.component_id,
      stable_key: "endpoint:reviews",
      protocol: "http",
      method: "GET",
      route: "/api/v1/reviews",
      direction: "inbound",
      support_tier: "exact",
      completeness: "complete",
      confidence: 0.99,
      metadata: { framework: "FastAPI" },
    },
  ],
  provenance: [],
  findings: graph.findings,
  annotation: null,
  evidence_truncated: false,
  limits: { detail_row_limit: 250 },
};

const trace: SemanticTrace = {
  snapshot_id: "snapshot-1",
  status: "complete",
  components: [service, library],
  relations: [graph.relations[1]],
  endpoints: detail.endpoints,
  resources: detail.resources,
  provenance: [
    {
      provenance_id: "provenance-1",
      entity_kind: "relation",
      entity_id: "relation-call",
      file_id: "file-1",
      relative_path: "api/app.py",
      start_line: 10,
      end_line: 12,
      derivation: "static_call",
      extractor_version: "semantic-v1",
      confidence: 0.7,
      metadata: {},
    },
  ],
  evidence_truncated: false,
  max_hops: 4,
};

const finding = {
  finding_id: "finding-1",
  fingerprint: "fingerprint-1",
  title: "Unsafe query construction",
  claim: "The query is assembled from an untrusted value.",
  severity: "high",
  confidence: 0.93,
  candidate_ids: ["candidate-1"],
  supporting_evidence_ids: ["evidence-1"],
  conflicting_candidate_ids: [],
  recommendation: "Use a parameterized query.",
  run_id: "run-1",
  review_state: "validated",
  review_note: null,
  review_version: 1,
  reviewed_at: null,
  candidates: [
    {
      relationship: "supporting",
      candidate: {
        candidate_id: "candidate-1",
        title: "Unsafe query construction",
        claim: "The query is assembled from an untrusted value.",
        category: "security",
        affected_path: "api/app.py",
        impact: "Untrusted input can alter the database query.",
        proposed_severity: "high",
        proposed_confidence: 0.93,
        recommendation: "Use a parameterized query.",
      },
      role: null,
      verdict: null,
    },
  ],
  evidence: [
    {
      evidence_id: "evidence-1",
      relative_path: "api/app.py",
      content_hash: "hash-1",
      start_line: 10,
      end_line: 12,
      evidence_kind: "source_span",
      provenance: {},
      excerpt: "query = f'SELECT {value}'",
      integrity: "valid",
      redacted: false,
    },
  ],
  review_history: [],
} as FindingDetail;

describe("Architecture Explorer feature components", () => {
  it("toggles semantic layers with their server-derived counts", async () => {
    const user = userEvent.setup();
    const onToggle = vi.fn();
    render(
      <LayerRail
        layers={[
          { id: "service", label: "Services", count: 2, active: true },
          { id: "datastore", label: "Datastores", count: 1, active: false },
        ]}
        onToggle={onToggle}
      />,
    );

    expect(screen.getByText("1 visible")).toBeInTheDocument();
    await user.click(screen.getByRole("checkbox", { name: /datastores/i }));
    expect(onToggle).toHaveBeenCalledWith("datastore", true);
  });

  it("renders a bounded button-backed topology and owns local zoom and pan", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    const onFocus = vi.fn();
    render(
      <TopologyViewport
        graph={graph}
        onFocusComponent={onFocus}
        onSelectComponent={onSelect}
        selectedComponentId={service.component_id}
      />,
    );

    expect(screen.getByText("Bounded result")).toBeInTheDocument();
    expect(screen.getByText("Core services")).toBeInTheDocument();
    const node = screen.getByRole("button", { name: /Review API/i });
    expect(node).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("1 correlated finding")).toBeInTheDocument();
    await user.click(node);
    expect(onSelect).toHaveBeenCalledWith(service);
    fireEvent.doubleClick(node);
    expect(onFocus).toHaveBeenCalledWith(service);

    await user.click(screen.getByRole("button", { name: "Zoom in" }));
    expect(screen.getByText("110%")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Zoom out" }));
    expect(screen.getByText("100%")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Pan left" }));
    await user.click(screen.getByRole("button", { name: "Pan up" }));
    await user.click(screen.getByRole("button", { name: "Pan down" }));
    await user.click(screen.getByRole("button", { name: "Pan right" }));
    expect(
      screen.getByTestId("topology-viewport").querySelector(".architecture-canvas"),
    ).toHaveStyle({ transform: "translate(0px, 0px) scale(1)" });
    await user.click(screen.getByRole("button", { name: "Reset" }));

    const decorativeDiagram = screen.getByTestId("topology-viewport").querySelector("svg");
    expect(decorativeDiagram).toHaveAttribute("aria-hidden", "true");
  });

  it("keeps an explicit empty topology when active server bounds return no components", () => {
    render(
      <TopologyViewport
        graph={{ ...graph, components: [], relations: [], findings: [], boundaries: [] }}
        onSelectComponent={vi.fn()}
        selectedComponentId={null}
      />,
    );

    expect(screen.getByText("No components match the active layers.")).toBeInTheDocument();
    expect(
      screen.getByText("Enable a layer or adjust the server-side filters."),
    ).toBeInTheDocument();
  });

  it("uses the browser fullscreen API only when it is supported", async () => {
    const requestFullscreen = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(HTMLElement.prototype, "requestFullscreen", {
      configurable: true,
      value: requestFullscreen,
    });
    try {
      const user = userEvent.setup();
      render(
        <TopologyViewport graph={graph} onSelectComponent={vi.fn()} selectedComponentId={null} />,
      );

      const button = await screen.findByRole("button", { name: "Enter fullscreen" });
      expect(button).toBeEnabled();
      await user.click(button);
      expect(requestFullscreen).toHaveBeenCalledOnce();
    } finally {
      Reflect.deleteProperty(HTMLElement.prototype, "requestFullscreen");
    }
  });

  it("keeps the text alternative visible and distinguishes static calls from runtime flow", () => {
    render(
      <ArchitectureTextAlternative
        graph={graph}
        onSelectComponent={vi.fn()}
        selectedComponentId={null}
      />,
    );

    const textView = screen
      .getByRole("heading", { name: "Text architecture view" })
      .closest("section");
    expect(textView).toBeVisible();
    expect(screen.getByText("Boundaries (1)")).toBeInTheDocument();
    expect(
      within(textView as HTMLElement).getByText(/data access via SQLite/i),
    ).toBeInTheDocument();
    expect(
      within(textView as HTMLElement).getByText(/static call reference.*70% confidence/i),
    ).toBeInTheDocument();
    expect(
      within(textView as HTMLElement).getByText(
        "Static repository relationship; not evidence of runtime data flow.",
      ),
    ).toBeInTheDocument();
  });

  it("shows defensive membership and endpoint metadata in the inspector", async () => {
    const user = userEvent.setup();
    const onFinding = vi.fn();
    const onSaveAnnotation = vi.fn().mockResolvedValue(undefined);
    render(
      <ComponentInspector
        component={service}
        detail={detail}
        onOpenFinding={onFinding}
        onSaveAnnotation={onSaveAnnotation}
      />,
    );

    expect(screen.getByText("file-1")).toBeInTheDocument();
    expect(screen.getByText("unknown / unknown lines")).toBeInTheDocument();
    expect(screen.getByText("FastAPI / complete")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Unsafe query construction/i }));
    expect(onFinding).toHaveBeenCalledWith("finding-1");
    await user.type(screen.getByLabelText("Component annotation"), "Owner verified");
    await user.click(screen.getByRole("button", { name: "Save annotation" }));
    expect(onSaveAnnotation).toHaveBeenCalledWith("Owner verified", 0);
  });

  it("makes bounded component evidence explicit in the inspector", () => {
    render(
      <ComponentInspector component={service} detail={{ ...detail, evidence_truncated: true }} />,
    );
    expect(screen.getByText(/reached the server limit of 250 rows/i)).toBeInTheDocument();
  });

  it("provides an ordered typed trace without mislabeling a static call", () => {
    render(<TraceRail trace={trace} />);

    const steps = screen.getAllByRole("listitem");
    expect(steps[0]).toHaveTextContent("Review API");
    expect(steps[1]).toHaveTextContent("Static helpers");
    expect(screen.getByText("static call reference")).toBeInTheDocument();
    expect(screen.getByText("Static relationship, not runtime data flow")).toBeInTheDocument();
    expect(screen.getAllByText(/static call.*70%/i)).toHaveLength(2);
    expect(screen.getByText("Entry point")).toBeInTheDocument();
    expect(screen.getByText("Resource")).toBeInTheDocument();
    expect(screen.getByText("Source evidence")).toBeInTheDocument();
  });

  it("surfaces unsupported trace state explicitly", () => {
    render(
      <TraceRail
        trace={{
          ...trace,
          status: "unsupported",
          components: [],
          relations: [],
          endpoints: [],
          resources: [],
          provenance: [],
          evidence_truncated: false,
        }}
      />,
    );
    expect(screen.getByText("Unsupported evidence")).toBeInTheDocument();
    expect(screen.queryByRole("list")).not.toBeInTheDocument();
  });

  it("makes bounded trace evidence truncation explicit", () => {
    render(<TraceRail trace={{ ...trace, evidence_truncated: true }} />);
    expect(screen.getByText(/Evidence details reached the server bound/i)).toBeInTheDocument();
  });

  it("renders verified evidence, truthful export, copy, and keyboard close actions", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const onCopy = vi.fn();
    render(
      <FindingEvidenceDrawer
        exportHref="/api/v1/findings/finding-1/export?format=json"
        finding={finding}
        openEvidenceHref="/findings?finding=finding-1"
        onClose={onClose}
        onCopyEvidence={onCopy}
      />,
    );

    expect(screen.getByRole("link", { name: "Export finding" })).toHaveAttribute(
      "href",
      "/api/v1/findings/finding-1/export?format=json",
    );
    expect(screen.getByRole("link", { name: "Open evidence workspace" })).toHaveAttribute(
      "href",
      "/findings?finding=finding-1",
    );
    expect(screen.getByText("Evidence-derived impact")).toBeInTheDocument();
    expect(screen.getByText("Untrusted input can alter the database query.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Copy evidence" }));
    expect(onCopy).toHaveBeenCalledWith("query = f'SELECT {value}'");
    expect(screen.getByText("Evidence excerpt copied.")).toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalled();
  });

  it("reports unavailable clipboard access without losing evidence", async () => {
    const user = userEvent.setup();
    render(
      <FindingEvidenceDrawer
        finding={finding}
        onClose={vi.fn()}
        onCopyEvidence={() => Promise.reject(new Error("Clipboard denied"))}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Copy evidence" }));
    expect(screen.getByText("Clipboard access is unavailable.")).toBeInTheDocument();
    expect(screen.getByText("query = f'SELECT {value}'")).toBeInTheDocument();
  });

  it("composes all workspace regions without imposing page routing state", () => {
    render(
      <ArchitectureWorkspace
        inspector={<div>Inspector slot</div>}
        layers={<div>Layer slot</div>}
        textAlternative={<div>Text slot</div>}
        topology={<div>Topology slot</div>}
        trace={<div>Trace slot</div>}
      />,
    );
    for (const label of [
      "Inspector slot",
      "Layer slot",
      "Text slot",
      "Topology slot",
      "Trace slot",
    ]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
  });
});
