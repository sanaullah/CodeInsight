import { describe, expect, it } from "vitest";
import {
  ARCHITECTURE_VIEW_VERSION,
  COMPONENT_KINDS,
  decodeArchitectureView,
  encodeArchitectureView,
  RELATION_KINDS,
} from "./viewState";

describe("architecture view state", () => {
  it("round-trips a snapshot-pinned focused trace deterministically", () => {
    const encoded = encodeArchitectureView({
      snapshotId: "snapshot/one",
      focus: "service:orders",
      selectedId: "service-orders",
      showBoundaries: false,
      componentKinds: new Set(["service", "datastore"]),
      relationKinds: new Set(["request", "data_access"]),
      traceSource: "service-orders",
      traceTarget: "store-orders",
      lens: "impact",
      compareSnapshotId: "snapshot-zero",
    });

    expect(encoded.toString()).toBe(
      "view=1&snapshot=snapshot%2Fone&component_kind=service&component_kind=datastore" +
        "&relation_kind=request&relation_kind=data_access&focus=service%3Aorders" +
        "&selected=service-orders&boundaries=false&trace_source=service-orders" +
        "&trace_target=store-orders&lens=impact&compare=snapshot-zero",
    );
    expect(decodeArchitectureView(encoded.toString())).toEqual({
      snapshotId: "snapshot/one",
      focus: "service:orders",
      selectedId: "service-orders",
      showBoundaries: false,
      componentKinds: new Set(["service", "datastore"]),
      relationKinds: new Set(["request", "data_access"]),
      traceSource: "service-orders",
      traceTarget: "store-orders",
      lens: "impact",
      compareSnapshotId: "snapshot-zero",
    });
  });

  it("distinguishes no visible layers from the default all-visible view", () => {
    const encoded = encodeArchitectureView({
      snapshotId: "snapshot-1",
      focus: "",
      selectedId: "",
      showBoundaries: true,
      componentKinds: new Set(),
      relationKinds: new Set(),
      traceSource: "",
      traceTarget: "",
      lens: "topology",
      compareSnapshotId: "",
    });

    expect(encoded.get("component_kind")).toBe("none");
    expect(encoded.get("relation_kind")).toBe("none");
    const decoded = decodeArchitectureView(encoded.toString());
    expect(decoded.componentKinds.size).toBe(0);
    expect(decoded.relationKinds.size).toBe(0);
    expect(decoded.lens).toBe("topology");
    expect(decoded.compareSnapshotId).toBe("");
  });

  it("keeps legacy and unknown-filter URLs on safe all-visible defaults", () => {
    const decoded = decodeArchitectureView("?component_kind=bogus&relation_kind=bogus");

    expect(decoded.componentKinds).toEqual(new Set(COMPONENT_KINDS));
    expect(decoded.relationKinds).toEqual(new Set(RELATION_KINDS));
    expect(decoded.lens).toBe("topology");
    expect(ARCHITECTURE_VIEW_VERSION).toBe("1");
  });
});
