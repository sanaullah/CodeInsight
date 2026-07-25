import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { expect, it, vi } from "vitest";
import type { SemanticComponent } from "../../api/contracts";
import { ArchitectureFocusSearch } from "./ArchitectureFocusSearch";

const components: SemanticComponent[] = [
  {
    component_id: "orders",
    snapshot_id: "snapshot",
    stable_key: "service:orders",
    name: "Orders API",
    component_kind: "service",
    support_tier: "exact",
    completeness: "complete",
    confidence: 1,
    metadata: {},
  },
  {
    component_id: "store",
    snapshot_id: "snapshot",
    stable_key: "datastore:orders",
    name: "Orders database",
    component_kind: "datastore",
    support_tier: "exact",
    completeness: "complete",
    confidence: 1,
    metadata: {},
  },
];

it("searches partial local evidence before applying an exact server focus", async () => {
  const onChange = vi.fn();
  const onApply = vi.fn();
  const onSelect = vi.fn();
  const user = userEvent.setup();
  function Harness() {
    const [value, setValue] = useState("");
    return (
      <ArchitectureFocusSearch
        appliedValue=""
        components={components}
        onApply={onApply}
        onChange={(next) => {
          onChange(next);
          setValue(next);
        }}
        onSelect={onSelect}
        value={value}
      />
    );
  }
  render(<Harness />);

  await user.type(screen.getByLabelText("Search and focus components"), "orders");
  expect(onChange).toHaveBeenCalledTimes(6);
  expect(screen.getByText("2 local matches")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: /Orders API/i }));
  expect(onSelect).toHaveBeenCalledWith(components[0]);
  await user.click(screen.getByRole("button", { name: "Apply exact focus" }));
  expect(onApply).toHaveBeenCalledWith("orders");
});
