// @ts-expect-error Node's filesystem module is supplied by the Vitest runtime.
import { readFileSync } from "node:fs";
import { render, screen, within } from "@testing-library/react";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { AppShell } from "./components/AppShell";
import { StatCard } from "./components/primitives";
import type { OperationalState } from "./hooks/useOperationalState";

const appStyles = readFileSync("src/styles.css", "utf8");
const architectureMetrics = [
  { icon: "architecture" as const, label: "Services" },
  { icon: "database" as const, label: "Data stores" },
  { icon: "files" as const, label: "Files" },
  { icon: "trace" as const, label: "Relations" },
];

const historyMetrics = [
  { icon: "history" as const, label: "Reviews" },
  { icon: "findings" as const, label: "Findings" },
  { icon: "coins" as const, label: "Cost" },
  { icon: "clock" as const, label: "Average time" },
];

let styleElement: HTMLStyleElement;

function productionRule(selector: string) {
  const start = appStyles.indexOf(`${selector} {`);
  if (start < 0) throw new Error(`Missing production CSS rule: ${selector}`);
  const end = appStyles.indexOf("}", start);
  if (end < 0) throw new Error(`Unterminated production CSS rule: ${selector}`);
  return appStyles.slice(start, end + 1);
}

beforeAll(() => {
  styleElement = document.createElement("style");
  styleElement.dataset.testStyles = "application";
  // jsdom rejects some modern declarations used elsewhere in the complete
  // stylesheet. Inject the exact production rules under test so computed
  // styles remain deterministic without copying their values into fixtures.
  styleElement.textContent = [
    productionRule(":root"),
    productionRule(".stat-card"),
    productionRule(".stat-icon"),
    productionRule(".stat-icon > svg"),
    productionRule(".app-footer"),
    productionRule(".page.dense-workspace"),
  ].join("\n");
  document.head.append(styleElement);
});

afterAll(() => {
  styleElement.remove();
});

function renderMetricCards(metrics: typeof architectureMetrics | typeof historyMetrics) {
  render(
    <div className="stat-grid">
      {metrics.map(({ icon, label }) => (
        <StatCard detail={`${label} detail`} icon={icon} key={label} label={label} value="1" />
      ))}
    </div>,
  );
}

function expectCenteredMetricIcon(label: string) {
  const labelElement = screen.getByText(label);
  const card = labelElement.closest(".stat-card");
  expect(card).not.toBeNull();

  const iconBox = card?.querySelector<HTMLElement>(".stat-icon");
  const icon = iconBox?.querySelector<SVGSVGElement>("svg");
  expect(iconBox).not.toBeNull();
  expect(icon).not.toBeNull();

  const cardStyle = window.getComputedStyle(card as HTMLElement);
  const boxStyle = window.getComputedStyle(iconBox as HTMLElement);
  const iconStyle = window.getComputedStyle(icon as SVGSVGElement);
  expect(cardStyle.display).toBe("flex");
  expect(cardStyle.alignItems).toBe("center");
  expect(boxStyle.display).toBe("grid");
  expect(boxStyle.width).toBe("42px");
  expect(boxStyle.height).toBe("42px");
  expect(boxStyle.placeItems).toBe("center");
  expect(boxStyle.flex).toBe("0 0 auto");
  expect(iconStyle.display).toBe("block");
  expect(iconStyle.width).toBe("20px");
  expect(iconStyle.height).toBe("20px");
  expect(icon).toHaveAttribute("width", "20");
  expect(icon).toHaveAttribute("height", "20");
  expect(icon).toHaveAttribute("viewBox", "0 0 24 24");
}

describe("metric icon geometry", () => {
  it("keeps Architecture Explorer service, store, file, and relation icons centered", () => {
    renderMetricCards(architectureMetrics);

    for (const { label } of architectureMetrics) {
      expectCenteredMetricIcon(label);
    }
  });

  it("keeps Review History metric icons centered with the same box geometry", () => {
    renderMetricCards(historyMetrics);

    for (const { label } of historyMetrics) {
      expectCenteredMetricIcon(label);
    }
  });
});

describe("application bottom banner regression", () => {
  it("keeps the operational footer after main content with its bottom-banner layout", () => {
    const operational: OperationalState = {
      capabilities: null,
      error: null,
      health: null,
      loading: false,
      refresh: vi.fn(),
    };

    render(
      <AppShell operational={operational} pathname="/architecture">
        <div>Architecture content</div>
      </AppShell>,
    );

    const main = screen.getByRole("main");
    const footer = screen.getByRole("contentinfo");
    expect(main.nextElementSibling).toBe(footer);
    expect(within(footer).getByText("CodeInsight")).toBeInTheDocument();
    expect(within(footer).getByRole("link", { name: "API documentation" })).toHaveAttribute(
      "href",
      "/api/docs",
    );

    const style = window.getComputedStyle(footer);
    expect(style.display).toBe("grid");
    expect(style.position).toBe("relative");
    expect(style.minHeight).toBe("82px");
    expect(productionRule(".app-footer")).toContain("border-top: 1px solid var(--line);");
  });
});

describe("adaptive dense workspace geometry", () => {
  it("removes the shared page cap while retaining adaptive inline gutters", () => {
    const rule = productionRule(".page.dense-workspace");
    expect(rule).toContain("width: 100%;");
    expect(rule).toContain("max-width: none;");
    expect(rule).toContain("padding-inline: clamp(20px, 2vw, 44px);");
  });
});
