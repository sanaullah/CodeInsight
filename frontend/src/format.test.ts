import { describe, expect, it, vi } from "vitest";
import { formatDuration, formatLabel, formatTokens, shortId } from "./format";

describe("format helpers", () => {
  it("formats durable identifiers without hiding short values", () => {
    expect(shortId("run-short")).toBe("run-short");
    expect(shortId("run-123456789012345")).toBe("run-1234…2345");
  });

  it("formats labels, tokens, and elapsed boundaries", () => {
    expect(formatLabel("needs_attention")).toBe("Needs Attention");
    expect(formatLabel("change-set")).toBe("Change Set");
    expect(formatTokens(1234)).toContain("1,234");
    expect(formatDuration(null)).toBe("Not started");
    expect(formatDuration("2026-07-25T12:00:00Z", "2026-07-25T12:00:45Z")).toBe("45s");
    expect(formatDuration("2026-07-25T12:00:00Z", "2026-07-25T12:02:05Z")).toBe("2m 5s");
    expect(formatDuration("2026-07-25T12:00:45Z", "2026-07-25T12:00:00Z")).toBe("0s");
  });

  it("uses the current clock for active runs", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-07-25T12:00:12Z"));
    expect(formatDuration("2026-07-25T12:00:00Z")).toBe("12s");
    vi.useRealTimers();
  });
});
