import { describe, expect, it, vi } from "vitest";
import { isModifiedClick, navigate } from "./router";

describe("router", () => {
  it("updates history and emits a popstate event", () => {
    const listener = vi.fn();
    const scroll = vi.spyOn(window, "scrollTo").mockImplementation(() => undefined);
    window.addEventListener("popstate", listener);

    navigate("/new-review");

    expect(window.location.pathname).toBe("/new-review");
    expect(listener).toHaveBeenCalledOnce();
    expect(scroll).toHaveBeenCalledWith({ top: 0, behavior: "instant" });
    window.removeEventListener("popstate", listener);
  });

  it("does not duplicate the current history location", () => {
    window.history.replaceState({}, "", "/new-review");
    const dispatch = vi.spyOn(window, "dispatchEvent");
    navigate("/new-review");
    expect(dispatch).not.toHaveBeenCalled();
  });

  it("preserves browser-modified link behavior", () => {
    expect(
      isModifiedClick({
        ctrlKey: true,
        metaKey: false,
        altKey: false,
        shiftKey: false,
        button: 0,
      } as never),
    ).toBe(true);
    expect(
      isModifiedClick({
        ctrlKey: false,
        metaKey: false,
        altKey: false,
        shiftKey: false,
        button: 0,
      } as never),
    ).toBe(false);
    expect(
      isModifiedClick({
        ctrlKey: false,
        metaKey: false,
        altKey: false,
        shiftKey: false,
        button: 1,
      } as never),
    ).toBe(true);
  });
});
