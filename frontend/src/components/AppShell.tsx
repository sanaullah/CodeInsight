import type { AnchorHTMLAttributes, ReactNode } from "react";
import { useEffect, useRef, useState } from "react";
import type { OperationalState } from "../hooks/useOperationalState";
import { isModifiedClick, navigate } from "../router";
import { Icon } from "./Icon";
import { OperationalStatus } from "./OperationalStatus";

const navigation = [
  { path: "/", label: "Overview", icon: "overview" as const },
  { path: "/new-review", label: "New review", icon: "plus" as const },
  { path: "/architecture", label: "Architecture", icon: "architecture" as const },
  { path: "/findings", label: "Findings", icon: "findings" as const },
  { path: "/history", label: "History", icon: "history" as const },
  { path: "/settings", label: "Settings", icon: "settings" as const },
];

export function AppShell({
  pathname,
  operational,
  children,
}: {
  pathname: string;
  operational: OperationalState;
  children: ReactNode;
}) {
  const [menuOpen, setMenuOpen] = useState(false);
  const closeButton = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!menuOpen) return;
    closeButton.current?.focus();
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMenuOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [menuOpen]);

  return (
    <div className="app-frame">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      <button
        aria-controls="primary-navigation"
        aria-expanded={menuOpen}
        aria-label="Open navigation"
        className="mobile-menu-button"
        onClick={() => setMenuOpen(true)}
        type="button"
      >
        <Icon name="menu" />
      </button>
      {menuOpen ? (
        <button
          aria-label="Dismiss navigation"
          className="navigation-scrim"
          onClick={() => setMenuOpen(false)}
          type="button"
        />
      ) : null}
      <aside className={`sidebar ${menuOpen ? "is-open" : ""}`}>
        <div className="sidebar-header">
          <AppLink className="brand" path="/" onNavigate={() => setMenuOpen(false)}>
            <span className="brand-mark">CI</span>
            <span>CodeInsight</span>
          </AppLink>
          <button
            aria-label="Close navigation"
            className="sidebar-close"
            onClick={() => setMenuOpen(false)}
            ref={closeButton}
            type="button"
          >
            <Icon name="close" />
          </button>
        </div>
        <nav aria-label="Primary" id="primary-navigation">
          {navigation.map((item) => {
            const active =
              item.path === "/"
                ? pathname === "/"
                : pathname === item.path ||
                  (item.path === "/new-review" && pathname.startsWith("/reviews/"));
            return (
              <AppLink
                aria-current={active ? "page" : undefined}
                className={active ? "nav-link active" : "nav-link"}
                key={item.path}
                onNavigate={() => setMenuOpen(false)}
                path={item.path}
              >
                <Icon name={item.icon} />
                <span>{item.label}</span>
              </AppLink>
            );
          })}
        </nav>
        <div className="sidebar-invariant">
          <Icon name="shield" />
          <div>
            <strong>Read-only analysis</strong>
            <span>Repository writes are not available.</span>
          </div>
        </div>
      </aside>
      <div className="app-content">
        <main id="main-content">{children}</main>
        <footer className="app-footer">
          <div>
            <strong>CodeInsight</strong>
            <span>
              {operational.health?.version ?? "Version unavailable"} · FastAPI · uv-managed
            </span>
          </div>
          <OperationalStatus state={operational} />
          <a className="docs-link" href={operational.health?.runtime.api_docs_url ?? "/api/docs"}>
            <Icon name="docs" />
            API documentation
          </a>
        </footer>
      </div>
    </div>
  );
}

export function AppLink({
  path,
  onNavigate,
  children,
  onClick,
  ...props
}: {
  path: string;
  onNavigate?: () => void;
  children: ReactNode;
} & Omit<AnchorHTMLAttributes<HTMLAnchorElement>, "href">) {
  return (
    <a
      href={path}
      {...props}
      onClick={(event) => {
        onClick?.(event);
        if (event.defaultPrevented || isModifiedClick(event)) return;
        event.preventDefault();
        navigate(path);
        onNavigate?.();
      }}
    >
      {children}
    </a>
  );
}
