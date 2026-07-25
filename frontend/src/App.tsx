import type { ReactNode } from "react";
import { AppShell } from "./components/AppShell";
import { useOperationalState } from "./hooks/useOperationalState";
import { CommandCenterPage } from "./pages/CommandCenterPage";
import { FindingsPage } from "./pages/FindingsPage";
import { NewReviewPage } from "./pages/NewReviewPage";
import { OverviewPage } from "./pages/OverviewPage";
import { PlaceholderPage } from "./pages/PlaceholderPage";
import { usePathname } from "./router";

export function App() {
  const pathname = usePathname();
  const operational = useOperationalState();
  const reviewMatch = pathname.match(/^\/reviews\/([^/]+)$/);

  let page: ReactNode;
  if (pathname === "/") {
    page = <OverviewPage operational={operational} />;
  } else if (pathname === "/new-review") {
    page = <NewReviewPage operational={operational} />;
  } else if (reviewMatch) {
    page = <CommandCenterPage runId={decodeURIComponent(reviewMatch[1])} />;
  } else if (pathname === "/architecture") {
    page = (
      <PlaceholderPage
        description="A bounded, evidence-linked repository graph is planned after its typed API projection exists."
        title="Architecture explorer"
      />
    );
  } else if (pathname === "/findings") {
    page = <FindingsPage />;
  } else if (pathname === "/history") {
    page = (
      <PlaceholderPage
        description="Cross-run comparison and trends will use stable server-derived fingerprints and aggregates."
        title="Review history"
      />
    );
  } else if (pathname === "/settings") {
    page = (
      <PlaceholderPage
        description="Local defaults and presets will be added only after their SQLite-backed settings contract is defined."
        title="Settings"
      />
    );
  } else {
    page = (
      <PlaceholderPage
        description="The requested frontend route does not exist."
        title="Page not found"
      />
    );
  }

  return (
    <AppShell operational={operational} pathname={pathname}>
      {page}
    </AppShell>
  );
}
