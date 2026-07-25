import { AppLink } from "../components/AppShell";
import { EmptyState, PageHeader, Panel } from "../components/primitives";

export function PlaceholderPage({ title, description }: { title: string; description: string }) {
  return (
    <div className="page">
      <PageHeader description={description} title={title} />
      <Panel>
        <EmptyState
          action={
            <AppLink className="button button-primary" path="/new-review">
              Start a review
            </AppLink>
          }
          description="This route is established in the new shell, but its data contract belongs to a later validated milestone."
          title="Foundation ready; feature not yet implemented"
        />
      </Panel>
    </div>
  );
}
