import { formatLabel } from "../format";
import { Icon } from "./Icon";

const stages = [
  {
    label: "Repository index",
    values: ["discover", "snapshot", "index"],
  },
  {
    label: "Plan specialists",
    values: ["plan_wave", "plan_follow_up_wave"],
  },
  {
    label: "Specialist analysis",
    values: ["dispatch_tasks", "verify_evidence", "deduplicate_and_correlate", "assess_coverage"],
  },
  {
    label: "Synthesis",
    values: ["synthesize", "complete"],
  },
];

const flattened = stages.flatMap((stage) => stage.values);

export function StageStepper({
  currentStage,
  terminal,
}: {
  currentStage: string | null;
  terminal: boolean;
}) {
  const current = currentStage ? flattened.indexOf(currentStage) : -1;
  const currentGroup = stages.findIndex((stage) => stage.values.includes(currentStage ?? ""));

  return (
    <ol className="stage-stepper" aria-label="Analysis stages">
      {stages.map((stage, index) => {
        const complete = terminal || current > flattened.indexOf(stage.values.at(-1) ?? "");
        const active = !terminal && index === currentGroup;
        return (
          <li className={complete ? "complete" : active ? "active" : "pending"} key={stage.label}>
            <span className="stage-number">{complete ? <Icon name="check" /> : index + 1}</span>
            <div>
              <strong>{stage.label}</strong>
              <span>
                {complete ? "Completed" : active ? formatLabel(currentStage ?? "") : "Pending"}
              </span>
            </div>
          </li>
        );
      })}
    </ol>
  );
}
