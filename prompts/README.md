# CodeInsight prompt templates

Prompt templates are Git-tracked, human-readable Markdown files. They are the
editable source for CodeInsight's reusable analysis behavior.

## Rules

- Keep templates under `core/<phase>/` with stable names.
- Keep source bodies out of templates; runtime code attaches bounded context.
- Python owns typed input/output contracts, permissions, and validation.
- Every executed template is stored in SQLite with its path, version, content
  hash, rendered text, and run/result correlation. Git is authoring history;
  SQLite is immutable execution history.
- Prompt-only changes still require focused contract tests before commit.

The first migrated template is
`core/phases/planning/architecture-discovery.md`, adapted from V6 with
explicit V2 evidence, uncertainty, and bounded-output requirements.
