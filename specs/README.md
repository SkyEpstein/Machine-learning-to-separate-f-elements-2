# Feature specs

Each new feature gets its own dated folder, `specs/YYYY-MM-DD-short-name/`, created by copying the three files in `_template/`:

- `requirements.md`: scope, key decisions, and context (the what and why)
- `plan.md`: numbered task groups (the how), with a status updated as work proceeds
- `validation.md`: success criteria and a merge checklist, ticked off as each is met

The spec is the source of truth. When something changes, update the spec rather than applying a quick fix, so that context survives across sessions. This follows the plan-implement-verify loop in the `spec-feature` skill, and the rules in `docs/PROJECT_CONSTITUTION.md`.
