# Working on Metallix

Read [DEVELOPMENT.md](DEVELOPMENT.md) before substantive work. Use the existing
roadmap and model qualification contracts to choose the next executable gate.

## Public repository boundary

This repository is public. Keep session handoffs, agent reviews, work inventories,
machine-local measurements and raw diagnostic receipts under ignored `.agents/`.
Model payloads and generated artifacts stay in their existing ignored locations.
Do not force-add ignored reports or copy their contents into tracked docs merely
to preserve session continuity.

Committed documentation should explain user-visible behavior, stable contracts,
reproduction methods or relevant public research. Prefer updating the existing
page over adding another status report. Public benchmark results are appropriate
when deliberately curated with reproducible methods and clear limits; session
logs, private paths, account details and local operating inventories are not.
Public docs must stand on their own without requiring an ignored receipt.

Filter research by relevance to a named consumer, implementation decision or
qualification gate. A model or paper mentioned for investigation does not need
to appear in progress updates or the active roadmap. Keep unrelated findings
out of those surfaces rather than treating every research lead as a workstream.

Before committing docs, inspect the staged diff for session identifiers,
personal paths, credentials, machine-specific setup and report-only content.
Keep implementation status separate from research plans and upstream claims.

## Execution and validation

Preserve concurrent work. A shared or unowned checkout is observation-only;
perform changes in an explicitly owned checkout. Stage exact owned paths.

Use `just check` or `just check-metal` as the canonical gate, with focused checks
for the changed boundary. Serialize Cargo builds and device probes. Preserve
the configured `RUSTC_WRAPPER`. Report a failing gate accurately rather than
weakening it or hiding it behind unrelated passing tests.

Synthetic reduced graphs, isolated Metal kernels and protocol tests qualify
their stated scope only. They do not establish real checkpoint generation,
whole-request speed, beyond-RAM feasibility or general agent task quality.
Keep calibration and held-out data separate; do not widen numerical bounds to
make an implementation pass.
