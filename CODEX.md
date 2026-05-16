# CODEX.md - BreakOut Codex Operating Manual

This file is the entry point for Codex work in this workspace. Read `SPEC.md`
first, then load the matching pattern and module files for the task.

## Five-Layer Stack

1. `SPEC.md` - memory layer: project constitution, expectations, naming, safety.
2. `patterns/` - knowledge layer: reusable task playbooks.
3. `validators/` - guardrail layer: preflight, safety scan, and test runners.
4. `modules/` - delegation layer: five AI employee role prompts.
5. `integrations/` - integration layer: manifest, onboarding, and team runbook.

## Default Operating Loop

1. Identify the task type and read the matching file under `patterns/`.
2. Pick the responsible role from `modules/`; use more than one only when the
   task genuinely needs independent review or validation.
3. Make the smallest useful change that satisfies the request.
4. Run `powershell -ExecutionPolicy Bypass -File validators/preflight.ps1`.
5. Run targeted tests or the command recommended by the relevant pattern.
6. Report what changed, what was verified, and any remaining risk.

## Hard Guardrails

- Live trading stays disabled unless the user explicitly asks for live trading
  changes in that turn. Prefer demo, paper, or dry-run behavior.
- Do not expose, print, copy, or commit secrets. Treat `.env`, account IDs,
  broker tokens, and API keys as sensitive.
- Do not overwrite large datasets, parquet snapshots, pickles, or reports unless
  the user explicitly asks for regeneration.
- Do not claim strategy quality from in-sample results alone. Include fees,
  slippage, date ranges, and out-of-sample or walk-forward evidence when making
  performance claims.
- Root `C:\Users\plane\BreakOut` is a workspace containing multiple repos; do
  not assume it is a single Git repository.

## Fast Task Routing

- Research/backtest request: `patterns/backtest-research.md` and
  `modules/02-research-analyst.md`.
- Episodic pivot / EP request: `patterns/episodic-pivot.md`,
  `patterns/backtest-research.md`, and `modules/02-research-analyst.md`.
- Live trading or broker behavior: `patterns/live-trading-change.md`,
  `modules/03-implementation-engineer.md`, and `modules/05-test-operator.md`.
- Data ingestion or feature pipeline: `patterns/data-pipeline.md`.
- Bug fix: `patterns/bugfix-patch.md`.
- Report or result summary: `patterns/reporting.md`.
- Review request: `modules/04-code-reviewer.md`.
