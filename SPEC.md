# SPEC.md - BreakOut Project Constitution

## Mission

BreakOut is a research and automation workspace for US equity breakout,
momentum, ORB, VWAP/ATR, swing, day-trading, and Recognition Gap EP systems.
The workspace mixes research scripts, generated reports, large market datasets,
and several nested strategy repos. Codex should preserve reproducibility,
protect live-trading safety, and keep experiments traceable.

## Workspace Map

- `src/`: reusable research pipeline code for features, backtests, validation,
  reports, and model experiments.
- `Auto-Swing-Trade-Bot/`: primary structured trading bot repo with `core/`,
  `signals/`, `backtesting/`, `research/`, `scripts/`, and `configs/`.
- `Stallion-System-Trade*`, `QQQ-VWAP-ATRTrailingStop*`, `vwap_repo/`: related
  strategy implementations and reference systems.
- `analysis_outputs/`, `outputs/`, `reports/`, `data/`: generated artifacts and
  datasets. Treat as outputs unless a task explicitly targets them.
- `recognition_gap_ep_system.py`: adaptive-loop research engine for EP plus
  policy/industry theme plus earnings/orders/backlog recognition-gap ranking.
- `patterns/recognition-gap-ep.md`: reusable playbook for the Recognition Gap
  EP workflow.
- `scratch/`: temporary analysis. Keep durable logic out of this folder.
- `_reference_repos/`: read-only reference material unless the user asks to
  patch it.

## Naming Rules

- Python files, functions, variables, and experiment slugs use `snake_case`.
- Classes use `PascalCase`; constants and environment variables use
  `UPPER_SNAKE_CASE`.
- Research entry points should keep existing prefixes:
  `analyze_*`, `run_*`, `verify_*`, `search_*`, `export_*`, `reproduce_*`.
- Generated experiment folders should be named
  `analysis_outputs/<strategy_or_question>/<variant_slug>/` when practical.
- Config files should use YAML or JSON, not hard-coded parameters in scripts,
  when the value is expected to be tuned or reused.

## Engineering Expectations

- Prefer existing modules and local helper APIs over new abstractions.
- Keep edits scoped to the target repo or module. This workspace contains many
  sibling repos with similar files.
- Separate research code from production/live-trading code. Research scripts may
  explore; live-trading paths must be conservative and explicit.
- Record assumptions: market universe, timeframe, timezone, fees, slippage,
  split adjustment, data vendor, and train/test period.
- Use structured data readers (`pandas`, `pyarrow`, YAML/JSON parsers) instead
  of ad hoc parsing when possible.
- Do not silently change position sizing, risk caps, broker mode, order type, or
  execution timing.

## Trading Safety

- Default to demo, paper, dry-run, or backtest execution.
- Live order placement requires an explicit user request in the current turn.
- Changes touching broker code, order submission, account state, credentials,
  buying power, or scheduler activation require a targeted validation plan.
- Never print secrets or account identifiers in logs or final answers.
- Any deployment or scheduler change must state whether it can place real
  orders.

## Validation Standard

Before calling work complete, run the narrowest useful checks:

```powershell
powershell -ExecutionPolicy Bypass -File validators/preflight.ps1
```

Then run tests or scripts relevant to the changed area. Examples:

```powershell
python -m compileall -q validators
python -m pytest
python Auto-Swing-Trade-Bot\scripts\backtest.py
```

If a command is too expensive, unavailable, or unsafe, say so and explain the
substitute check.

## Definition Of Done

- The task is implemented or answered in the smallest reasonable scope.
- Matching pattern(s) and role module(s) were followed.
- Safety scan and relevant tests/checks were run or explicitly deferred.
- Outputs are traceable, and live-trading risk is clear.
