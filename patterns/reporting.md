# Pattern: Reporting / Result Narrative

## Use When

The task asks for a report, summary, markdown document, PDF narrative, results
comparison, or decision memo.

## Workflow

1. Trace each claim to a concrete file, command output, or dataset.
2. Separate facts, interpretation, and next actions.
3. Include enough context for trading results: period, universe, fees, slippage,
   trade count, drawdown, and baseline.
4. Keep generated reports in the existing report/output folder for that
   experiment.

## Checks

- Verify referenced files exist.
- Avoid overstating performance or certainty.
- Run `validators/preflight.ps1` after adding operational docs or scripts.

