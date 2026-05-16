# Pattern: Backtest / Research

## Use When

The task asks to test, compare, reproduce, optimize, analyze, or validate a
strategy or signal.

## Workflow

1. Locate the exact strategy implementation and the data source used by the
   current result.
2. Preserve the baseline before changing parameters or logic.
3. Make the experiment reproducible: fixed config, clear universe, date range,
   fees, slippage, timezone, and output path.
4. Prefer walk-forward, out-of-sample, or frozen-period validation over
   in-sample optimization.
5. Write outputs under `analysis_outputs/<question>/<variant_slug>/` unless the
   existing code already has a more specific convention.

## Required Reporting

- State sample period, universe, entry/exit rules, fees/slippage, and data
  vendor.
- Compare against baseline and mention trade count, drawdown, win rate, and
  whether the result is in-sample or out-of-sample.
- Flag any survivorship bias, lookahead risk, small sample size, or vendor
  limitation.

## Checks

```powershell
powershell -ExecutionPolicy Bypass -File validators/preflight.ps1
```

Then run the narrow backtest or verification script that produced the result.

