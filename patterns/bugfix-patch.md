# Pattern: Bug Fix / Regression Patch

## Use When

The task includes an error, traceback, failing check, unexpected behavior, or
small correctness issue.

## Workflow

1. Reproduce or localize the failure with the narrowest command.
2. Read the surrounding code before editing.
3. Patch the root cause, not just the symptom.
4. Add or update a focused test when behavior could regress.
5. Re-run the failing command and one adjacent validation if feasible.

## Checks

```powershell
powershell -ExecutionPolicy Bypass -File validators/preflight.ps1
```

Then run the exact failing command or the closest local substitute.

