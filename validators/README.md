# validators/ - Guardrail Layer

These scripts are lightweight checks for Codex-generated changes. They are safe
to run from the workspace root and avoid touching live trading services.

## Standard Preflight

```powershell
powershell -ExecutionPolicy Bypass -File validators/preflight.ps1
```

This runs the safety scanner against the Codex kit files and compiles validator
Python files.

## Broader Safety Scan

```powershell
powershell -ExecutionPolicy Bypass -File validators/preflight.ps1 -All
```

Use `-All` when a task changed production code and a broader scan is worth the
extra time. Generated outputs and large data folders are ignored.

## Test Runner

```powershell
powershell -ExecutionPolicy Bypass -File validators/run_tests.ps1
```

The runner uses `pytest` when a test directory exists. If not, it falls back to
compile checks for the validator scripts.

