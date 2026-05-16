# Team Install / Onboarding

## For Each Codex Session

1. Read `CODEX.md`.
2. Read `SPEC.md`.
3. Match the task to one file in `patterns/`.
4. Use the relevant role in `modules/`.
5. Run the validator before final handoff:

```powershell
powershell -ExecutionPolicy Bypass -File validators/preflight.ps1
```

## Five AI Employees

- Spec Steward: converts goals into scoped acceptance criteria.
- Research Analyst: validates trading ideas and result claims.
- Implementation Engineer: makes code changes in the right module.
- Code Reviewer: checks regressions, risks, and missing tests.
- Test Operator: runs safe commands and reports confidence.

## Live Trading Policy

No employee may enable live trading, start schedulers, deploy Docker services, or
send broker orders without explicit user approval in the current turn.

## Distribution

Copy the five top-level items together when moving this kit to another repo:

- `CODEX.md`
- `SPEC.md`
- `patterns/`
- `validators/`
- `modules/`
- `integrations/`

