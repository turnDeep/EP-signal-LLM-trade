# Pattern: Live Trading / Broker / Scheduler Change

## Use When

The task touches order placement, broker APIs, account state, scheduler timing,
position sizing, buying power, stop logic, live quotes, or deployment.

## Workflow

1. Confirm whether the path can place real orders.
2. Keep demo/paper/dry-run as the default.
3. Isolate execution changes from signal research changes.
4. Preserve risk caps, position sizing, max positions, and end-of-day flattening
   unless the user explicitly asks to change them.
5. Add or update tests around order intent, not real broker side effects.

## Required Safety Notes

- Say whether the final code can place live orders.
- Never reveal account IDs, tokens, or API keys.
- Do not run a scheduler, Docker deployment, or live trader unless explicitly
  requested.

## Checks

```powershell
powershell -ExecutionPolicy Bypass -File validators/preflight.ps1
```

Prefer dry-run tests, mocks, or compile checks over live API calls.

