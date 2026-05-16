# Pattern: Data Pipeline / Feature Build

## Use When

The task touches market data ingestion, feature extraction, parquet/CSV/pickle
datasets, symbol universes, or daily/nightly pipelines.

## Workflow

1. Identify the source vendor, symbol universe, bar interval, timezone, and
   adjustment policy.
2. Avoid overwriting large datasets. Write a new versioned output unless the
   user asks for in-place regeneration.
3. Validate schema and row counts before and after the change.
4. Keep feature definitions documented when they affect model or signal output.
5. Cache responsibly and make retries/rate limits explicit for API work.

## Checks

- Small sample load or schema validation.
- No secret keys printed in logs.
- `powershell -ExecutionPolicy Bypass -File validators/preflight.ps1`.

