# patterns/ - Knowledge Layer

Use these playbooks automatically when the task description matches the pattern.
They are short on purpose: load only the files relevant to the current request.

| Task signal | Pattern |
|---|---|
| backtest, verify, reproduce, performance, walk-forward, strategy research | `backtest-research.md` |
| live trader, broker, order, scheduler, Webull, account, buying power | `live-trading-change.md` |
| dataset, feature, parquet, CSV, FMP, yfinance, ingestion, pipeline | `data-pipeline.md` |
| bug, failing test, exception, traceback, regression | `bugfix-patch.md` |
| report, summary, dashboard, markdown/PDF result narrative | `reporting.md` |
| episodic pivot, EP, gap catalyst, stocks in play, post-EP entry | `episodic-pivot.md` |
| recognition gap, policy theme, LITE, backlog, bookings, multibagger candidate | `recognition-gap-ep.md` |

Always combine the pattern with `SPEC.md` and the responsible module under
`modules/`.
