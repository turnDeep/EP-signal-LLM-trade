# Pattern: Recognition Gap EP System

## Use When

The task asks for EP candidates where a policy/industry theme and business-quality change may create a market recognition gap.

This pattern is for "old label -> new reality" research, not blind theme buying.

## Core Thesis

The best EP candidates often combine four forces:

1. EP: price and volume prove that real money has started to react.
2. EP anchor delta: compare the actual previous-earnings aftermath with the current EP aftermath.
3. Policy/industry theme: the story can persist beyond one news item.
4. Earnings/orders/backlog: the business has structurally changed.
5. Recognition gap: the market still describes the company with an old label.

## Adaptive Reasoning Loop

Use `recognition_gap_ep_system.py`.

The script uses a compact OpenMythos-style architecture:

1. Prelude: merge candidate sources.
2. Recurrent expert loops: re-score each symbol through specialists.
3. Adaptive halting: spend 2-6 loops depending on score, contradiction, theme strength, and source weakness.
4. Coda: emit action labels, evidence, contradictions, and an auditable trace.

The exported trace is not hidden chain-of-thought. It is a compact audit trail of evidence, contradictions, score stability, and loop count.

## Expert Modules

- `theme_expert`: classifies policy/industry persistence such as AI semiconductor infrastructure, data-center electrification, nuclear/grid, space/defense, critical minerals, and energy.
- `earnings_backlog_expert`: scores revenue acceleration, gross margin, profitability, EPS surprise, bookings, backlog, book-to-bill, design wins, and visibility language.
- `recognition_gap_expert`: compares the old market perception with the new economic reality and discounts candidates that already look fully recognized.
- `anchor_delta_expert`: anchors on the EP event date and compares current post-EP behavior with the first trading session after the actual previous earnings date.
- `technical_expert`: confirms EP lane and EP score only. Pullback/10MA position is not part of the recognition-gap score.
- `risk_expert`: penalizes valuation, weak profitability, large-cap torque limits, and explicit exclusion notes.
- `source_quality_expert`: tracks whether evidence is based on earnings/source material, curated notes, or hypotheses still requiring primary-source checks.
- `entry_gate`: labels whether a post-EP pullback/10MA entry exists. This gate controls buyability but does not change the score.

## Data Inputs

Default inputs:

```powershell
analysis_outputs\policy_theme_ep_rank_20260510\policy_theme_ep_ranking.csv
analysis_outputs\lite_before_surge_candidates_20260510\lite_before_surge_candidates.csv
analysis_outputs\episodic_pivot_daily_entry_signals_20260504_20260508\entry_signals_final_tradable.csv
analysis_outputs\episodic_pivot_daily_entry_signals_20260504_20260508\combined_daily_features_input.parquet
configs\recognition_gap_research_notes.csv
```

Curated notes must use `source_status`:

- `partially_verified`: allowed to boost score, still review source quality.
- `needs_primary_source`: allowed onto the watchlist, but cannot be treated as fully validated.
- `rejected`: demote or exclude theme-fit false positives.

## Command

```powershell
python recognition_gap_ep_system.py --max-candidates 180 --top-n 60 --out-dir analysis_outputs\recognition_gap_ep_system_20260515
```

By default, candidates must have an EP event. Use `--allow-no-ep` only for exploratory research.
Use `--require-ep-entry` only when you want to hide candidates that have not yet produced a post-EP pullback entry.
The actual previous-earnings anchor is also required by default. Use `--allow-missing-earnings-anchor` only when debugging data coverage.

The anchor priority is:

1. `entry_ep_date`
2. `recent_ep_date`
3. `latest_earnings_date`
4. `earnings_date`

For production-style ranking, an EP event is required, but the pullback entry is a separate gate. The previous comparison point is not a 60-day proxy anymore; it is loaded from FMP earnings calendar/surprise history and mapped to the first trading day after the previous earnings date.

## Outputs

- `recognition_gap_ranking.csv`: full ranking without traces.
- `recognition_gap_ranking_top.csv`: top candidates.
- `recognition_gap_traces.json`: loop-by-loop audit traces.
- `recognition_gap_report.md`: human-readable report.

## Action Labels

- `buy_zone_research_validated`: score is high and `entry_gate=entry_ready`.
- `high_conviction_wait_for_pullback`: score is high but the pullback gate is not ready.
- `watchlist_entry_ready`: story is strong enough to monitor and the pullback gate is ready.
- `watchlist_wait_for_pullback`: good research candidate, wait for the entry gate.
- `research_only`: insufficient combined evidence.

## Rule Of Thumb

Do not buy a stock only because the administration or market narrative likes the theme.

Buy candidates where the theme, business numbers, order/backlog quality, and EP behavior all point to the same recognition gap, then wait for the separate pullback gate.
