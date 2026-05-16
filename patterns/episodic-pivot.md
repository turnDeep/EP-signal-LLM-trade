# Pattern: Episodic Pivot System

## Use When

The task asks to find, validate, trade, or monitor episodic pivot (EP) stocks.

## Detection Rules

Use the three exclusive EP lanes implemented in `scan_r3000_2025_episodic_pivots.py`.

1. `a_plus_gap_ep`
   - Gap >= 10%.
   - Volume >= 3.0x prior 20-day average.
   - Close in top 30% of the daily range.
   - Close above prior 50-day high.

2. `gap_ep`
   - Gap >= 8%.
   - Volume >= 2.0x prior 20-day average.
   - Close in top 40% of the daily range.
   - Close > open.
   - Breakout-like behavior: close above prior 20-day high or day change >= 10%.

3. `displacement_ep`
   - Day change >= 12%.
   - Volume >= 2.5x prior 20-day average.
   - Close in top 35% of the daily range.
   - Close above prior 20-day high.
   - Close > open.

## Watchlist Rule

- Add every EP signal to an EP watchlist for 20 trading days after the EP date.
- Do not buy the EP close blindly; use it to identify stocks in play.

## Entry Rules Tested

Daily-bar validation favors pullback/hold entries over pure EP-high chasing.

1. Primary: `pullback_10ma_hold`
   - Within 20 trading days after EP.
   - Low comes within 3% above the 10-day moving average.
   - Close >= 10-day moving average.
   - Close > open.
   - Close > EP low.
   - Close >= 80% of the EP close.
   - The first valid day after the EP is the only entry signal for that EP event.

2. Secondary: `pullback_20ma_hold`
   - Same as above, using the 20-day moving average.

3. Momentum confirmation: `ep_high_close_breakout`
   - Close above EP day high within the watch window.
   - This validates continuation, but daily-close entries can be late.

## Stop Rules Tested

- `ep_low`: hard disaster stop below the EP low. This preserves more upside but accepts deeper drawdowns.
- `anchored_vwap_close`: close below EP-day anchored VWAP. This cuts risk faster but can remove runners.
- `prior_low`: daily prior-low stop was too tight in 2025 validation and killed many runners; use as a warning or intraday tactical stop, not as the default daily-system stop.
- Combined daily stop with prior low is not recommended as the default.

## Tradability Filters

Use these as risk/liquidity guardrails, not as alpha filters:

- Entry price >= 5.00.
- EP prior 20-day average dollar volume >= 5,000,000.
- Market cap >= 300,000,000.
- Sector metadata present.
- Exclude `Biotechnology` unless the task explicitly includes biotech/news binary risk.

## Operational Scanner

Use `scan_ep_daily_entry_signals.py` for current daily-only EP entry scans.

Default production-style command:

```powershell
python scan_ep_daily_entry_signals.py --signal-start 2026-05-04 --signal-end 2026-05-09 --out-dir analysis_outputs\episodic_pivot_daily_entry_signals_20260504_20260508
```

The scanner:

- Combines the local Russell 3000 daily parquet with recent yfinance daily bars.
- Uses the three-lane EP detector, then applies the primary `pullback_10ma_hold` entry.
- Writes raw signals, final-tradable signals, yfinance fetch status, EP events, a Markdown report, and PNG charts.
- Renders final-tradable charts with candles, volume, 10MA, 20MA, EP high, EP low, EP day, and entry day.

## 2025 Russell 3000 Validation Notes

Outputs are saved under `analysis_outputs/episodic_pivot_entry_rules_2025/`.

- EP watchlist: 1,391 EP events across 853 symbols.
- Best raw entry profile: 10MA/20MA pullback holds with 60-day time exit.
- Best stopped profiles still had negative median trade return; the edge is skewed and depends on large winners.
- Prior-low daily stop was too tight. EP low and anchored VWAP should be tuned with intraday data before live use.

## Next Validation

- Add 5-minute data for intraday EP-high triggers and real VWAP stops.
- Add risk-to-stop filter, e.g. entry to EP low <= 12%.
- Add portfolio constraints: max positions, sector caps, and one active trade per symbol.
- Include slippage and commissions.
