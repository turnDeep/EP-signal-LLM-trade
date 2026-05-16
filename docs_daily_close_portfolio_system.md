# Daily Close Portfolio System

This is the daily-close orchestration layer for the Recognition Gap EP system.

## What It Does

1. Reads the latest `ep_llm_rerating_pipeline.py` candidate output.
2. Reads current holdings from Webull when credentials are configured.
3. Falls back to configured holdings such as `MXL`, `SIMO`, and `AAOI` when Webull is unavailable.
4. Pulls recent FMP quote/news context for holdings.
5. Produces:
   - new candidate ranking,
   - holding review,
   - proposed actions,
   - optional Discord report,
   - optional opencode go multi-model review,
   - DeepSeek Japanese synthesis for Discord when opencode review is enabled.

## Safety Defaults

- `trading_mode` defaults to `dry_run`.
- `allow_live_orders` defaults to `false`.
- API keys are read from `.env` or environment variables only.
- Secrets are not passed to opencode prompts.
- Account values are excluded from LLM prompts by default.

## Required Environment Variables

Set these in `.env` or your shell. Do not commit them.

```text
FMP_API_KEY=...
DISCORD_BOT_TOKEN=...
DISCORD_CHANNEL_ID=...
WEBULL_APP_KEY=...
WEBULL_APP_SECRET=...
WEBULL_ACCOUNT_ID=...
```

For opencode go, configure the provider credentials through opencode itself or environment variables outside the repo.

## Run Locally

Dry-run report only:

```powershell
python daily_close_portfolio_system.py --config configs\daily_close_portfolio_system.example.json --no-opencode
```

Post the report to Discord:

```powershell
python daily_close_portfolio_system.py --config configs\daily_close_portfolio_system.example.json --post-discord
```

Post the report and keep the bot alive so action buttons work:

```powershell
python daily_close_portfolio_system.py --config configs\daily_close_portfolio_system.example.json --serve-discord
```

Use `--no-opencode` when you want the old fast fallback report without multi-model review and DeepSeek synthesis.

The Discord action cards include:

- `Approve`: approves the action. In dry-run mode it records the simulated order payload.
- `Reject`: rejects the action.
- `Details`: shows the proposed order payload.
- `Confirm Live Order`: appears only after approval when `trading_mode=live` and `allow_live_orders=true`.

The default config enables opencode go review. It runs Kimi, MiniMax, GLM, and DeepSeek, then asks DeepSeek to aggregate those reports into Japanese Discord-ready summaries for both holdings and new candidates.

```json
"enable_opencode_review": true,
"deepseek_synthesis_model": "opencode-go/deepseek-v4-pro",
"discord_summary_char_budget": 3800
```

## Live Order Policy

Live order execution should remain disabled until:

1. Discord reporting works.
2. Webull balance/positions are parsed correctly.
3. Order payloads are verified with tiny test orders or paper mode.
4. The two-step Discord confirmation flow is tested in `dry_run`.

The implementation prepares proposed actions and defaults to dry-run. It does not silently place live orders.
