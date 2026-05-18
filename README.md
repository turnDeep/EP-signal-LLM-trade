# EP-signal-LLM-trade

Recognition Gap EP System for US equities.

このリポジトリは、Episodic Pivot（EP）を起点に「市場の認識のズレ」を検出し、候補ランキング、保有銘柄レビュー、Discord通知、Webullの保有情報読み取りまでを一体化するための研究・運用システムです。

Webullの売買・注文送信機能は実装していません。Discordから売買操作を行うボタンもありません。

## コンセプト

このシステムは、単に「政策テーマ銘柄を買う」ためのものではありません。

EPで株価と出来高の変化を検出し、ニュース、決算、ガイダンス、受注、バックログ、セクター文脈を重ねて、市場がまだ十分に織り込んでいない可能性のある構造変化を候補化します。

基本フロー:

1. 日足ベースでEP候補を検出する
2. EP前後の価格、出来高、ニュース文脈を比較する
3. FMPから決算、売上成長、ニュースを取得する
4. LLMで「市場の認識のズレ」を多段レビューする
5. 候補ランキングとエントリー/保有/乗り換え候補を出す
6. Discordへ日本語で投稿する
7. Webull連携は保有情報の読み取り専用にする

## 主なファイル

| パス | 役割 |
|---|---|
| `ep_llm_rerating_pipeline.py` | EP必須のRecognition Gap候補ランキングパイプライン |
| `daily_close_portfolio_system.py` | 大引け後レビュー、Webull保有取得、Discord投稿、opencode goレビュー統合 |
| `recognition_gap_ep_system.py` | ループ推論型のRecognition Gap研究エンジン |
| `scan_ep_daily_entry_signals.py` | EP後の日足エントリーシグナル検出 |
| `validate_ep_entry_rules_2025.py` | EPエントリールール検証 |
| `rank_policy_theme_ep_candidates.py` | 政策・産業テーマ付きEP候補ランキング |
| `configs/daily_close_portfolio_system.example.json` | 大引け後レビュー設定例 |

## セットアップ

Python 3.11以上を推奨します。

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

Webull OpenAPI SDKを使う場合は、Webull公式SDKの導入も必要です。環境によってパッケージ名と導入方法が異なるため、WebullのOpenAPIドキュメントに従ってください。

## 環境変数

`.env` を作成し、必要なキーを設定します。`.env` はコミット禁止です。

```text
FMP_API_KEY=...
DISCORD_BOT_TOKEN=...
DISCORD_CHANNEL_ID=...
WEBULL_APP_KEY=...
WEBULL_APP_SECRET=...
WEBULL_ACCOUNT_ID=...
OPENCODE_GO_API_KEY=...
```

`OPENAI_API_KEY` を使う場合は、`ep_llm_rerating_pipeline.py` のOpenAIクライアント経路で利用できます。現在の大引け後レビューでは、opencode go経由の各モデルレビューとDeepSeek集約を前提にしています。

## 使い方

### 1. EP/Recognition Gap候補を作る

入力CSVやParquetは環境に依存します。代表例:

```powershell
python ep_llm_rerating_pipeline.py `
  --ep-signals analysis_outputs\\episodic_pivot_daily_entry_signals_20260504_20260508\\entry_signals_final_tradable.csv `
  --daily-features analysis_outputs\\russell3000_daily_features.parquet `
  --asof-date 2026-05-14 `
  --max-candidates 80
```

主要出力:

```text
analysis_outputs/ep_llm_rerating_pipeline_ep_event_asof_YYYYMMDD_*/
  ep_llm_candidates.csv
  ep_llm_company_memos.json
  ep_llm_report.md
```

### 2. 大引け後レビューを走らせる

ドライランのみ:

```powershell
python daily_close_portfolio_system.py --config configs\\daily_close_portfolio_system.example.json --no-opencode
```

Discordへ投稿する:

```powershell
python daily_close_portfolio_system.py --config configs\\daily_close_portfolio_system.example.json --post-discord
```

Discord Botを常駐する:

```powershell
python daily_close_portfolio_system.py --config configs\\daily_close_portfolio_system.example.json --serve-discord
```

## Discord投稿の設計

`daily_close_portfolio_system.py` は以下を行います。

1. 最新の `ep_llm_candidates.csv` を読み込む
2. Webullから保有銘柄を取得する
3. FMPで株価、ニュース、プロフィールを補完する
4. Kimi、MiniMax、GLM、DeepSeekで独立レビューを実行する
5. DeepSeek V4 Proで日本語のDiscord向け要約に集約する
6. 保有銘柄、新規候補、提案アクションをDiscordへ投稿する

ニュース混入対策として、銘柄名・会社名に一致しないニュース見出しはLLM入力前に除外します。除外された見出しは根拠として使いません。

## Webull連携

Webullは保有銘柄と口座情報の読み取り用途だけで使います。

注文作成、注文送信、発注確認、Discord上の承認/却下/詳細ボタンは削除済みです。Discord投稿は分析メモと候補表示のみを行います。

## 検証

最低限の構文チェック:

```powershell
python -m py_compile `
  daily_close_portfolio_system.py `
  ep_llm_rerating_pipeline.py `
  recognition_gap_ep_system.py `
  scan_ep_daily_entry_signals.py `
  validate_ep_entry_rules_2025.py
```

## 注意事項

- 本リポジトリは投資助言ではありません。
- Webullの売買・注文送信機能はありません。
- APIキー、Discord Bot Token、Webull認証情報、口座IDはコミットしないでください。
- `analysis_outputs/`、`outputs/`、`.pkl`、`.parquet`、`.env` はGit管理対象外です。
- バックテスト結果を評価する際は、手数料、スリッページ、データ時点、先見バイアス、ニュース混入を必ず確認してください。
