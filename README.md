# EP-signal-LLM-trade

Recognition Gap EP System for US equities.

このリポジトリは、Episodic Pivot（EP）を起点に「市場の認識のズレ」を検出し、候補ランキング、保有銘柄レビュー、Discord通知、Webull連携のドライラン提案までを一体化するための研究・運用システムです。

ライブ発注はデフォルトで無効です。`trading_mode=dry_run` と `allow_live_orders=false` が初期値です。

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
7. Webull連携は原則ドライランで注文案だけを作る

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
| `patterns/` | Codex向けの再利用可能な作業パターン |
| `validators/` | 安全スキャン、プリフライト、検証コマンド |
| `modules/` | 5つのAI社員ロール定義 |
| `integrations/` | チーム配布・導入用メタ情報 |

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

Discordボタンを有効にしたままBotを常駐する:

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

## Webullと発注安全性

初期設定では実注文は出ません。

```json
"trading_mode": "dry_run",
"allow_live_orders": false
```

Discordの承認ボタンを押しても、上記設定では注文案のドライラン記録だけを作ります。ライブ発注を有効化する場合は、別途小口テスト、権限確認、二段階確認フローの検証が必要です。

## Codex Development Kit

このリポジトリには、Codexを開発チームとして運用するための5層構成も含めています。

1. `SPEC.md`: プロジェクト憲法
2. `patterns/`: 再利用可能な実装・検証パターン
3. `validators/`: 安全スキャンとプリフライト
4. `modules/`: 5つのAI社員ロール
5. `integrations/`: 配布・導入用メタ情報

作業前に `CODEX.md` と `SPEC.md` を読み、対象タスクに合う `patterns/` と `modules/` を使う運用を想定しています。

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

Codexプリフライト:

```powershell
powershell -ExecutionPolicy Bypass -File validators\\preflight.ps1
```

## 注意事項

- 本リポジトリは投資助言ではありません。
- ライブ発注は明示的に有効化しない限り行いません。
- APIキー、Discord Bot Token、Webull認証情報、口座IDはコミットしないでください。
- `analysis_outputs/`、`outputs/`、`.pkl`、`.parquet`、`.env` はGit管理対象外です。
- バックテスト結果を評価する際は、手数料、スリッページ、データ時点、先見バイアス、ニュース混入を必ず確認してください。
