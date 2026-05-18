#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "analysis_outputs" / "daily_close_portfolio_system"
DEFAULT_CONFIG_PATH = ROOT / "configs" / "daily_close_portfolio_system.example.json"
DEFAULT_OPENCODE_MODELS = [
    "opencode-go/kimi-k2.6",
    "opencode-go-minimax/minimax-m2.7",
    "opencode-go/glm-5.1",
    "opencode-go/deepseek-v4-pro",
]
FMP_BASE = "https://financialmodelingprep.com/api/v3"


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "nat"} else text


def fnum(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def pct(value: Any) -> str:
    n = fnum(value)
    return "" if not np.isfinite(n) else f"{n * 100:.1f}%"


def money(value: Any) -> str:
    n = fnum(value)
    if not np.isfinite(n):
        return ""
    if abs(n) >= 1_000_000_000:
        return f"${n / 1_000_000_000:.2f}B"
    if abs(n) >= 1_000_000:
        return f"${n / 1_000_000:.2f}M"
    return f"${n:,.2f}"


def parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [clean(v) for v in value if clean(v)]
    text = clean(value)
    if not text:
        return []
    try:
        import ast

        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [clean(v) for v in parsed if clean(v)]
    except Exception:
        pass
    if "||" in text:
        return [p.strip() for p in text.split("||") if p.strip()]
    if "|" in text:
        return [p.strip() for p in text.split("|") if p.strip()]
    return [text]


ENTITY_STOPWORDS = {
    "inc",
    "inc.",
    "corp",
    "corp.",
    "corporation",
    "company",
    "co",
    "co.",
    "ltd",
    "ltd.",
    "limited",
    "plc",
    "holdings",
    "holding",
    "group",
    "technology",
    "technologies",
    "systems",
    "solutions",
    "class",
    "common",
    "stock",
}


def entity_terms(symbol: str, *names: Any) -> list[str]:
    terms = {clean(symbol).upper()}
    for name in names:
        text = clean(name)
        if not text:
            continue
        terms.add(text.lower())
        for token in re.findall(r"[A-Za-z0-9]+", text):
            lowered = token.lower()
            if len(lowered) >= 4 and lowered not in ENTITY_STOPWORDS:
                terms.add(lowered)
    return [term for term in terms if term]


def title_matches_entity(title: Any, symbol: str, terms: list[str]) -> bool:
    text = clean(title).lower()
    if not text:
        return False
    sym = clean(symbol).upper()
    if re.search(rf"(?<![A-Z0-9]){re.escape(sym)}(?![A-Z0-9])", clean(title).upper()):
        return True
    return any(term.lower() in text for term in terms if len(term) >= 4)


def filter_entity_titles(titles: Any, symbol: str, terms: list[str]) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    removed: list[str] = []
    for title in parse_list(titles):
        if title_matches_entity(title, symbol, terms):
            kept.append(title)
        else:
            removed.append(title)
    return kept, removed


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_text_safe(path: Path, max_chars: int = 20_000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars]


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def trim_text(value: Any, max_chars: int) -> str:
    text = clean(value)
    if not text:
        return "-"
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 18].rstrip() + "\n...[省略]"


def response_json(response: Any) -> Any:
    try:
        return response.json()
    except Exception:
        return None


@dataclasses.dataclass(frozen=True)
class SystemConfig:
    asof_date: str = "auto"
    candidate_dir: str = ""
    top_new_candidates: int = 10
    compare_top_n: int = 5
    fallback_holdings: tuple[str, ...] = ("MXL", "SIMO", "AAOI")
    include_account_values_in_llm_prompt: bool = False
    run_entry_pipeline_command: tuple[str, ...] = ()
    enable_opencode_review: bool = False
    opencode_models: tuple[str, ...] = tuple(DEFAULT_OPENCODE_MODELS)
    opencode_timeout_seconds: int = 1800
    deepseek_synthesis_model: str = "opencode-go/deepseek-v4-pro"
    discord_summary_char_budget: int = 3800

    @staticmethod
    def from_path(path: Path) -> "SystemConfig":
        raw = load_json(path, {})
        return SystemConfig(
            asof_date=clean(raw.get("asof_date")) or "auto",
            candidate_dir=clean(raw.get("candidate_dir")),
            top_new_candidates=int(raw.get("top_new_candidates", 10)),
            compare_top_n=int(raw.get("compare_top_n", 5)),
            fallback_holdings=tuple(s.upper().lstrip("$") for s in raw.get("fallback_holdings", ["MXL", "SIMO", "AAOI"])),
            include_account_values_in_llm_prompt=bool(raw.get("include_account_values_in_llm_prompt", False)),
            run_entry_pipeline_command=tuple(str(x) for x in raw.get("run_entry_pipeline_command", [])),
            enable_opencode_review=bool(raw.get("enable_opencode_review", False)),
            opencode_models=tuple(raw.get("opencode_models", DEFAULT_OPENCODE_MODELS)),
            opencode_timeout_seconds=int(raw.get("opencode_timeout_seconds", 1800)),
            deepseek_synthesis_model=clean(raw.get("deepseek_synthesis_model")) or "opencode-go/deepseek-v4-pro",
            discord_summary_char_budget=int(raw.get("discord_summary_char_budget", 3800)),
        )


class FmpClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get(self, endpoint: str, params: dict[str, Any] | None = None, timeout: int = 20) -> Any:
        if not self.api_key:
            return None
        query = {"apikey": self.api_key}
        if params:
            query.update(params)
        try:
            res = requests.get(f"{FMP_BASE}/{endpoint}", params=query, timeout=timeout)
            res.raise_for_status()
            return res.json()
        except Exception:
            return None

    def quote(self, symbol: str) -> dict[str, Any]:
        data = self.get(f"quote/{symbol}")
        if isinstance(data, list) and data:
            return data[0] if isinstance(data[0], dict) else {}
        return {}

    def profile(self, symbol: str) -> dict[str, Any]:
        data = self.get(f"profile/{symbol}", timeout=15)
        if isinstance(data, list) and data:
            return data[0] if isinstance(data[0], dict) else {}
        return {}

    def news(self, symbol: str, days: int = 21, limit: int = 20) -> list[dict[str, Any]]:
        to_date = datetime.utcnow().date()
        from_date = to_date - timedelta(days=days)
        data = self.get(
            "stock_news",
            {"tickers": symbol, "from": from_date.isoformat(), "to": to_date.isoformat(), "limit": limit},
        )
        return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []

    def earnings_surprises(self, symbol: str, limit: int = 4) -> list[dict[str, Any]]:
        data = self.get(f"earnings-surprises/{symbol}", {"limit": limit})
        return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []


class WebullGateway:
    def __init__(self):
        self.account_id = os.getenv("WEBULL_ACCOUNT_ID", "")
        self.trade_api = None
        self.quotes_api = None

    def connect(self) -> bool:
        app_key = os.getenv("WEBULL_APP_KEY")
        app_secret = os.getenv("WEBULL_APP_SECRET")
        if not app_key or not app_secret or not self.account_id:
            return False
        try:
            from webullsdkcore.client import ApiClient

            try:
                from webullsdkcore.common.region import Region
            except Exception:
                from webullsdkcore.common.enums import Region

            from webullsdkmdata.quotes.market_data import MarketData
            from webullsdktrade.api import API

            api_client = ApiClient(app_key, app_secret, Region.JP.value)
            self.trade_api = API(api_client)
            self.quotes_api = MarketData(api_client)
            return True
        except Exception:
            self.trade_api = None
            self.quotes_api = None
            return False

    def account_summary(self) -> dict[str, Any]:
        if self.trade_api is None:
            return {"connected": False, "error": "webull_not_connected"}
        out: dict[str, Any] = {"connected": True}
        for currency in ["JPY", "USD"]:
            try:
                res = self.trade_api.account.get_account_balance(self.account_id, currency)
                out[f"balance_status_{currency}"] = getattr(res, "status_code", "")
                data = response_json(res)
                if isinstance(data, dict):
                    out[f"balance_keys_{currency}"] = list(data.keys())
                    if currency == "JPY":
                        out["balance_raw"] = data
            except Exception as exc:
                out[f"balance_error_{currency}"] = str(exc)
        return out

    def positions(self) -> list[dict[str, Any]]:
        if self.trade_api is None:
            return []
        methods = [
            ("account", "get_account_position"),
            ("account", "get_account_positions"),
            ("account", "get_positions"),
        ]
        for parent_name, method_name in methods:
            try:
                parent = getattr(self.trade_api, parent_name)
                method = getattr(parent, method_name)
                res = method(account_id=self.account_id)
                data = response_json(res)
                parsed = self._parse_positions(data)
                if parsed:
                    return parsed
            except Exception:
                continue
        return []

    @staticmethod
    def _parse_positions(data: Any) -> list[dict[str, Any]]:
        candidates: list[Any] = []
        if isinstance(data, list):
            candidates = data
        elif isinstance(data, dict):
            for key in ["positions", "holdings", "data", "items", "account_position", "account_positions", "position_list"]:
                if isinstance(data.get(key), list):
                    candidates = data[key]
                    break
        out = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            symbol = clean(item.get("symbol") or item.get("ticker") or item.get("instrument_symbol"))
            qty = fnum(item.get("quantity") or item.get("qty") or item.get("position") or item.get("holding_quantity"), 0)
            avg_cost = fnum(item.get("avg_cost") or item.get("average_price") or item.get("cost_price") or item.get("unit_cost"), np.nan)
            market_value = fnum(item.get("market_value") or item.get("marketValue"), np.nan)
            if symbol and qty:
                out.append({"symbol": symbol.upper(), "quantity": qty, "avg_cost": avg_cost, "market_value": market_value, "raw": item})
        return out


def latest_candidate_dir() -> Path:
    dirs = []
    for path in (ROOT / "analysis_outputs").glob("ep_llm_rerating_pipeline_ep_event_asof_*"):
        if path.is_dir() and (path / "ep_llm_candidates.csv").exists():
            dirs.append(path)
    if not dirs:
        raise FileNotFoundError("No ep_llm candidate output directory found.")

    def asof_key(path: Path) -> tuple[str, float]:
        import re

        match = re.search(r"asof_(\d{8})", path.name)
        return (match.group(1) if match else "", path.stat().st_mtime)

    return max(dirs, key=asof_key)


def candidate_priority(row: pd.Series) -> float:
    ep = fnum(row.get("entry_ep_score"), 0.0)
    rev = fnum(row.get("rev_yoy"), 0.0)
    vol = fnum(row.get("post_ep_volume_persistence"), 0.0)
    post = fnum(row.get("post_ep_return_pct"), 0.0)
    positives = len(parse_list(row.get("positive_news_flags")))
    negatives = [x for x in parse_list(row.get("negative_news_flags")) if x != "generic_momentum_article"]
    score = 0.10 * ep + min(max(rev, -0.5), 1.5) * 18 + min(vol, 10) * 2.5 + positives * 3
    if post > 0.75:
        score -= 10
    if post > 1.50:
        score -= 15
    score -= 12 * len(negatives)
    return float(score)


def load_candidates(candidate_dir: Path, top_n: int) -> pd.DataFrame:
    df = pd.read_csv(candidate_dir / "ep_llm_candidates.csv")
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["daily_system_score"] = df.apply(candidate_priority, axis=1)
    df = df.sort_values(["daily_system_score", "entry_ep_score"], ascending=False, kind="mergesort")
    return df.head(top_n).reset_index(drop=True)


def holding_review(symbol: str, candidate_row: pd.Series | None, fmp: FmpClient | None) -> dict[str, Any]:
    quote = fmp.quote(symbol) if fmp else {}
    news = fmp.news(symbol, days=21, limit=12) if fmp else []
    close = fnum(quote.get("price") or quote.get("previousClose"), np.nan)
    change_pct = fnum(quote.get("changesPercentage"), np.nan) / 100 if quote else np.nan
    titles = [clean(x.get("title")) for x in news if clean(x.get("title"))]

    positives = parse_list(candidate_row.get("positive_news_flags")) if candidate_row is not None else []
    negatives = parse_list(candidate_row.get("negative_news_flags")) if candidate_row is not None else []
    news_blob = " ".join(titles).lower()
    positive_news_terms = [
        "ai",
        "data center",
        "datacenter",
        "hyperscale",
        "800g",
        "1.6t",
        "ramp",
        "guidance",
        "earnings",
        "contract",
        "partnership",
        "capacity",
        "record",
    ]
    negative_news_terms = [
        "offering",
        "dilution",
        "lawsuit",
        "class action",
        "short report",
        "downgrade",
        "misses",
        "delay",
        "cuts guidance",
    ]
    derived_positive_flags = [term for term in positive_news_terms if term in news_blob]
    derived_negative_flags = [term for term in negative_news_terms if term in news_blob]
    positives = list(dict.fromkeys(positives + [f"news:{x}" for x in derived_positive_flags[:6]]))
    negatives = list(dict.fromkeys(negatives + [f"news:{x}" for x in derived_negative_flags[:4]]))
    rev_yoy = fnum(candidate_row.get("rev_yoy"), np.nan) if candidate_row is not None else np.nan
    post_ep_volume = fnum(candidate_row.get("post_ep_volume_persistence"), np.nan) if candidate_row is not None else np.nan
    post_return = fnum(candidate_row.get("post_ep_return_pct"), np.nan) if candidate_row is not None else np.nan

    thesis_alive_score = 0
    thesis_alive_score += 2 if positives else 0
    thesis_alive_score += 1 if len(derived_positive_flags) >= 2 else 0
    thesis_alive_score += 2 if np.isfinite(rev_yoy) and rev_yoy > 0.25 else 0
    thesis_alive_score += 1 if np.isfinite(post_ep_volume) and post_ep_volume > 2 else 0
    thesis_alive_score -= 3 * len([x for x in negatives if x != "generic_momentum_article"])
    thesis_alive_score -= 1 if np.isfinite(change_pct) and change_pct < -0.08 else 0

    if thesis_alive_score >= 4 and (not np.isfinite(post_return) or post_return < 0.90):
        action = "HOLD_CORE"
        color = "green"
    elif thesis_alive_score >= 3:
        action = "HOLD_CORE_TRIM_TRADE"
        color = "blue"
    elif thesis_alive_score >= 1:
        action = "PROTECT_PROFIT"
        color = "orange"
    else:
        action = "EXIT_REVIEW"
        color = "red"

    return {
        "symbol": symbol,
        "action": action,
        "color": color,
        "close": close,
        "change_pct": change_pct,
        "rev_yoy": rev_yoy,
        "post_ep_volume_persistence": post_ep_volume,
        "post_ep_return_pct": post_return,
        "positive_flags": positives,
        "negative_flags": negatives,
        "recent_news": titles[:6],
        "thesis_alive_score": thesis_alive_score,
    }


def build_actions(candidates: pd.DataFrame, holdings: list[dict[str, Any]], reviews: list[dict[str, Any]], compare_top_n: int) -> list[dict[str, Any]]:
    held = {h["symbol"] for h in holdings}
    actions: list[dict[str, Any]] = []
    for review in reviews:
        if review["action"] in {"HOLD_CORE_TRIM_TRADE", "PROTECT_PROFIT", "EXIT_REVIEW"}:
            action_type = "TRIM" if review["action"] == "HOLD_CORE_TRIM_TRADE" else "PROTECT" if review["action"] == "PROTECT_PROFIT" else "EXIT_REVIEW"
            actions.append(
                {
                    "id": uuid.uuid4().hex[:12],
                    "created_at": now_utc_iso(),
                    "type": action_type,
                    "symbol": review["symbol"],
                    "status": "proposed",
                    "reason": review["action"],
                }
            )
    for _, row in candidates.head(compare_top_n).iterrows():
        sym = clean(row.get("symbol")).upper()
        if sym in held:
            continue
        actions.append(
            {
                "id": uuid.uuid4().hex[:12],
                "created_at": now_utc_iso(),
                "type": "WATCH",
                "symbol": sym,
                "status": "proposed",
                "reason": f"new_candidate_score={fnum(row.get('daily_system_score'), 0):.1f}",
            }
        )
    return actions


class OpenCodeReviewer:
    def __init__(self, cfg: SystemConfig, out_dir: Path):
        self.cfg = cfg
        self.out_dir = out_dir
        self.fmp = FmpClient(os.getenv("FMP_API_KEY", ""))
        self._profile_cache: dict[str, dict[str, Any]] = {}

    def run(self, candidates: pd.DataFrame, reviews: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.cfg.enable_opencode_review:
            return {"runs": [], "synthesis": {}}
        payload = self._build_prompt(candidates, reviews)
        outputs = []
        for model in self.cfg.opencode_models:
            model_dir = self.out_dir / "opencode" / model.replace("/", "_")
            model_dir.mkdir(parents=True, exist_ok=True)
            report_path = model_dir / "review.md"
            prompt_path = model_dir / "prompt.md"
            prompt = payload + f"\n\nWrite your concise Markdown review to this exact path:\n{report_path}\n"
            prompt_path.write_text(prompt, encoding="utf-8")
            started = time.time()
            try:
                res = self._run_opencode(model, prompt_path, self.cfg.opencode_timeout_seconds)
                (model_dir / "opencode_stdout.txt").write_text(res.stdout or "", encoding="utf-8")
                (model_dir / "opencode_stderr.txt").write_text(res.stderr or "", encoding="utf-8")
                if not report_path.exists() and res.stdout:
                    report_path.write_text(res.stdout, encoding="utf-8")
                outputs.append(
                    {
                        "model": model,
                        "returncode": res.returncode,
                        "seconds": round(time.time() - started, 1),
                        "report_path": str(report_path),
                        "report_exists": report_path.exists(),
                    }
                )
            except subprocess.TimeoutExpired:
                outputs.append({"model": model, "error": "timeout", "report_path": str(report_path)})
        write_json(self.out_dir / "opencode_runs.json", outputs)
        synthesis = self._run_deepseek_synthesis(candidates, reviews, outputs)
        write_json(self.out_dir / "model_synthesis.json", synthesis)
        return {"runs": outputs, "synthesis": synthesis}

    def _run_opencode(self, model: str, prompt_path: Path, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        opencode_bin = shutil.which("opencode") or shutil.which("opencode.cmd")
        if opencode_bin:
            cmd = [opencode_bin, "run", "--model", model, "Follow the attached prompt file exactly.", "-f", str(prompt_path)]
        else:
            cmd = ["cmd", "/c", "opencode", "run", "--model", model, "Follow the attached prompt file exactly.", "-f", str(prompt_path)]
        return subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            encoding="utf-8",
            errors="replace",
        )

    def _build_prompt(self, candidates: pd.DataFrame, reviews: list[dict[str, Any]]) -> str:
        compact_candidates = self._compact_candidates(candidates)
        compact_reviews = self._compact_reviews(reviews)
        return textwrap.dedent(
            f"""
            You are an independent financial-system reviewer.
            Do not assume future data. Review only these as-of daily-close records.
            Do not request or expose API keys, account ids, tokens, or private data.

            Tasks:
            1. Compare new EP recognition-gap candidates against current holdings.
            2. Decide whether any holding thesis is deteriorating.
            3. Decide whether any new candidate is clearly better than a holding.
            4. Explain the recognition gap in plain language, using evidence only.
            5. Return concise findings and action cautions. No live trading instructions.

            New candidates:
            {json.dumps(compact_candidates, ensure_ascii=False, indent=2)}

            Holding reviews:
            {json.dumps(compact_reviews, ensure_ascii=False, indent=2)}
            """
        ).strip()

    def _compact_candidates(self, candidates: pd.DataFrame) -> list[dict[str, Any]]:
        cand_cols = [
            "symbol",
            "company_name",
            "daily_system_score",
            "entry_ep_date",
            "entry_ep_lane",
            "post_ep_return_pct",
            "post_ep_high_return_pct",
            "post_ep_volume_persistence",
            "days_since_ep",
            "rev_yoy",
            "rev_qoq",
            "rev_accelerating",
            "latest_reported_date",
            "positive_news_flags",
            "negative_news_flags",
            "new_terms_after_ep",
            "pre_ep_theme_terms",
            "post_ep_theme_terms",
            "order_backlog_evidence",
            "guidance_evidence",
            "customer_contract_evidence",
            "product_launch_evidence",
            "pre_ep_news_titles",
            "post_ep_news_titles",
        ]
        records = candidates[[c for c in cand_cols if c in candidates.columns]].head(self.cfg.compare_top_n).to_dict("records")
        cleaned_records = []
        for record in records:
            symbol = clean(record.get("symbol")).upper()
            profile = self._profile_cache.get(symbol)
            if profile is None:
                profile = self.fmp.profile(symbol) if symbol else {}
                self._profile_cache[symbol] = profile
            company_name = clean(record.get("company_name")) or clean(profile.get("companyName")) or clean(profile.get("companyName"))
            terms = entity_terms(symbol, company_name, profile.get("companyName"), profile.get("companyNameLong"))
            record["company_name"] = company_name
            record["entity_terms_used"] = terms[:8]

            removed_all: list[str] = []
            for col in [
                "order_backlog_evidence",
                "guidance_evidence",
                "customer_contract_evidence",
                "product_launch_evidence",
                "pre_ep_news_titles",
                "post_ep_news_titles",
            ]:
                kept, removed = filter_entity_titles(record.get(col), symbol, terms)
                record[col] = kept
                removed_all.extend(removed[:5])

            positive_flags = parse_list(record.get("positive_news_flags"))
            if "order_or_backlog" in positive_flags and not record.get("order_backlog_evidence"):
                positive_flags.remove("order_or_backlog")
                record["new_terms_after_ep"] = [x for x in parse_list(record.get("new_terms_after_ep")) if x != "backlog_orders"]
                record["post_ep_theme_terms"] = [x for x in parse_list(record.get("post_ep_theme_terms")) if x != "backlog_orders"]
            if "customer_or_contract" in positive_flags and not record.get("customer_contract_evidence"):
                positive_flags.remove("customer_or_contract")
            if "product_launch" in positive_flags and not record.get("product_launch_evidence"):
                positive_flags.remove("product_launch")
            record["positive_news_flags"] = positive_flags
            record["negative_news_flags"] = parse_list(record.get("negative_news_flags"))
            record["removed_possible_mismatched_news_count"] = len(list(dict.fromkeys(removed_all)))
            record["evidence_quality_note"] = (
                "Some non-matching headlines were excluded by entity filter; do not use excluded headlines as evidence."
                if removed_all
                else ""
            )
            cleaned_records.append(record)
        return cleaned_records

    def _compact_reviews(self, reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact_reviews = []
        for review in reviews:
            item = dict(review)
            if not self.cfg.include_account_values_in_llm_prompt:
                item.pop("close", None)
                item.pop("market_value", None)
            compact_reviews.append(item)
        return compact_reviews

    def _run_deepseek_synthesis(
        self,
        candidates: pd.DataFrame,
        reviews: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        synth_dir = self.out_dir / "opencode" / "deepseek_synthesis"
        synth_dir.mkdir(parents=True, exist_ok=True)
        out_path = synth_dir / "discord_synthesis.json"
        prompt_path = synth_dir / "prompt.md"
        reports = []
        for output in outputs:
            report_path = Path(clean(output.get("report_path")))
            reports.append(
                {
                    "model": output.get("model"),
                    "returncode": output.get("returncode"),
                    "report": read_text_safe(report_path, max_chars=10_000),
                }
            )
        prompt = textwrap.dedent(
            f"""
            You are DeepSeek V4 Pro acting as the final Japanese Discord editor.
            Aggregate the model reports and the structured data into human-readable Japanese.

            Non-negotiable rules:
            - Do not assume future data.
            - Do not expose API keys, account IDs, tokens, private account values, or credentials.
            - Do not invent news, orders, customers, or fundamentals not present in the data.
            - Separate "evidence" from "interpretation".
            - This is decision support, not a guaranteed trading recommendation.
            - Never quote or rely on excluded/mismatched headlines. If evidence_quality_note is present, mention only that news quality was filtered.
            - Use "検討" or "候補" for actions. Do not use wording that sounds like a live order instruction.

            Output MUST be valid JSON only. No Markdown fences.
            Schema:
            {{
              "overall": "Japanese summary under 1200 chars.",
              "holdings": [
                {{
                  "symbol": "SIMO",
                  "title": "Short Japanese title",
                  "color": "green|blue|orange|red|gray",
                  "body": "Japanese Discord embed body. Aim for 2500-3600 chars when evidence exists. Max {self.cfg.discord_summary_char_budget} chars."
                }}
              ],
              "candidates": [
                {{
                  "symbol": "INOD",
                  "title": "Short Japanese title",
                  "color": "green|blue|orange|red|gray",
                  "body": "Japanese Discord embed body. Aim for 2500-3600 chars when evidence exists. Max {self.cfg.discord_summary_char_budget} chars."
                }}
              ]
            }}

            For each holding and candidate body, use this structure:
            結論:
            認識のズレ:
            根拠:
            懸念:
            次に見る点:
            アクション案:

            Make the body dense and readable. Prefer clear sentences over label lists.
            If a news item appears to belong to another company, explicitly say that evidence quality is weak.

            Structured new candidates:
            {json.dumps(self._compact_candidates(candidates), ensure_ascii=False, indent=2)}

            Structured holding reviews:
            {json.dumps(self._compact_reviews(reviews), ensure_ascii=False, indent=2)}

            Model reports to aggregate:
            {json.dumps(reports, ensure_ascii=False, indent=2)}

            Write the same JSON to this exact path:
            {out_path}
            """
        ).strip()
        prompt_path.write_text(prompt, encoding="utf-8")
        started = time.time()
        try:
            res = self._run_opencode(self.cfg.deepseek_synthesis_model, prompt_path, self.cfg.opencode_timeout_seconds)
            (synth_dir / "opencode_stdout.txt").write_text(res.stdout or "", encoding="utf-8")
            (synth_dir / "opencode_stderr.txt").write_text(res.stderr or "", encoding="utf-8")
            raw = read_text_safe(out_path, max_chars=120_000) or (res.stdout or "")
            parsed = extract_json_object(raw)
            parsed["_meta"] = {
                "model": self.cfg.deepseek_synthesis_model,
                "returncode": res.returncode,
                "seconds": round(time.time() - started, 1),
                "path": str(out_path),
            }
            return parsed
        except subprocess.TimeoutExpired:
            return {"_meta": {"model": self.cfg.deepseek_synthesis_model, "error": "timeout", "path": str(out_path)}}


def run_optional_entry_pipeline(cfg: SystemConfig, out_dir: Path) -> dict[str, Any]:
    if not cfg.run_entry_pipeline_command:
        return {"skipped": True}
    started = time.time()
    try:
        res = subprocess.run(
            list(cfg.run_entry_pipeline_command),
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
        )
        (out_dir / "entry_pipeline_stdout.txt").write_text(res.stdout or "", encoding="utf-8")
        (out_dir / "entry_pipeline_stderr.txt").write_text(res.stderr or "", encoding="utf-8")
        return {"returncode": res.returncode, "seconds": round(time.time() - started, 1)}
    except Exception as exc:
        return {"error": str(exc)}


def run_daily_close(cfg: SystemConfig) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    asof = datetime.now().strftime("%Y%m%d") if cfg.asof_date == "auto" else cfg.asof_date.replace("-", "")
    out_dir = OUTPUT_ROOT / asof
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_result = run_optional_entry_pipeline(cfg, out_dir)
    candidate_dir = Path(cfg.candidate_dir) if cfg.candidate_dir else latest_candidate_dir()
    candidates = load_candidates(candidate_dir, cfg.top_new_candidates)

    fmp = FmpClient(os.getenv("FMP_API_KEY", ""))
    webull = WebullGateway()
    webull_connected = webull.connect()
    positions = webull.positions() if webull_connected else []
    if not positions:
        positions = [{"symbol": sym, "quantity": np.nan, "avg_cost": np.nan, "market_value": np.nan} for sym in cfg.fallback_holdings]
    held_symbols = [p["symbol"] for p in positions]

    candidate_by_symbol = {str(row["symbol"]).upper(): row for _, row in candidates.iterrows()}
    all_candidate_rows = pd.read_csv(candidate_dir / "ep_llm_candidates.csv")
    all_candidate_rows["symbol"] = all_candidate_rows["symbol"].astype(str).str.upper()
    all_by_symbol = {str(row["symbol"]).upper(): row for _, row in all_candidate_rows.iterrows()}

    reviews = [holding_review(sym, all_by_symbol.get(sym), fmp) for sym in held_symbols]
    actions = build_actions(candidates, positions, reviews, cfg.compare_top_n)
    opencode_result = OpenCodeReviewer(cfg, out_dir).run(candidates, reviews)
    opencode_runs = opencode_result.get("runs", []) if isinstance(opencode_result, dict) else opencode_result
    model_synthesis = opencode_result.get("synthesis", {}) if isinstance(opencode_result, dict) else {}

    candidates.to_csv(out_dir / "new_candidate_ranking.csv", index=False)
    pd.DataFrame(reviews).to_csv(out_dir / "holding_reviews.csv", index=False)
    pd.DataFrame(actions).to_csv(out_dir / "proposed_actions.csv", index=False)
    write_json(out_dir / "actions.json", actions)

    report = {
        "created_at": now_utc_iso(),
        "candidate_dir": str(candidate_dir),
        "pipeline_result": pipeline_result,
        "webull_connected": webull_connected,
        "positions": positions,
        "top_candidates": candidates.head(cfg.compare_top_n).to_dict("records"),
        "holding_reviews": reviews,
        "actions": actions,
        "opencode_runs": opencode_runs,
        "model_synthesis": model_synthesis,
        "out_dir": str(out_dir),
    }
    write_json(out_dir / "daily_close_report.json", report)
    return report


def embed_color(name: str) -> int:
    return {
        "green": 0x2ECC71,
        "blue": 0x3498DB,
        "orange": 0xF39C12,
        "red": 0xE74C3C,
        "purple": 0x9B59B6,
        "gray": 0x95A5A6,
    }.get(name, 0x95A5A6)


ACTION_LABEL_JA = {
    "HOLD_CORE": "継続保有（コア）",
    "HOLD_CORE_TRIM_TRADE": "コア保有＋一部利確候補",
    "PROTECT_PROFIT": "利益保護を優先",
    "EXIT_REVIEW": "エグジット再点検",
    "WATCH": "新規監視候補",
    "TRIM": "一部利確",
    "PROTECT": "防御",
}


FLAG_LABEL_JA = {
    "positive_business_change": "事業変化",
    "earnings_or_guidance": "決算/ガイダンス",
    "order_or_backlog": "受注/バックログ",
    "customer_or_contract": "顧客/契約",
    "product_launch": "製品/サービス発表",
    "policy_theme": "政策テーマ",
    "generic_momentum_article": "モメンタム記事",
    "ai": "AI",
    "data center": "データセンター",
    "datacenter": "データセンター",
    "hyperscale": "ハイパースケール",
    "800g": "800G",
    "1.6t": "1.6T",
    "ramp": "生産/売上ランプ",
    "guidance": "ガイダンス",
    "earnings": "決算",
    "contract": "契約",
    "partnership": "提携",
    "capacity": "生産能力",
    "record": "過去最高",
    "offering": "増資/売出し",
    "dilution": "希薄化",
    "lawsuit": "訴訟",
    "class action": "集団訴訟",
    "short report": "空売りレポート",
    "downgrade": "格下げ",
    "misses": "未達",
    "delay": "遅延",
    "cuts guidance": "ガイダンス下方修正",
}


def action_label_ja(value: Any) -> str:
    text = clean(value)
    return ACTION_LABEL_JA.get(text, text or "-")


def flag_label_ja(value: Any) -> str:
    text = clean(value)
    if not text:
        return ""
    if text.startswith("news:"):
        term = text.split(":", 1)[1].strip()
        return f"ニュース: {FLAG_LABEL_JA.get(term.lower(), term)}"
    return FLAG_LABEL_JA.get(text, text.replace("_", " "))


def flags_ja(values: Any, limit: int = 900) -> str:
    labels = [flag_label_ja(v) for v in values if flag_label_ja(v)]
    return ", ".join(labels)[:limit] or "-"


def reason_ja(value: Any) -> str:
    text = clean(value)
    if not text:
        return "-"
    if text.startswith("new_candidate_score="):
        return f"新規候補スコア={text.split('=', 1)[1]}"
    return action_label_ja(text)


def synthesis_by_symbol(report: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    synthesis = report.get("model_synthesis") if isinstance(report.get("model_synthesis"), dict) else {}
    items = synthesis.get(key, []) if isinstance(synthesis, dict) else []
    out = {}
    for item in items:
        if isinstance(item, dict) and clean(item.get("symbol")):
            out[clean(item.get("symbol")).upper()] = item
    return out


def synthesis_meta_line(report: dict[str, Any]) -> str:
    synthesis = report.get("model_synthesis") if isinstance(report.get("model_synthesis"), dict) else {}
    meta = synthesis.get("_meta", {}) if isinstance(synthesis, dict) else {}
    if not meta:
        return "モデル統合: 未実行"
    model = clean(meta.get("model")) or "DeepSeek"
    status = "成功" if meta.get("returncode") == 0 else clean(meta.get("error")) or f"returncode={meta.get('returncode')}"
    return f"モデル統合: Kimi / MiniMax / GLM / DeepSeek → {model}で集約（{status}）"


async def post_discord_report(report: dict[str, Any], cfg: SystemConfig, stay_online: bool = False) -> None:
    try:
        import discord
    except Exception as exc:
        raise RuntimeError("discord.py is not installed. Run: pip install discord.py") from exc

    token = os.getenv("DISCORD_BOT_TOKEN", "")
    channel_id = int(os.getenv("DISCORD_CHANNEL_ID", "0") or 0)
    if not token or not channel_id:
        raise RuntimeError("DISCORD_BOT_TOKEN or DISCORD_CHANNEL_ID is missing.")

    intents = discord.Intents.default()
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready() -> None:  # type: ignore[no-untyped-def]
        channel = client.get_channel(channel_id) or await client.fetch_channel(channel_id)
        if channel is None:
            await client.close()
            return

        holding_summaries = synthesis_by_symbol(report, "holdings")
        candidate_summaries = synthesis_by_symbol(report, "candidates")
        synthesis = report.get("model_synthesis") if isinstance(report.get("model_synthesis"), dict) else {}
        overall = trim_text(synthesis.get("overall", ""), 1200) if synthesis else ""
        body_limit = max(1200, min(int(cfg.discord_summary_char_budget), 4000))
        title = "大引け後 Recognition Gap レビュー"
        summary = (
            f"新規候補: {len(report.get('top_candidates', []))}件\n"
            f"保有銘柄: {len(report.get('holding_reviews', []))}件\n"
            f"提案アクション: {len(report.get('actions', []))}件\n"
            f"Webull: 読み取り専用（注文機能なし）\n"
            f"{synthesis_meta_line(report)}"
        )
        if overall:
            summary = f"{summary}\n\n{overall}"
        embed = discord.Embed(title=title, description=summary, color=embed_color("purple"))
        embed.add_field(name="出力先", value=f"`{report.get('out_dir')}`", inline=False)
        await channel.send(embed=embed)

        for review in report.get("holding_reviews", []):
            symbol = clean(review.get("symbol")).upper()
            synthesized = holding_summaries.get(symbol, {})
            if synthesized:
                embed = discord.Embed(
                    title=clean(synthesized.get("title")) or f"保有銘柄: {symbol} -> {action_label_ja(review['action'])}",
                    description=trim_text(synthesized.get("body"), body_limit),
                    color=embed_color(clean(synthesized.get("color")) or review.get("color", "gray")),
                )
                embed.add_field(
                    name="数値",
                    value=f"終値 {money(review.get('close')) or '-'} / 当日変化率 {pct(review.get('change_pct')) or '-'} / 売上YoY {pct(review.get('rev_yoy')) or '-'}",
                    inline=False,
                )
            else:
                embed = discord.Embed(
                    title=f"保有銘柄: {review['symbol']} -> {action_label_ja(review['action'])}",
                    color=embed_color(review.get("color", "gray")),
                )
                embed.add_field(name="終値", value=money(review.get("close")), inline=True)
                embed.add_field(name="当日変化率", value=pct(review.get("change_pct")), inline=True)
                embed.add_field(name="売上YoY", value=pct(review.get("rev_yoy")), inline=True)
                embed.add_field(name="ポジティブ材料", value=flags_ja(review.get("positive_flags") or []), inline=False)
                embed.add_field(name="ネガティブ材料", value=flags_ja(review.get("negative_flags") or []), inline=False)
                news = "\n".join(f"- {t}" for t in (review.get("recent_news") or [])[:4]) or "-"
                embed.add_field(name="最近のニュース（原文）", value=news[:1000], inline=False)
            await channel.send(embed=embed)

        for cand in report.get("top_candidates", []):
            symbol = clean(cand.get("symbol")).upper()
            synthesized = candidate_summaries.get(symbol, {})
            if synthesized:
                embed = discord.Embed(
                    title=clean(synthesized.get("title")) or f"新規候補: {symbol}",
                    description=trim_text(synthesized.get("body"), body_limit),
                    color=embed_color(clean(synthesized.get("color")) or "blue"),
                )
                embed.add_field(
                    name="数値",
                    value=(
                        f"スコア {fnum(cand.get('daily_system_score'), 0):.1f} / "
                        f"EP {clean(cand.get('entry_ep_date'))} / "
                        f"EP後 {pct(cand.get('post_ep_return_pct')) or '-'} / "
                        f"売上YoY {pct(cand.get('rev_yoy')) or '-'}"
                    ),
                    inline=False,
                )
            else:
                embed = discord.Embed(
                    title=f"新規候補: {cand.get('symbol')}",
                    color=embed_color("blue"),
                )
                embed.add_field(name="スコア", value=f"{fnum(cand.get('daily_system_score'), 0):.1f}", inline=True)
                embed.add_field(name="EP", value=f"{clean(cand.get('entry_ep_date'))} / {clean(cand.get('entry_ep_lane'))}", inline=True)
                embed.add_field(name="EP後リターン", value=pct(cand.get("post_ep_return_pct")), inline=True)
                embed.add_field(name="売上YoY", value=pct(cand.get("rev_yoy")), inline=True)
                embed.add_field(name="材料", value=flags_ja(parse_list(cand.get("positive_news_flags")), limit=1000), inline=False)
            await channel.send(embed=embed)

        if report.get("actions"):
            await channel.send("提案メモです。Discordから売買操作はできません。")
            for action in report.get("actions", []):
                color = "blue" if action.get("type") == "WATCH" else "orange" if action.get("type") == "TRIM" else "red"
                embed = discord.Embed(
                    title=f"提案: {action_label_ja(action.get('type'))} {action.get('symbol')}",
                    description=f"理由: `{reason_ja(action.get('reason'))}`\nID: `{action.get('id')}`",
                    color=embed_color(color),
                )
                await channel.send(embed=embed)
        if not stay_online:
            await client.close()

    await client.start(token)


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily close EP portfolio review and Discord reporter.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--post-discord", action="store_true")
    parser.add_argument("--serve-discord", action="store_true", help="Post report and keep the Discord bot process online.")
    parser.add_argument("--no-opencode", action="store_true")
    args = parser.parse_args()

    cfg = SystemConfig.from_path(Path(args.config))
    if args.no_opencode:
        cfg = dataclasses.replace(cfg, enable_opencode_review=False)

    report = run_daily_close(cfg)
    print(json.dumps({"out_dir": report["out_dir"], "actions": len(report["actions"])}, ensure_ascii=False, indent=2))
    if args.post_discord or args.serve_discord:
        asyncio.run(post_discord_report(report, cfg, stay_online=args.serve_discord))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
