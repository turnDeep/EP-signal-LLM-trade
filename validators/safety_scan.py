#!/usr/bin/env python
"""Lightweight safety checks for Codex-generated changes."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


DEFAULT_TARGETS = [
    "CODEX.md",
    "SPEC.md",
    "patterns",
    "validators",
    "modules",
    "integrations",
]

ALL_TARGETS = [
    "CODEX.md",
    "SPEC.md",
    "patterns",
    "validators",
    "modules",
    "integrations",
    "src",
    "configs",
    "Auto-Swing-Trade-Bot/core",
    "Auto-Swing-Trade-Bot/signals",
    "Auto-Swing-Trade-Bot/backtesting",
    "Auto-Swing-Trade-Bot/research",
    "Auto-Swing-Trade-Bot/scripts",
    "Auto-Swing-Trade-Bot/configs",
]

IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "env",
    "analysis_outputs",
    "outputs",
    "data",
    "reports",
    "scratch",
    "_reference_repos",
    "vendor",
    "node_modules",
}

TEXT_SUFFIXES = {
    ".py",
    ".ps1",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".cfg",
    ".ini",
}

SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*=\s*['\"][^'\"\s]{12,}['\"]"),
    re.compile(r"(?i)(WEBULL|FMP|OPENAI|ANTHROPIC|GITHUB).{0,24}(key|secret|token).{0,8}['\"][^'\"\s]{12,}['\"]"),
]

LIVE_TRADING_PATTERNS = [
    re.compile(r"(?i)\b(dry_run|paper|demo)\s*=\s*False\b"),
    re.compile(r"(?i)\b(live_trading|enable_live_orders)\s*=\s*True\b"),
    re.compile(r"(?i)\b(place_order|submit_order|send_order)\s*\("),
]


def iter_files(root: Path, targets: list[str]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        path = (root / target).resolve()
        if not path.exists():
            continue
        if path.is_file():
            files.append(path)
            continue
        for child in path.rglob("*"):
            if child.is_dir():
                continue
            if any(part in IGNORED_DIRS for part in child.relative_to(root).parts):
                continue
            if child.suffix.lower() in TEXT_SUFFIXES:
                files.append(child)
    return files


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig", errors="ignore")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Specific files or directories to scan")
    parser.add_argument("--all", action="store_true", help="Scan main code areas too")
    parser.add_argument(
        "--allow-live-trading",
        action="store_true",
        help="Allow live order patterns for explicitly approved live-trading work",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    targets = args.paths or (ALL_TARGETS if args.all else DEFAULT_TARGETS)
    files = iter_files(root, targets)
    findings: list[str] = []

    for path in files:
        rel = path.relative_to(root)
        text = read_text(path)
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(f"{rel}: possible hard-coded secret")
                break
        if not args.allow_live_trading:
            for pattern in LIVE_TRADING_PATTERNS:
                if pattern.search(text):
                    findings.append(f"{rel}: live-trading-sensitive pattern: {pattern.pattern}")
                    break

    if findings:
        print("Safety scan failed:")
        for finding in findings:
            print(f" - {finding}")
        print("Use --allow-live-trading only after explicit user approval for live order work.")
        return 1

    print(f"Safety scan passed ({len(files)} files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

