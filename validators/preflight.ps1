param(
    [switch]$All,
    [switch]$AllowLiveTrading,
    [string[]]$Path = @()
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$ScanArgs = @("validators/safety_scan.py")
if ($All) {
    $ScanArgs += "--all"
}
if ($AllowLiveTrading) {
    $ScanArgs += "--allow-live-trading"
}
if ($Path.Count -gt 0) {
    $ScanArgs += $Path
}

python @ScanArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$SyntaxCheck = @"
import ast
from pathlib import Path

for path in Path("validators").glob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

print("Validator syntax check passed.")
"@

$SyntaxCheck | python -
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Preflight passed."
