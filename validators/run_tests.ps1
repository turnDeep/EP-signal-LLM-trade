param(
    [string[]]$Target = @()
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

if ($Target.Count -gt 0) {
    python -m pytest @Target
    exit $LASTEXITCODE
}

if (Test-Path "tests") {
    python -m pytest tests
    exit $LASTEXITCODE
}

if (Test-Path "Auto-Swing-Trade-Bot/tests") {
    python -m pytest Auto-Swing-Trade-Bot/tests
    exit $LASTEXITCODE
}

Write-Host "No test directory found; running validator syntax check."
$SyntaxCheck = @"
import ast
from pathlib import Path

for path in Path("validators").glob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

print("Validator syntax check passed.")
"@

$SyntaxCheck | python -
exit $LASTEXITCODE
