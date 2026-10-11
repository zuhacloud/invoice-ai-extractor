# Starts DocExtract (API + web UI) and opens it in your browser. Usage: .\run.ps1
# Run .\setup.ps1 once first.
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[docextract] Run .\setup.ps1 first." -ForegroundColor Yellow
    exit 1
}
& ".venv\Scripts\python.exe" run.py
