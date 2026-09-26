Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Launching Autonomous AI Anomaly Detection Agent..." -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir

# Check for Python 3.13 where packages are installed
$py313 = "$env:LOCALAPPDATA\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe"

if (Test-Path $py313) {
    & $py313 -m streamlit run app.py
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    py -3.13 -m streamlit run app.py
} else {
    python -m streamlit run app.py
}
