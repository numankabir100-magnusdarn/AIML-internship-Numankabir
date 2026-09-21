$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $pythonExe)) {
    Write-Host 'Missing project virtual environment at .venv. Create it with:' -ForegroundColor Yellow
    Write-Host '  python -m venv .venv' -ForegroundColor Yellow
    Write-Host '  .\.venv\Scripts\python.exe -m pip install -r requirements.txt' -ForegroundColor Yellow
    exit 1
}

& $pythonExe (Join-Path $projectRoot 'main.py')
