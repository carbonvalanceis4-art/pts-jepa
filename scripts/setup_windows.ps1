$ErrorActionPreference = 'Stop'

Write-Host 'PTS-JEPA Windows setup' -ForegroundColor Cyan
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw 'Python was not found. Install Python 3.10+ and ensure python.exe is on PATH.'
}

python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pytest -q

Write-Host ''
Write-Host 'Environment ready.' -ForegroundColor Green
Write-Host 'Activate with: .\.venv\Scripts\Activate.ps1'
Write-Host 'Then prepare MUMDMC with:'
Write-Host '  python data\prepare_mumdmc.py --local-source "C:\path\to\MUMDMC2025_DataSet_sample.zip"'
