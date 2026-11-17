# Levanta el sistema completo en un solo paso: dependencias, frontend y API.
#   .\run.ps1              -> http://127.0.0.1:8000
#   .\run.ps1 -Port 9000
param([int]$Port = 8000)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) {
    Write-Host "==> creando entorno virtual .venv"
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "no se pudo crear .venv" }
}
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Write-Host "==> instalando dependencias de Python"
& $py -m pip install -q -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "fallo pip install" }

Write-Host "==> construyendo el frontend"
Push-Location frontend
if (-not (Test-Path node_modules)) { npm ci; if ($LASTEXITCODE -ne 0) { throw "fallo npm ci" } }
npm run build
if ($LASTEXITCODE -ne 0) { throw "fallo npm run build" }
Pop-Location

New-Item -ItemType Directory -Force data | Out-Null
Write-Host "==> API + cliente web en http://127.0.0.1:$Port"
& $py -m uvicorn backend.api.main:app --host 127.0.0.1 --port $Port
