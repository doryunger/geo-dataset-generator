$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .env)) {
    Write-Error "Missing .env (needs MAPBOX_ACCESS_TOKEN=...)"
    exit 1
}
Get-Content .env | Where-Object { $_ -match '^\s*[^#\s][^=]*=' } | ForEach-Object {
    $name, $value = $_ -split '=', 2
    [System.Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), "Process")
}
$env:WORKSPACE = "experiments"
Write-Host "Serving WORKSPACE=experiments on http://127.0.0.1:8001 (production stays on 8000)"
& (Join-Path $PSScriptRoot ".venv\Scripts\uvicorn.exe") api:app --app-dir scripts --port 8001 @args
