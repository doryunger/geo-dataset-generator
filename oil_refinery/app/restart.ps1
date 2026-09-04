$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$REPO_ROOT = Join-Path $PSScriptRoot "..\.."

function Stop-Port($port) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        Write-Host "Stopping process on port $port (pid $($c.OwningProcess))..."
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
    }
}

Stop-Port 8010
Stop-Port 5173
Remove-Item -Path ".server.pid", ".web.pid" -Force -ErrorAction SilentlyContinue

if (-not (Test-Path (Join-Path $REPO_ROOT ".env"))) {
    Write-Error "Missing .env at repo root (needs MAPBOX_ACCESS_TOKEN=...)"
    exit 1
}

Get-Content (Join-Path $REPO_ROOT ".env") | Where-Object { $_ -match '^\s*[^#\s][^=]*=' } | ForEach-Object {
    $name, $value = $_ -split '=', 2
    [System.Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), "Process")
}
if (-not $env:INFERENCE_DEVICE) { $env:INFERENCE_DEVICE = "cpu" }
if (-not $env:HOST) { $env:HOST = "127.0.0.1" }
if (-not $env:PORT) { $env:PORT = "8010" }

$uvicornExe = Join-Path $REPO_ROOT ".venv\Scripts\uvicorn.exe"
$serverProc = Start-Process -FilePath $uvicornExe `
    -ArgumentList "server:app", "--app-dir", "server", "--host", $env:HOST, "--port", $env:PORT `
    -WorkingDirectory $PSScriptRoot -RedirectStandardOutput "server.log" -RedirectStandardError "server.err.log" `
    -WindowStyle Hidden -PassThru
$serverProc.Id | Out-File -FilePath ".server.pid" -Encoding ascii -NoNewline
Write-Host "Backend started (pid $($serverProc.Id), port $($env:PORT)), logging to server.log / server.err.log"

$webProc = Start-Process -FilePath "npm.cmd" -ArgumentList "run", "dev" `
    -WorkingDirectory (Join-Path $PSScriptRoot "web") -RedirectStandardOutput "web.log" -RedirectStandardError "web.err.log" `
    -WindowStyle Hidden -PassThru
$webProc.Id | Out-File -FilePath ".web.pid" -Encoding ascii -NoNewline
Write-Host "Frontend started (pid $($webProc.Id), port 5173), logging to web.log / web.err.log"
