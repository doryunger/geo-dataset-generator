# Windows equivalent of restart.sh: kills any running instance of the app, then starts a
# fresh one in the background.
# Usage: .\restart.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$LogFile = "app.log"
$PidFile = ".app.pid"

# Match by cmdline, not just the pidfile, so this also catches instances started manually
# or by a previous run whose pidfile went stale.
$existing = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'uvicorn api:app' }
if ($existing) {
    $pidList = ($existing | ForEach-Object { $_.ProcessId }) -join ', '
    Write-Host "Stopping running app (pid(s): $pidList)..."
    foreach ($p in $existing) {
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
}
Remove-Item -Path $PidFile -Force -ErrorAction SilentlyContinue

if (-not (Test-Path .env)) {
    Write-Error "Missing .env (needs MAPBOX_ACCESS_TOKEN=...)"
    exit 1
}

Get-Content .env | Where-Object { $_ -match '^\s*[^#\s][^=]*=' } | ForEach-Object {
    $name, $value = $_ -split '=', 2
    [System.Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), "Process")
}

$uvicornCmd = "`"$PSScriptRoot\.venv\Scripts\uvicorn.exe`" api:app --app-dir scripts >> `"$LogFile`" 2>&1"
$proc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $uvicornCmd -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru

$proc.Id | Out-File -FilePath $PidFile -Encoding ascii -NoNewline
Write-Host "App started (pid $($proc.Id)), logging to $LogFile"
