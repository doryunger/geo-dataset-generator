# Windows equivalent of restart.sh: kills any running instance of the app, then starts a
# fresh one in the background.
# Usage: .\restart.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$LogFile = "app.log"
$PidFile = ".app.pid"

# Match by process name, not just the pidfile, so this also catches instances started
# manually or by a previous run whose pidfile went stale. Matched on Name alone -- CommandLine
# comes back blank for other processes in this environment even when same-user (some sandboxing
# restriction on cross-process command-line reads), so a CommandLine-based match is unreliable
# here. Name alone is fine: this is a single-purpose dev machine, nothing else runs uvicorn.exe.
$existing = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'uvicorn.exe' }
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

$uvicornExe = Join-Path $PSScriptRoot ".venv\Scripts\uvicorn.exe"
$ErrLogFile = "app.err.log"
$proc = Start-Process -FilePath $uvicornExe -ArgumentList "api:app", "--app-dir", "scripts" `
    -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $LogFile -RedirectStandardError $ErrLogFile `
    -WindowStyle Hidden -PassThru

$proc.Id | Out-File -FilePath $PidFile -Encoding ascii -NoNewline
Write-Host "App started (pid $($proc.Id)), logging to $LogFile / $ErrLogFile"
