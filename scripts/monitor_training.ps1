# Windows equivalent of monitor_training.sh: shows progress of OBB training run(s) under
# models/ without needing to ask Claude -- reads the same results.csv Ultralytics writes
# live during training, plus the process table for whether it's still actually running.
#
# Usage:
#   scripts\monitor_training.ps1              # most recently active run
#   scripts\monitor_training.ps1 -All         # every *_run directory, newest first
#   scripts\monitor_training.ps1 -Watch       # re-run every 30s until Ctrl-C
param(
    [switch]$All,
    [switch]$Watch
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

function Show-Run {
    param($RunDir)

    $results = Join-Path $RunDir "results.csv"
    $argsFile = Join-Path $RunDir "args.yaml"

    Write-Host "=== $RunDir ==="

    $running = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'train_obb.*\.py' }
    if ($running) {
        $pidList = ($running | ForEach-Object { $_.ProcessId }) -join ', '
        Write-Host "process: RUNNING (pid(s) $pidList)"
    } else {
        Write-Host "process: not running right now (finished, or not started yet)"
    }

    if (-not (Test-Path $results)) {
        Write-Host "results.csv: not written yet (still on epoch 0 / initializing)"
        Write-Host ""
        return
    }

    $total = "?"
    if (Test-Path $argsFile) {
        $match = Select-String -Path $argsFile -Pattern '^epochs:\s*(\S+)' | Select-Object -First 1
        if ($match) { $total = $match.Matches[0].Groups[1].Value }
    }

    $rows = Import-Csv $results
    $last = $rows[-1]
    $epoch = [int]$last.epoch
    $elapsedS = [double]$last.time

    Write-Host ("epoch: {0} / {1}  (elapsed {2:N0}s = {3:N1} min)" -f $epoch, $total, $elapsedS, ($elapsedS / 60))

    if ($total -ne "?" -and $epoch -gt 0) {
        $perEpoch = $elapsedS / $epoch
        $remaining = [int]$total - $epoch
        $etaS = $perEpoch * $remaining
        Write-Host ("pace: {0:N1}s/epoch -> ETA ~{1:N0} min if it runs to completion" -f $perEpoch, ($etaS / 60))
        Write-Host "      (patience may stop it earlier once val metrics plateau)"
    }

    Write-Host ""
    Write-Host "latest metrics (epoch $epoch):"
    $last.PSObject.Properties | Where-Object { $_.Name -match 'precision|recall|mAP|fitness' } |
        ForEach-Object { Write-Host ("  {0,-30} {1}" -f $_.Name, $_.Value) }
    Write-Host ""
}

function Get-Runs {
    Get-ChildItem -Path "models" -Directory -Filter "*_run" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending
}

if ($All) {
    $runs = Get-Runs
    if (-not $runs) { Write-Host "No runs found under models/"; exit }
    foreach ($d in $runs) { Show-Run $d.FullName }
} elseif ($Watch) {
    while ($true) {
        Clear-Host
        Get-Date
        Write-Host ""
        $latest = Get-Runs | Select-Object -First 1
        if ($latest) { Show-Run $latest.FullName } else { Write-Host "No runs found under models/" }
        Start-Sleep -Seconds 30
    }
} else {
    $latest = Get-Runs | Select-Object -First 1
    if (-not $latest) { Write-Host "No runs found under models/"; exit 1 }
    Show-Run $latest.FullName
}
