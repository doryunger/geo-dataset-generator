# Windows equivalent of stop.sh: kills any running instance of this app's two processes (backend +
# frontend dev server), without starting new ones -- the "stop" half of restart.ps1.
# Usage: .\stop.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Matched by port, not process name -- unlike the root restart.ps1 (which matches uvicorn.exe by
# name, safe there since it's the only thing that ever runs uvicorn on this machine), this app is
# meant to run *alongside* the main /manual app, which is also uvicorn.exe. Killing by whichever
# process is actually listening on this app's own port avoids taking down the other app's server.
function Stop-Port($port) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if (-not $conns) {
        Write-Host "Nothing listening on port $port"
        return
    }
    foreach ($c in $conns) {
        Write-Host "Stopping process on port $port (pid $($c.OwningProcess))..."
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
    }
}

Stop-Port 8010
Stop-Port 5173
Remove-Item -Path ".server.pid", ".web.pid" -Force -ErrorAction SilentlyContinue
