$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

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
