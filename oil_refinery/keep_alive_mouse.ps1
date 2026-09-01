Add-Type -AssemblyName System.Windows.Forms

Write-Host "Mouse keep-alive running -- nudges the cursor every 2 minutes."
Write-Host "Close this window or press Ctrl+C to stop."

while ($true) {
    Start-Sleep -Seconds 120

    $pos = [System.Windows.Forms.Cursor]::Position
    $dx = Get-Random -Minimum -8 -Maximum 8
    $dy = Get-Random -Minimum -8 -Maximum 8
    [System.Windows.Forms.Cursor]::Position = New-Object System.Drawing.Point(($pos.X + $dx), ($pos.Y + $dy))
    Start-Sleep -Milliseconds 150
    [System.Windows.Forms.Cursor]::Position = $pos

    Write-Host "$(Get-Date -Format 'HH:mm:ss') -- nudged mouse"
}
