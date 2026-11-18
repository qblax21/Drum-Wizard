# Start drumwizard-ui automatically when you log in to Windows (a Task Scheduler task for the current user).
#
#   powershell -ExecutionPolicy Bypass -File scripts\install-autostart.ps1                          # this PC only (127.0.0.1)
#   powershell -ExecutionPolicy Bypass -File scripts\install-autostart.ps1 -HostName 100.70.135.52   # also reachable over Tailscale
#
# No admin rights needed. The server runs hidden, restarts itself 30 s after it stops (for example if the Tailscale
# address is not up yet at logon), and writes its output to %LOCALAPPDATA%\DrumWizard\server.log.
# Remove it again with scripts\uninstall-autostart.ps1.
param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8765,
    [string]$TaskName = "DrumWizardUI"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root ".venv\Scripts\drumwizard-ui.exe"
if (-not (Test-Path $exe)) { throw "Not found: $exe. Create the venv and run: pip install -e '.[server]'" }
$logDir = Join-Path $env:LOCALAPPDATA "DrumWizard"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir "server.log"

$loop = "while (`$true) { & '$exe' --host $HostName --port $Port *>> '$log'; Start-Sleep 30 }"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -WorkingDirectory $root `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -Command `"$loop`""
$me = "$env:USERDOMAIN\$env:USERNAME"
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $me
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId $me -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
    -Description "Drum Wizard web UI" -Force | Out-Null
Write-Host "Installed task '$TaskName': drumwizard-ui will start at logon on http://${HostName}:$Port"
Write-Host "Start it now without logging out:  Start-ScheduledTask -TaskName $TaskName"
Write-Host "Log file: $log"
