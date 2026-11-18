# Remove the logon task made by install-autostart.ps1 (and stop the copy it is running).
param([string]$TaskName = "DrumWizardUI")
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed task '$TaskName'."
} else {
    Write-Host "No task named '$TaskName'."
}
