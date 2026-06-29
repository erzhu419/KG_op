$ErrorActionPreference = "Stop"

$TaskName = "GPRKG_RZDT_FULL_20260519"
$BatPath = "C:\Users\erzhu419\GPR_KG_Code\server311_run_checkpointed_rzdt_full.bat"

$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatPath`""

$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Force | Out-Null

Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 3

Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName,State
Get-ScheduledTaskInfo -TaskName $TaskName | Select-Object LastRunTime,LastTaskResult,NextRunTime
