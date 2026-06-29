$ErrorActionPreference = "Stop"

$Root = "E:\从Dell移动到X1\科研同行\谭真\OR投稿\Final_Submission"
$LogDir = Join-Path $Root "server311_keepalive_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$RunId = "structured_default_full_20260519"
$LocalOut = Join-Path $LogDir "$RunId.ssh.out.log"
$LocalErr = Join-Path $LogDir "$RunId.ssh.err.log"

$RemoteCommand = "cmd /c C:\Users\erzhu419\GPR_KG_Code\server311_run_checkpointed_rzdt_full.bat"
$Process = Start-Process -FilePath "ssh" `
    -ArgumentList @("server3112080", $RemoteCommand) `
    -WindowStyle Hidden `
    -RedirectStandardOutput $LocalOut `
    -RedirectStandardError $LocalErr `
    -PassThru

"LOCAL_SSH_PID=$($Process.Id)"
"LOCAL_OUT=$LocalOut"
"LOCAL_ERR=$LocalErr"
"REMOTE_RUN_ID=$RunId"
