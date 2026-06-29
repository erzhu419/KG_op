$ErrorActionPreference = "Stop"

$CodeDir = "C:\Users\erzhu419\GPR_KG_Code"
$LogDir = Join-Path $CodeDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$RunId = "structured_default_full_20260519"
$OutLog = Join-Path $LogDir "$RunId.out.log"
$ErrLog = Join-Path $LogDir "$RunId.err.log"

$Args = @(
    "-u",
    "-m", "experiments.run_rzdt_checkpointed",
    "--run_id", $RunId,
    "--results_root", "results\rzdt_checkpointed",
    "--problems", "RZDT1", "RZDT2", "RZDT5_RR",
    "--n_reps", "10",
    "--N", "150",
    "--n0", "30",
    "--d", "5",
    "--sigma", "0.04",
    "--alpha", "0.05"
)

$Process = Start-Process -FilePath "python" `
    -ArgumentList $Args `
    -WorkingDirectory $CodeDir `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -WindowStyle Hidden `
    -PassThru

"PID=$($Process.Id)"
"OUT=$OutLog"
"ERR=$ErrLog"
"RESULTS=$(Join-Path $CodeDir "results\rzdt_checkpointed\$RunId")"
