@echo off
set CODEDIR=C:\Users\erzhu419\GPR_KG_Code
set RUNID=structured_default_full_20260519
cd /d %CODEDIR%
if not exist logs mkdir logs
python -u -m experiments.run_rzdt_checkpointed --run_id %RUNID% --results_root results\rzdt_checkpointed --problems RZDT1 RZDT2 RZDT5_RR --n_reps 10 --N 150 --n0 30 --d 5 --sigma 0.04 --alpha 0.05 > logs\%RUNID%.out.log 2> logs\%RUNID%.err.log
