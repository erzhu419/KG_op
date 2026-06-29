@echo off
set CODEDIR=C:\Users\erzhu419\GPR_KG_Code
set RUNID=server311_bat_smoke_checkpointed
cd /d %CODEDIR%
if not exist logs mkdir logs
python -u -m experiments.run_rzdt_checkpointed --smoke --run_id %RUNID% --results_root results\server_validation --restart > logs\%RUNID%.out.log 2> logs\%RUNID%.err.log
