@echo off
cd /d C:\Users\erzhu419\GPR_KG_Code
start "gprkg_common_random_full_20260521" /min cmd /c C:\Users\erzhu419\GPR_KG_Code\experiments\run_common_random_gprkg_server311.cmd
start "gprkgnv_common_random_full_20260521" /min cmd /c C:\Users\erzhu419\GPR_KG_Code\experiments\run_common_random_gprkgnv_server311.cmd
start "baselines_common_random_full_20260521" /min cmd /c C:\Users\erzhu419\GPR_KG_Code\experiments\run_common_random_baselines_server311.cmd
echo STARTED common-random RZDT jobs
