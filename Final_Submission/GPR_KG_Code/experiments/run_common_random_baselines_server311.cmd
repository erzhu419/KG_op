@echo off
cd /d C:\Users\erzhu419\GPR_KG_Code
if not exist results\logs mkdir results\logs
python experiments\run_structured_baselines.py --initial_design common_random --run_id server311_common_random_baselines_full_20260521 --results_root results\rzdt_common_random_baselines --methods cEHVI cParEGO NSGA-II-K --problems RZDT1 RZDT2 RZDT5_RR --n_reps 10 --seed_base 1000 --N 150 --n0 30 --d 5 --sigma 0.04 --alpha 0.05 --restart --force > results\logs\baselines_common_random_full_20260521.out.log 2> results\logs\baselines_common_random_full_20260521.err.log
