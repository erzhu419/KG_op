@echo off
cd /d C:\Users\erzhu419\GPR_KG_Code
if not exist results\logs mkdir results\logs
python experiments\run_rzdt_checkpointed.py --method GPR-KG --initial_design common_random --variance_surrogate ridge_logvar --variance_surrogate_rho0 0.5 --variance_surrogate_alpha 0.01 --variance_surrogate_min_samples 20 --variance_surrogate_only_constraint --variance_surrogate_clip_low 0.5 --variance_surrogate_clip_high 2.0 --run_id server311_hvs_guarded_pilot_gprkg_20260525 --results_root results\rzdt_hvs_pilot --problems RZDT1 RZDT2 RZDT5_RR --n_reps 3 --seed_base 1000 --N 80 --n0 30 --d 5 --sigma 0.04 --alpha 0.05 --restart --force > results\logs\hvs_guarded_pilot_gprkg_20260525.out.log 2> results\logs\hvs_guarded_pilot_gprkg_20260525.err.log
