@echo off
cd /d C:\Users\erzhu419\GPR_KG_Code
if not exist results\logs mkdir results\logs
python experiments\run_rzdt_checkpointed.py --method GPR-KG --initial_design common_random --replication_policy boundary --replication_max_per_solution 3 --replication_score_threshold 0.0005 --replication_boundary_scale 1.0 --run_id server311_ra_boundary_pilot_gprkg_20260526 --results_root results\rzdt_ra_pilot --problems RZDT1 RZDT2 RZDT5_RR --n_reps 3 --seed_base 1000 --N 80 --n0 30 --d 5 --sigma 0.04 --alpha 0.05 --restart --force > results\logs\ra_boundary_pilot_gprkg_20260526.out.log 2> results\logs\ra_boundary_pilot_gprkg_20260526.err.log
