# InTAS Case Study — Setup Guide

## 1. Download InTAS

```bash
git clone https://github.com/silaslobo/InTAS  C:\InTAS
```

Or set `INTAS_ROOT` environment variable to your clone path and update `config.py`.

## 2. Install SUMO with libsumo

```bash
# Windows (recommended): download SUMO installer from https://sumo.dlr.de
# Then add SUMO to PATH and set SUMO_HOME:
set SUMO_HOME=C:\Program Files\Eclipse\Sumo
set PYTHONPATH=%SUMO_HOME%\tools;%PYTHONPATH%

# Verify libsumo is available:
python -c "import libsumo; print('libsumo OK')"
```

## 3. Parse Signal Phases (one-time)

```bash
cd GPR_KG_Code
python -m experiments.intas.parse_network
# Output: results/intas/decision_space.json
# Shows d = total decision variables, TLS IDs, green time bounds
```

## 4. Compute Baseline T0, A0, E0 (one-time, ~50 SUMO runs ≈ 3h)

```bash
python -m experiments.intas.compute_baseline
# Output: results/intas/baseline.json
```

## 5. Heteroscedastic Noise Test (Experiment 0)

```bash
python -m experiments.intas.run_hetero_test
# Output: results/intas/hetero_test.json
#         results/figures/figH_hetero.pdf
```

Expected result: Levene's test p < 0.05, confirming heteroscedastic noise.

## 6. Main Experiment

```bash
python -m experiments.intas.run_main --n_reps 5
# Each rep ≈ 2-3h; 5 reps × 2 methods ≈ 25h total
# Outputs: results/intas/GPR_KG_rep01.json ... GPR_KG_nV_rep05.json
```

## 7. Generate Figures

```bash
python -m experiments.intas.make_figures_intas
```

## File Structure

```
experiments/intas/
  __init__.py
  config.py           ← paths, simulation parameters, bounds
  parse_network.py    ← extract TLS signal phases from net.xml
  sumo_sim.py         ← libsumo simulation interface (f1, f2, f3)
  intas_problem.py    ← GPR-KG-compatible problem class
  compute_baseline.py ← compute T0, A0, E0 (run once)
  run_hetero_test.py  ← Experiment 0: heteroscedasticity test
  run_main.py         ← main experiment runner
  make_figures_intas.py ← figure generation (to be written after data)
  SETUP.md            ← this file
```

## Decision Variables

After running `parse_network.py`, `decision_space.json` will contain:
- `d`: total number of optimisable green phases (≈ 60, exact value TBD)  
- `var_map`: list of (tls_id, phase_index) for each variable
- `bounds`: [lb, ub] per variable (±50% around real Ingolstadt values)
- `defaults`: real Ingolstadt green times (baseline x₀)

## Simulation Details

| Parameter | Value |
|-----------|-------|
| Time window | 07:00–09:00 (morning peak) |
| Step length | 0.1 s |
| Car-following | Krauss (stochastic) |
| Rerouting prob | 0.82 |
| Emission model | SUMO HBEFA3 (CO₂) |
| Demand scenarios | InTAS_001.rou.xml … InTAS_022.rou.xml |
| Backend | libsumo (no socket overhead) |
