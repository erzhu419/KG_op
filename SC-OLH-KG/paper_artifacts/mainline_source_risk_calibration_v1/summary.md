# Source task-variance candidate, frozen for the unseen-geometry test

Only the old g task covariance is rescaled, before source conditioning; source and query kernels use the same multiplier. The old common mean/covariance, new mean4 variances and eta prior remain fixed. Eta component masses and all8 outer identity weights use the changed source likelihood.

- deconvolved identity: multiplier0.3863865504; source log-evidence gain0.725754; ESS1004.89; cumulative split gap0.00354.
- observed identity: multiplier0.3827593751; source log-evidence gain0.737116; ESS1003.28; cumulative split gap0.01396.
- Both one-dimensional fits converge in7 iterations. Both generalized common-mean refit candidates were rejected before this fit.
- On the same18 usable policy-deletion groups (35 rows), mean log predictive density improves1.255334→1.261606, while g MSE worsens0.004969→0.005302 (6.7%). This is mixed evidence; it is not an MSE validation pass.
- All4 task-deletion mixtures fail the importance criterion. Frozen selector/mean/hyperparameters already used all48 labels, so these are conditional diagnostics rather than independent heldout coverage.

The candidate is adopted only for the authorized, predeclared unseen-geometry decision experiment. No target outcome selected its parameters. The calibration is frozen; practical improvement remains to be measured.

This stage used2 new scalar source fits,112 small training-fold GLS fits,0 new source labels/simulator calls/sampling sweeps, and 44.738415 CPU seconds across2 recorded attempts. Tests:5 unique tests passed (6 executions across the initial check and regression run); see tests.json.
