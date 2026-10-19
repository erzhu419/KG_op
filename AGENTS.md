# KG_op execution policy

SC-OLH-KG meaningful computation runs through the existing scheduleurm scheduler
on node001–node006. This includes simulation, source fitting, posterior restoration,
multi-task evaluation and CPU-heavy saved-trajectory analysis. Do not launch local
multi-process experiments. Local work is limited to editing, small inspections,
deterministic wiring tests and reading compact results.

Use the explicit node001–node006 pool and CPU-only `vram=0`. For independent batch
items use `submit-cpu-batch`; size workers from live free physical cores and keep
OMP/MKL/OpenBLAS threads at 1 per worker. Let the scheduler use available nodes;
do not hard-pin scientific shards to occupied nodes or fall back to local.

Reuse frozen sources, algorithms, budgets and random streams. Distribute independent
items within a stage while keeping the existing global search, terminal-input,
verification and truth-scoring barriers. Preserve completed work, failures and fees;
an infrastructure move does not authorize rerunning a completed scientific cohort.

Keep raw trajectories, large tables and model files on the server. Read logs and
small JSON summaries first; fetch selected CSVs only when needed, and checkpoints
only when logs or tables cannot answer the question. Never automatically pull an
entire experiment directory. See `SC-OLH-KG/docs/mainline_cluster_execution.md` for
the deployed environment and entry points.
