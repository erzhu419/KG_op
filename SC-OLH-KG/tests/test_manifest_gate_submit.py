import argparse
import importlib.util
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/submit_scolhkg_manifest_gate_scheduler.py"
SPEC = importlib.util.spec_from_file_location("manifest_gate_submit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ManifestGateSubmitTests(unittest.TestCase):
    def test_gate_is_one_seed_per_cpu_node_task_without_checkpoint_sync(self):
        args = argparse.Namespace(
            deploy=Path("/deploy"),
            python="/env/bin/python",
            manifest=Path("/deploy/SC-OLH-KG/performance/manifest.json"),
            run_id="v22_test",
            heldouts=(
                "FactorShockStatePolicyRZDT1,InventorySupplyChain"
            ),
            line="lodo_teacher",
            seed_start=0,
            n_seeds=7,
            nodes=",".join(MODULE.CPU_NODES),
            cpu=33,
            ram_mb=8192,
            ordered_cumulative_exposure=True,
            ordered_max_frequency=8,
            ordered_active_dim=2,
            ordered_frequency_penalty=0.1,
            ordered_basis_mode="diagonal_quadratic",
            ordered_adaptive_sparsity=True,
            ordered_replace_local_kernel=False,
            ordered_semiparametric_residual=False,
            ordered_latent_structure_selection=True,
            ordered_group_shared_shrinkage=True,
            ordered_group_ridge_learning=False,
            task_posterior_safe_generalized=True,
            task_posterior_safe_boundary_weight=1.0,
            task_posterior_safe_pairwise_weight=1.0,
            task_posterior_safe_pairwise_history=16,
            task_posterior_safe_pairwise_floor=1e-6,
            exact_sampling_mode="iid",
            exact_mc_samples=2,
            exact_jobs=32,
            parallel_backend="process_fork",
            finalist_replication_budget=3,
            finalist_replication_count=2,
            finalist_replication_min_replicates=2,
            finalist_replication_delta=0.05,
            finalist_replication_variance_prior_df=2.0,
            finalist_replication_expert_stratified=True,
            finalist_replication_adaptive_race=True,
            finalist_replication_fixed_universe=True,
            allow_duplicate=False,
        )
        specs = MODULE.build_specs(args)
        self.assertEqual(len(specs), 14)
        self.assertEqual(len({item["signature"] for item in specs}), 14)
        self.assertEqual(len({item["ckpt_dir"] for item in specs}), 14)
        self.assertEqual(len({item["result_dir"] for item in specs}), 14)
        self.assertTrue(all(
            item["require_node"] in MODULE.CPU_NODES for item in specs
        ))
        self.assertTrue(all(
            tuple(item["allowed_nodes"]) == MODULE.CPU_NODES for item in specs
        ))
        self.assertTrue(all(
            "checkpoints" in item["stage_excludes"] for item in specs
        ))
        self.assertTrue(all(
            "/checkpoints/" not in item["result_dir"] for item in specs
        ))
        self.assertTrue(all(
            "--ordered-basis-mode diagonal_quadratic" in item["cmd"]
            and "PYTHONUNBUFFERED=1" in item["cmd"]
            and "--ordered-adaptive-sparsity" in item["cmd"]
            and "--ordered-replace-local-kernel" not in item["cmd"]
            and "--ordered-semiparametric-residual" not in item["cmd"]
            and "--ordered-latent-structure-selection" in item["cmd"]
            and "--ordered-group-shared-shrinkage" in item["cmd"]
            and "--task-posterior-safe-generalized" in item["cmd"]
            and "--task-posterior-safe-pairwise-history 16" in item["cmd"]
            and "--exact-sampling-mode iid" in item["cmd"]
            and "--finalist-replication-budget 3" in item["cmd"]
            and "--finalist-replication-count 2" in item["cmd"]
            and "--finalist-replication-min-replicates 2" in item["cmd"]
            and "--finalist-replication-expert-stratified" in item["cmd"]
            and "--finalist-replication-adaptive-race" in item["cmd"]
            and "--finalist-replication-fixed-universe" in item["cmd"]
            for item in specs
        ))


if __name__ == "__main__":
    unittest.main()
