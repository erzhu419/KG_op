import argparse
import importlib.util
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/submit_scolhkg_manifest_gate_scheduler.py"
SPEC = importlib.util.spec_from_file_location("manifest_gate_submit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
V33_SCRIPT = REPO / "scripts/submit_scolhkg_v33_terminal_matrix_scheduler.py"
V33_SPEC = importlib.util.spec_from_file_location("v33_terminal_submit", V33_SCRIPT)
V33_MODULE = importlib.util.module_from_spec(V33_SPEC)
V33_SPEC.loader.exec_module(V33_MODULE)
V33_REPAIR_SCRIPT = (
    REPO / "scripts/submit_scolhkg_v33_frontier_repair_scheduler.py")
V33_REPAIR_SPEC = importlib.util.spec_from_file_location(
    "v33_frontier_repair_submit", V33_REPAIR_SCRIPT)
V33_REPAIR_MODULE = importlib.util.module_from_spec(V33_REPAIR_SPEC)
V33_REPAIR_SPEC.loader.exec_module(V33_REPAIR_MODULE)


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
            task_posterior_mandatory_universal_count=8,
            task_latent_inference_mode="authoritative",
            task_latent_calibration_mode="expert_ridge",
            exact_sampling_mode="iid",
            exact_mc_samples=2,
            exact_jobs=32,
            parallel_backend="process_fork",
            finalist_replication_budget=3,
            finalist_replication_count=2,
            finalist_observed_safety_count=2,
            finalist_replication_min_replicates=2,
            finalist_replication_delta=0.05,
            finalist_replication_variance_prior_df=2.0,
            finalist_replication_expert_stratified=True,
            finalist_replication_adaptive_race=True,
            finalist_replication_fixed_universe=True,
            source_consensus_template_count=12,
            initial_design="common_sobol",
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
            and "--task-posterior-mandatory-universal-count 8" in item["cmd"]
            and "--source-consensus-template-count 12" in item["cmd"]
            and "--initial-design common_sobol" in item["cmd"]
            and "--task-latent-inference-mode authoritative" in item["cmd"]
            and "--task-latent-calibration-mode expert_ridge" in item["cmd"]
            and "--exact-sampling-mode iid" in item["cmd"]
            and "--finalist-replication-budget 3" in item["cmd"]
            and "--finalist-replication-count 2" in item["cmd"]
            and "--finalist-observed-safety-count 2" in item["cmd"]
            and "--finalist-replication-min-replicates 2" in item["cmd"]
            and "--finalist-replication-expert-stratified" in item["cmd"]
            and "--finalist-replication-adaptive-race" in item["cmd"]
            and "--finalist-replication-fixed-universe" in item["cmd"]
            for item in specs
        ))

    def test_v33_matrix_is_preregistered_105_task_factorial(self):
        args = argparse.Namespace(
            baseline=(
                REPO / "SC-OLH-KG/performance/baselines/lodo_current.json"),
            deploy=Path("/deploy"),
            python="/env/bin/python",
            manifest=Path("/deploy/SC-OLH-KG/performance/manifest.json"),
            run_id="v33_test",
            heldouts=(
                "FactorShockStatePolicyRZDT1,InventorySupplyChain,"
                "QueueResourceControl"
            ),
            line="lodo_teacher",
            seed_start=0,
            n_seeds=7,
            nodes=",".join(V33_MODULE.CPU_NODES),
            cpu=33,
            ram_mb=8192,
            finalist_terminal_max_arms=4,
            finalist_terminal_mc_samples=2,
            allow_duplicate=False,
            ordered_semiparametric_residual=False,
            task_posterior_safe_boundary_weight=1.0,
            task_posterior_safe_pairwise_weight=1.0,
            task_posterior_safe_pairwise_history=16,
            task_posterior_safe_pairwise_floor=1e-6,
            task_latent_inference_mode="shadow",
            task_latent_calibration_mode="source_profiles",
            finalist_replication_variance_prior_df=2.0,
        )
        specs = V33_MODULE.build_specs(args)
        self.assertEqual(len(specs), 105)
        self.assertEqual(len({item["signature"] for item in specs}), 105)
        self.assertEqual(
            {item["require_node"] for item in specs},
            set(V33_MODULE.CPU_NODES),
        )
        commands = [item["cmd"] for item in specs]
        for variant, policy, override in V33_MODULE.VARIANTS:
            selected = [
                item for item in specs
                if f"/{variant}/" in item["signature"]
            ]
            self.assertEqual(len(selected), 21)
            self.assertTrue(all(
                f"--finalist-replication-policy {policy}" in item["cmd"]
                and f"--finalist-empirical-override {override}" in item["cmd"]
                for item in selected
            ))
        self.assertTrue(all(
            "--finalist-replication-budget 3" in command
            and "--finalist-replication-fixed-universe" in command
            and "--finalist-terminal-max-arms 4" in command
            and "--finalist-terminal-mc-samples 2" in command
            for command in commands
        ))
        args.task_indices = "0-4,63,104"
        wave = V33_MODULE.build_specs(args)
        self.assertEqual(len(wave), 7)
        self.assertEqual(
            [row["signature"] for row in wave],
            [specs[index]["signature"] for index in [0, 1, 2, 3, 4, 63, 104]],
        )

    def test_v33_repair_matrix_separates_frontier_and_contract_changes(self):
        args = argparse.Namespace(
            baseline=(
                REPO / "SC-OLH-KG/performance/baselines/lodo_current.json"),
            deploy=Path("/deploy"),
            python="/env/bin/python",
            manifest=Path("/deploy/SC-OLH-KG/performance/manifest.json"),
            run_id="v33_repair_test",
            heldouts=(
                "FactorShockStatePolicyRZDT1,InventorySupplyChain,"
                "QueueResourceControl"
            ),
            line="lodo_teacher",
            seed_start=0,
            n_seeds=7,
            nodes=",".join(V33_REPAIR_MODULE.CPU_NODES),
            cpu=33,
            ram_mb=8192,
            task_indices="",
            allow_duplicate=False,
        )
        args = V33_REPAIR_MODULE._configure_base_args(args)
        specs = V33_REPAIR_MODULE.build_specs(args)
        self.assertEqual(len(specs), 84)
        self.assertEqual(len({item["signature"] for item in specs}), 84)
        variants = {row[0]: row for row in V33_REPAIR_MODULE.VARIANTS}
        for name, row in variants.items():
            selected = [
                item for item in specs if f"/{name}/" in item["signature"]
            ]
            self.assertEqual(len(selected), 21)
            command = selected[0]["cmd"]
            self.assertIn(
                f"--finalist-frontier-policy {row[3]}", command)
            self.assertIn(
                f"--finalist-terminal-max-arms {row[4]}", command)
            self.assertIn(
                f"--decision-contract-mode {row[5]}", command)
        coherent = [
            item for item in specs
            if "/v33_coherent_coverage_8/" in item["signature"]
        ]
        self.assertTrue(all(
            "--exact-terminal-mode tcb_certified_lexicographic" in item["cmd"]
            and "--finalist-terminal-value-mode certified_lexicographic"
            in item["cmd"]
            for item in coherent
        ))
        self.assertTrue(all(
            item["require_node"] in V33_REPAIR_MODULE.CPU_NODES
            and "checkpoints" in item["stage_excludes"]
            for item in specs
        ))


if __name__ == "__main__":
    unittest.main()
