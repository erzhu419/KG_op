#!/usr/bin/env python3
"""Submit paired V53 MC-fidelity or constrained-certificate gates."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


closure = _load(
    "promoted_v51_closure_submit",
    ROOT / "scripts/submit_scolhkg_promoted_v51_closure_gate_scheduler.py",
)
CPU_NODES = closure.CPU_NODES
DEFAULT_SOURCE_RUN_ID = closure.DEFAULT_SOURCE_RUN_ID
CONTROL = "v51_control"
V52 = "v52_action_superset"
V53 = "v53_certificate_constrained"
FIDELITY = ("v53_mc8", "v53_mc32")
HIGH_FIDELITY = "v53_mc128"
V54_FIDELITY = ("v54_mc128", "v54_mc512")
V55_FIDELITY = ("v55_mc128", "v55_mc512")
V56_FIDELITY = ("v56_confirm2048", "v56_confirm4096")
V57 = "v57_posterior_safe_terminal"
V58 = "v58_guard_decomposed_support"
V59 = "v59_terminal_gaussian_verification"
V60 = "v60_ordered_shortlist_verification"
VARIANTS = (
    CONTROL, V52, V53, *FIDELITY, HIGH_FIDELITY,
    *V54_FIDELITY, *V55_FIDELITY, *V56_FIDELITY,
    V57, V58, V59, V60,
)



def _command_option(cmd, option):
    tokens = shlex.split(str(cmd))
    try:
        index = tokens.index(str(option))
    except ValueError as exc:
        raise ValueError(f"V53 task is missing {option}") from exc
    if index + 1 >= len(tokens):
        raise ValueError(f"V53 task has no value for {option}")
    return tokens[index + 1]


def _local_deploy_path(args, remote_path):
    marker = "/SC-OLH-KG/"
    remote_path = str(remote_path)
    if marker not in remote_path:
        raise ValueError(
            "V53 initial-design path is outside the SC-OLH-KG deploy tree: "
            f"{remote_path}"
        )
    relative = remote_path.split(marker, 1)[1]
    return Path(args.deploy) / "SC-OLH-KG" / relative


def validate_frozen_design_seed_coverage(args, specs):
    """Fail locally when a frozen proposal does not contain requested seeds."""
    required = {}
    for spec in specs:
        remote_path = _command_option(spec["cmd"], "--initial-design-file")
        seed = int(_command_option(spec["cmd"], "--seed"))
        local_path = _local_deploy_path(args, remote_path)
        required.setdefault(local_path, set()).add(seed)

    coverage = {}
    for path, seeds in required.items():
        if not path.is_file():
            raise FileNotFoundError(
                "cannot validate V53 frozen-design seeds because the local "
                f"deploy mirror is missing {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        available = {int(seed) for seed in payload.get("designs", {})}
        missing = sorted(seeds - available)
        if missing:
            available_text = (
                f"{min(available)}..{max(available)}" if available else "none"
            )
            raise ValueError(
                f"frozen design {path} is missing requested seeds {missing}; "
                f"available seeds: {available_text}"
            )
        coverage[str(path)] = sorted(seeds)
    return coverage


def _v53_profile(
    common,
    mc_samples,
    score_normalization,
    score_transform="identity",
    new_action_count=6,
    guard_mode="uniform_score",
    pairwise_prefix_samples=32,
    pairwise_error_multiplier=1.25,
    new_action_policy=(
        "canonical_plus_posterior_risk_certificate_coverage"),
):
    bounded = str(score_transform) == "bounded_current_gain"
    paired_difference = str(guard_mode) == "paired_nested_difference"
    paired_absolute = str(guard_mode) == "paired_nested_absolute"
    independent_confirmation = str(guard_mode) == (
        "independent_confirmation")
    paired = paired_difference or paired_absolute
    normalized = str(score_normalization) == "current_terminal"
    if bounded and normalized:
        raise ValueError(
            "bounded_current_gain must use score_normalization=none")
    if paired and not bounded:
        raise ValueError(
            "paired_nested_difference requires bounded_current_gain")
    if independent_confirmation and not bounded:
        raise ValueError(
            "independent_confirmation requires bounded_current_gain")
    return {
        **common,
        "implementation_contract_id": (
            "v56_independent_confirmation_guard"
            if independent_confirmation
            else (
                "v55_current_relative_joint_guard"
                if paired_absolute
                else (
                    "v54_paired_nested_difference_guard"
                    if paired_difference
                    else (
                        "v53_constrained_certificate_deficit_bounded_gain"
                        if bounded
                        else (
                        "v53_constrained_certificate_deficit_normalized"
                        if normalized
                        else "v53_constrained_certificate_deficit"
                        )
                    )
                )
            )
        ),
        "theory_contract_id": (
            "v56_independent_confirmation_finite_look_v1"
            if independent_confirmation
            else (
                "v55_current_relative_joint_improvement_v1"
                if paired_absolute
                else (
                    "v54_paired_difference_guard_v1"
                    if paired_difference
                    else (
                    "v53_constrained_certificate_deficit_v3"
                    if bounded
                    else (
                        "v53_constrained_certificate_deficit_v2"
                        if normalized
                        else "v53_constrained_certificate_deficit_v1"
                        )
                    )
                )
            )
        ),
        "exact_mc_samples": int(mc_samples),
        "evaluate_or_replicate_new_action_count": int(new_action_count),
        "evaluate_or_replicate_new_action_policy": str(
            new_action_policy),
        "policy_improvement_mode": "certificate_constrained",
        "policy_improvement_score_normalization": str(score_normalization),
        "policy_improvement_score_transform": str(score_transform),
        "policy_improvement_guard_mode": str(guard_mode),
        "policy_improvement_pairwise_prefix_samples": int(
            pairwise_prefix_samples),
        "policy_improvement_pairwise_error_multiplier": float(
            pairwise_error_multiplier),
    }


def variant_profiles(args):
    common = {
        **closure.PROFILE,
        "source_records_per_domain": int(args.source_records_per_domain),
        "exact_mc_samples": int(args.exact_mc_samples),
        "exact_sampling_mode": str(args.exact_sampling_mode),
        "exact_jobs": int(args.exact_jobs),
        "evaluate_or_replicate_baseline_new_action_count": 4,
        "policy_improvement_mc_error_bound": float(args.risk_eta),
        "policy_improvement_certificate_mc_error_bound": float(
            args.certificate_eta),
        "policy_improvement_confirmation_samples": int(
            args.confirmation_samples),
        "policy_improvement_confirmation_batch_samples": int(
            args.confirmation_batch_samples),
        "policy_improvement_confirmation_delta": float(
            args.confirmation_delta),
        "policy_improvement_confirmation_jobs": int(
            args.confirmation_jobs),
        "policy_improvement_confirmation_lambda_min": float(
            args.confirmation_lambda_min),
        "policy_improvement_confirmation_lambda_count": int(
            args.confirmation_lambda_count),
        "policy_improvement_rollout_depth": 1,
        "policy_improvement_rollout_max_arms": 0,
        "policy_improvement_rollout_mc_samples": 0,
        "policy_improvement_rollout_mc_error_bound": 0.0,
    }
    return {
        CONTROL: {
            **common,
            "implementation_contract_id": (
                "promoted_v51_observed_terminal_closure"),
            "theory_contract_id": "v51_statistical_closure_v2",
            "evaluate_or_replicate_new_action_count": 4,
            "evaluate_or_replicate_new_action_policy": (
                "canonical_plus_posterior_risk"),
            "policy_improvement_mode": "off",
        },
        V52: {
            **common,
            "implementation_contract_id": "v52_safeguarded_policy_improvement",
            "theory_contract_id": "v52_safeguarded_closure_v1",
            "evaluate_or_replicate_new_action_count": 6,
            "evaluate_or_replicate_new_action_policy": (
                "canonical_plus_posterior_risk_certificate_coverage"),
            "policy_improvement_mode": "action_superset",
        },
        V53: _v53_profile(
            common,
            args.exact_mc_samples,
            args.score_normalization,
            args.score_transform,
            args.challenger_new_action_count,
            args.guard_mode,
            args.pairwise_prefix_samples,
            args.pairwise_error_multiplier,
            args.challenger_new_action_policy,
        ),
        "v53_mc8": _v53_profile(
            common, 8, args.score_normalization, args.score_transform,
            args.challenger_new_action_count, args.guard_mode,
            args.pairwise_prefix_samples, args.pairwise_error_multiplier,
            args.challenger_new_action_policy),
        "v53_mc32": _v53_profile(
            common, 32, args.score_normalization, args.score_transform,
            args.challenger_new_action_count, args.guard_mode,
            args.pairwise_prefix_samples, args.pairwise_error_multiplier,
            args.challenger_new_action_policy),
        HIGH_FIDELITY: _v53_profile(
            common, 128, args.score_normalization, args.score_transform,
            args.challenger_new_action_count, args.guard_mode,
            args.pairwise_prefix_samples, args.pairwise_error_multiplier,
            args.challenger_new_action_policy),
        "v54_mc128": _v53_profile(
            common, 128, args.score_normalization, args.score_transform,
            args.challenger_new_action_count, args.guard_mode,
            args.pairwise_prefix_samples, args.pairwise_error_multiplier,
            args.challenger_new_action_policy),
        "v54_mc512": _v53_profile(
            common, 512, args.score_normalization, args.score_transform,
            args.challenger_new_action_count, args.guard_mode,
            args.pairwise_prefix_samples, args.pairwise_error_multiplier,
            args.challenger_new_action_policy),
        "v55_mc128": _v53_profile(
            common, 128, args.score_normalization, args.score_transform,
            args.challenger_new_action_count, args.guard_mode,
            args.pairwise_prefix_samples, args.pairwise_error_multiplier,
            args.challenger_new_action_policy),
        "v55_mc512": _v53_profile(
            common, 512, args.score_normalization, args.score_transform,
            args.challenger_new_action_count, args.guard_mode,
            args.pairwise_prefix_samples, args.pairwise_error_multiplier,
            args.challenger_new_action_policy),
        "v56_confirm2048": {
            **_v53_profile(
                common, 512, "none", "bounded_current_gain",
                args.challenger_new_action_count,
                "independent_confirmation",
                args.pairwise_prefix_samples,
                args.pairwise_error_multiplier,
                args.challenger_new_action_policy,
            ),
            "policy_improvement_confirmation_samples": 2048,
        },
        "v56_confirm4096": {
            **_v53_profile(
                common, 512, "none", "bounded_current_gain",
                args.challenger_new_action_count,
                "independent_confirmation",
                args.pairwise_prefix_samples,
                args.pairwise_error_multiplier,
                args.challenger_new_action_policy,
            ),
            "policy_improvement_confirmation_samples": 4096,
        },
        V57: {
            **_v53_profile(
                common, 512, "none", "bounded_current_gain",
                args.challenger_new_action_count,
                "independent_confirmation",
                args.pairwise_prefix_samples,
                args.pairwise_error_multiplier,
                args.challenger_new_action_policy,
            ),
            "implementation_contract_id": (
                "v57_posterior_safe_terminal_closure"),
            "theory_contract_id": (
                "v57_confirmation_dominance_composition_v1"),
            "policy_improvement_confirmation_samples": 4096,
            "posterior_dominance_enabled": True,
            "posterior_dominance_delta": (
                0.05 / max(int(args.N) - int(args.n0), 1)
            ),
            "posterior_dominance_min_mean_gain": 0.0,
            "posterior_dominance_initialization": "risk",
        },
        V58: {
            **_v53_profile(
                common, 512, "none", "bounded_current_gain",
                args.challenger_new_action_count,
                "independent_confirmation",
                args.pairwise_prefix_samples,
                args.pairwise_error_multiplier,
                "canonical_plus_posterior_guard_decomposition",
            ),
            "implementation_contract_id": (
                "v58_guard_decomposed_action_support"),
            "theory_contract_id": (
                "v58_guard_decomposed_policy_improvement_v1"),
            "policy_improvement_confirmation_samples": 4096,
            "posterior_dominance_enabled": False,
        },
        V59: {
            **common,
            "implementation_contract_id": (
                "v59_frozen_policy_gaussian_verification"),
            "theory_contract_id": (
                "v59_gaussian_replication_certificate_v1"),
            "evaluate_or_replicate_new_action_count": 4,
            "evaluate_or_replicate_new_action_policy": (
                "canonical_plus_posterior_risk"),
            "policy_improvement_mode": "off",
            "posterior_dominance_enabled": False,
            "terminal_verification_budget": int(
                args.terminal_verification_budget),
            "terminal_verification_delta": float(
                args.terminal_verification_delta),
            "terminal_verification_mean_delta_fraction": float(
                args.terminal_verification_mean_delta_fraction),
        },
        V60: {
            **common,
            "implementation_contract_id": (
                "v60_frozen_ordered_shortlist_verification"),
            "theory_contract_id": (
                "v60_familywise_gaussian_shortlist_certificate_v1"),
            "evaluate_or_replicate_new_action_count": 4,
            "evaluate_or_replicate_new_action_policy": (
                "canonical_plus_posterior_risk"),
            "policy_improvement_mode": "off",
            "posterior_dominance_enabled": False,
            "terminal_verification_budget": int(
                args.terminal_verification_budget),
            "terminal_verification_delta": float(
                args.terminal_verification_delta),
            "terminal_verification_mean_delta_fraction": float(
                args.terminal_verification_mean_delta_fraction),
            "terminal_verification_policy": (
                "ordered_frozen_shortlist"),
            "terminal_verification_shortlist_size": int(
                args.terminal_verification_shortlist_size),
        },
    }


def build_specs(args):
    profiles = variant_profiles(args)
    requested = [
        value.strip() for value in str(args.variants).split(",")
        if value.strip()
    ]
    unknown = sorted(set(requested) - set(profiles))
    if unknown:
        raise ValueError(f"unknown V53 variants: {unknown}")
    if (
        any(name in V54_FIDELITY for name in requested)
        and str(args.guard_mode) != "paired_nested_difference"
    ):
        raise ValueError(
            "v54_mc128/v54_mc512 require paired_nested_difference")
    if (
        any(name in V55_FIDELITY for name in requested)
        and str(args.guard_mode) != "paired_nested_absolute"
    ):
        raise ValueError(
            "v55_mc128/v55_mc512 require paired_nested_absolute")
    if (
        any(name in (*V56_FIDELITY, V57, V58) for name in requested)
        and str(args.guard_mode) != "independent_confirmation"
    ):
        raise ValueError(
            "v56/v57/v58 confirmation variants require "
            "independent_confirmation")
    if any(name in {V59, V60} for name in requested):
        if int(args.terminal_verification_budget) < 2:
            raise ValueError(
                "terminal verification requires at least two calls")
        if not 0.0 < float(args.terminal_verification_delta) < 1.0:
            raise ValueError(
                "terminal verification delta must lie in (0, 1)")
        fraction = float(args.terminal_verification_mean_delta_fraction)
        if not 0.0 < fraction < 1.0:
            raise ValueError(
                "verification mean-delta fraction must lie in (0, 1)")
    if (
        V60 in requested
        and int(args.terminal_verification_shortlist_size) < 2
    ):
        raise ValueError("V60 requires a shortlist of at least two policies")
    args.variant_profiles = {name: profiles[name] for name in requested}
    args.variants = ",".join(requested)
    args.scenarios = closure.promoted.v51.v50.v49.v27.MEAN_SCENARIOS
    if requested == [V59]:
        args.stage_family = "v59_terminal_gaussian_verification"
        args.gate_label = "V59 frozen-policy Gaussian verification"
    elif requested == [V60]:
        args.stage_family = "v60_ordered_shortlist_verification"
        args.gate_label = "V60 frozen ordered-shortlist verification"
    else:
        args.stage_family = "v53_constrained_certificate_deficit"
        args.gate_label = "V53 constrained certificate-deficit policy"
    args.sequential = bool(int(args.N) > int(args.n0))
    return closure._root_module().build_specs(args)


def main():
    defaults = closure._root_module()
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=defaults.DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=defaults.DEFAULT_DEPLOY)
    parser.add_argument("--python", type=Path, default=defaults.REMOTE_PYTHON)
    parser.add_argument("--manifest", type=Path, default=(
        defaults.DEFAULT_DEPLOY
        / "SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json"))
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument(
        "--local-design-prerequisite",
        action="store_false",
        dest="remote_design_only",
    )
    parser.set_defaults(remote_design_only=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--run-id", default=(
        "scolh_v53_constrained_certificate_s5_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--nodes", default=",".join(CPU_NODES))
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--source-records-per-domain", type=int, default=64)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=512)
    parser.add_argument("--variance-audit-size", type=int, default=512)
    parser.add_argument("--misspecification-prior-df", type=float, default=4.0)
    parser.add_argument("--misspecification-ridge", type=float, default=1.0)
    parser.add_argument("--misspecification-max-scale", type=float, default=100.0)
    parser.add_argument("--misspecification-delta", type=float, default=0.05)
    parser.add_argument("--contrast-scale", type=float, default=1.0)
    parser.add_argument("--null-geometry-ridge", type=float, default=1e-3)
    parser.add_argument(
        "--variants",
        default=",".join((CONTROL, V52, V53)),
    )
    parser.add_argument("--exact-mc-samples", type=int, default=8)
    parser.add_argument(
        "--challenger-new-action-count", type=int, default=6)
    parser.add_argument(
        "--challenger-new-action-policy",
        choices=(
            "canonical_plus_posterior_risk_certificate_coverage",
            "canonical_plus_posterior_pareto_support",
            "canonical_plus_posterior_guard_decomposition",
        ),
        default="canonical_plus_posterior_risk_certificate_coverage",
    )
    parser.add_argument(
        "--exact-sampling-mode", default="factorized_rqmc_nested")
    parser.add_argument(
        "--score-normalization",
        choices=("none", "current_terminal"),
        default="current_terminal",
    )
    parser.add_argument(
        "--score-transform",
        choices=("identity", "bounded_current_gain"),
        default="identity",
    )
    parser.add_argument(
        "--guard-mode",
        choices=(
            "uniform_score",
            "paired_nested_difference",
            "paired_nested_absolute",
            "independent_confirmation",
        ),
        default="uniform_score",
    )
    parser.add_argument("--pairwise-prefix-samples", type=int, default=32)
    parser.add_argument(
        "--pairwise-error-multiplier", type=float, default=1.25)
    parser.add_argument("--confirmation-samples", type=int, default=4096)
    parser.add_argument(
        "--confirmation-batch-samples", type=int, default=512)
    parser.add_argument("--confirmation-delta", type=float, default=0.05)
    parser.add_argument("--confirmation-jobs", type=int, default=0)
    parser.add_argument(
        "--confirmation-lambda-min", type=float, default=0.001)
    parser.add_argument(
        "--confirmation-lambda-count", type=int, default=24)
    parser.add_argument("--risk-eta", type=float, default=0.0)
    parser.add_argument("--certificate-eta", type=float, default=0.0)
    parser.add_argument(
        "--terminal-verification-budget", type=int, default=48)
    parser.add_argument(
        "--terminal-verification-delta", type=float, default=0.05)
    parser.add_argument(
        "--terminal-verification-mean-delta-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--terminal-verification-shortlist-size", type=int, default=2)
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--exact-jobs", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=8192)
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if int(args.exact_jobs) <= 0:
        parser.error("--exact-jobs must be positive")
    if int(args.exact_jobs) > int(args.cpu):
        parser.error("--exact-jobs cannot exceed reserved --cpu")
    if int(args.confirmation_jobs) < 0:
        parser.error("--confirmation-jobs cannot be negative")
    if (
        int(args.confirmation_jobs) > 0
        and int(args.confirmation_jobs) > int(args.cpu)
    ):
        parser.error("--confirmation-jobs cannot exceed reserved --cpu")
    specs = build_specs(args)
    if args.dry_run:
        print(json.dumps(specs, indent=2))
        return
    coverage = validate_frozen_design_seed_coverage(args, specs)
    print(json.dumps({"validated_frozen_design_seeds": coverage}, indent=2))
    if not args.no_sync:
        subprocess.run([str(defaults.SYNC)], check=True, cwd=ROOT)
    payload = "\n".join(json.dumps(spec) for spec in specs) + "\n"
    subprocess.run(
        [
            sys.executable,
            str(args.scheduler),
            "submit-jsonl", "--stdin", "--trusted", "--json",
            "--intent-label", str(args.run_id),
        ],
        input=payload,
        text=True,
        check=True,
    )


if __name__ == "__main__":
    main()
