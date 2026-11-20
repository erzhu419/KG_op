#!/usr/bin/env python3
"""P0-6 Automated Contract, Simulator, Verifier & Manifest Metadata Exporter.

Single source of truth exporting code-verified dimensions, control channel counts,
OPSD market lists, Ingolstadt phase counts, terminal verification slot parameters
`(K_0, K_1, K_2) = (112, 128, 140)`, `delta = (0.025, 0.015, 0.010)`, critical
violation counts `k* = (1, 1, 1)`, and manifest task/seed/budget counts to JSON and
LaTeX tables so manuscript text, Markdown reports, and code can never drift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

SC_ROOT = Path(__file__).resolve().parents[2]
if str(SC_ROOT) not in sys.path:
    sys.path.insert(0, str(SC_ROOT))

from v2_or.dispatch.campaign_runner import build_family_tasks, compute_v2_or_code_sha256
from v2_or.priors.debiased_spectral_prior import DebiasedSpectralMetaPrior, _build_raw_dictionary
from v2_or.verification.gated_verifier_v2 import exact_binomial_critical_k

import numpy as np


def collect_code_and_manifest_metadata(manifest_relpaths: list[str]) -> dict[str, Any]:
    dummy_psi = np.zeros((2, 12), dtype=float)
    raw_dict_dim = int(_build_raw_dictionary(dummy_psi).shape[1])
    default_prior = DebiasedSpectralMetaPrior()

    slot_budgets = (112, 128, 140)
    slot_deltas = (0.025, 0.015, 0.010)
    alpha_default = 0.05
    slot_k_crit = tuple(exact_binomial_critical_k(k, alpha_default, d) for k, d in zip(slot_budgets, slot_deltas))

    verifier_spec = {
        "alpha_chance_tolerance": alpha_default,
        "familywise_delta": 0.05,
        "slot_roles": (
            "posterior_primary",
            "empirical_feasible_incumbent",
            "conservative_posterior_interior",
        ),
        "slot_budgets": slot_budgets,
        "slot_deltas": slot_deltas,
        "slot_critical_violations_at_alpha_005": slot_k_crit,
        "max_verification_calls": sum(slot_budgets),
        "abstention_deployed_policy": None,
    }

    prior_spec = {
        "psi_dim": 12,
        "raw_dictionary_size": raw_dict_dim,
        "lowpass_roughness_quantile": default_prior.lowpass_roughness_quantile,
        "active_rank": default_prior.active_rank,
        "max_occupancy_weight_cap": default_prior.max_weight_cap,
        "dct_modes": default_prior.dct_modes,
    }

    manifests_meta: dict[str, Any] = {}
    for rel in manifest_relpaths:
        mp = SC_ROOT / rel
        if not mp.is_file():
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        fams = {}
        for fam in m["families"]:
            entry: dict[str, Any] = {"kind": fam["kind"]}
            if "reuse_archive_from_family" not in fam and "source_config" in fam:
                s_tasks = build_family_tasks(fam, "source")
                entry["n_source_tasks"] = len(s_tasks)
                entry["source_dims"] = sorted({int(t.d) for t in s_tasks})
                entry["source_channels"] = int(s_tasks[0].num_control_channels)
                entry["source_psi_dim"] = int(s_tasks[0].nominal_exposure(s_tasks[0].domain_heuristic_anchor()).psi.size)
                if fam["kind"] == "opsd":
                    entry["source_markets"] = list(fam["source_config"]["markets"])
                src_proto = fam.get("source_archive_protocol", {})
                n_solvers = len(src_proto.get("solvers", []))
                entry["n_solver_trajectories"] = len(s_tasks) * n_solvers
                entry["per_solver_budget_calls"] = int(src_proto.get("per_solver_budget", 0))
            if "target_config" in fam and fam.get("online_step_budgets"):
                if fam["kind"] == "traffic_live_sumo":
                    # Avoid starting libsumo just for static dimension inspection
                    from v2_or.simulators.traffic_signal_des import _load_ingolstadt_phase_spec
                    lo, _, _, _, _ = _load_ingolstadt_phase_spec(max_tls=None)
                    entry["n_target_tasks"] = int(fam["target_config"]["n_tasks"])
                    entry["target_dims"] = [int(lo.size)]
                    entry["target_channels"] = 2
                    entry["target_psi_dim"] = 12
                else:
                    t_tasks = build_family_tasks(fam, "target")
                    entry["n_target_tasks"] = len(t_tasks)
                    entry["target_dims"] = sorted({int(t.d) for t in t_tasks})
                    entry["target_channels"] = int(t_tasks[0].num_control_channels)
                    entry["target_psi_dim"] = int(t_tasks[0].nominal_exposure(t_tasks[0].domain_heuristic_anchor()).psi.size)
                    if fam["kind"] == "opsd":
                        entry["target_markets"] = list(fam["target_config"]["markets"])
                        entry["opsd_buffer_hours"] = int(t_tasks[0].buffer_hours)
                        entry["opsd_clamp_limit"] = t_tasks[0].clamp_limit
                        entry["opsd_mean_clamped_hour_pct"] = float(
                            100.0 * np.mean([t.tail_clamped_fraction for t in t_tasks])
                        )
                entry["online_step_budgets"] = list(fam["online_step_budgets"])
                entry["n_rep_seeds"] = len(fam["target_eval_seeds"])
                entry["n0_init"] = int(fam.get("n0_init", 8))
                entry["batch_reps"] = int(fam.get("batch_reps", 2))
                entry["audit_replications"] = int(fam.get("audit_replications", 4000))
                entry["arms"] = [a["name"] for a in fam.get("arms", [])]
            fams[fam["name"]] = entry
        manifests_meta[m["campaign_name"]] = {
            "manifest_relpath": rel,
            "sealed_complete": bool(m.get("sealed_complete", False)),
            "families": fams,
        }

    return {
        "v2_or_code_sha256": compute_v2_or_code_sha256(),
        "verifier_contract": verifier_spec,
        "prior_contract": prior_spec,
        "manifests": manifests_meta,
    }


def render_latex_macros(meta: dict[str, Any]) -> str:
    vc = meta["verifier_contract"]
    pc = meta["prior_contract"]
    b0, b1, b2 = vc["slot_budgets"]
    d0, d1, d2 = vc["slot_deltas"]
    k0, k1, k2 = vc["slot_critical_violations_at_alpha_005"]
    return "\n".join([
        "% Auto-generated from Python contracts by `export_contract_metadata.py` (DO NOT EDIT MANUALLY)",
        f"\\newcommand{{\\PsiDim}}{{{pc['psi_dim']}}}",
        f"\\newcommand{{\\RawDictDim}}{{{pc['raw_dictionary_size']}}}",
        f"\\newcommand{{\\ActiveRank}}{{{pc['active_rank']}}}",
        f"\\newcommand{{\\SlotZeroK}}{{{b0}}}",
        f"\\newcommand{{\\SlotOneK}}{{{b1}}}",
        f"\\newcommand{{\\SlotTwoK}}{{{b2}}}",
        f"\\newcommand{{\\MaxVerifCalls}}{{{vc['max_verification_calls']}}}",
        f"\\newcommand{{\\SlotZeroDelta}}{{{d0:.3f}}}",
        f"\\newcommand{{\\SlotOneDelta}}{{{d1:.3f}}}",
        f"\\newcommand{{\\SlotTwoDelta}}{{{d2:.3f}}}",
        f"\\newcommand{{\\SlotCritK}}{{({k0}, {k1}, {k2})}}",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifests",
        nargs="+",
        default=[
            "v2_or/manifests/v2_or_confirm_v1.json",
            "v2_or/manifests/v2_or_p1_mech_v1.json",
        ],
    )
    parser.add_argument("--out-json", default="paper_artifacts/v2_or_contract_metadata.json")
    parser.add_argument("--out-tex-macros", default="manuscript_v2_or/macros_autogen.tex")
    args = parser.parse_args()

    meta = collect_code_and_manifest_metadata(args.manifests)
    out_j = SC_ROOT / args.out_json
    out_j.parent.mkdir(parents=True, exist_ok=True)
    out_j.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    out_m = SC_ROOT / args.out_tex_macros
    out_m.parent.mkdir(parents=True, exist_ok=True)
    out_m.write_text(render_latex_macros(meta), encoding="utf-8")
    print(f"Exported verified metadata to {out_j} and {out_m}")


if __name__ == "__main__":
    main()
