"""A16: saved source covariance/selection associations, no target evaluation."""
import gzip
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.source_head_association import allocation_alignment, selection_stability, source_archives

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_source_head_association_a16_20261003"


def summarize(values, indices):
    values = np.array(values)
    return {"mean": float(values.mean()),
            "paired_seed_bootstrap95": np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist(),
            "positive_seeds": int(np.sum(values > 1e-12)), "negative_seeds": int(np.sum(values < -1e-12)),
            "seed_values": values.tolist()}


def main():
    start, wall_start = time.process_time(), time.monotonic()
    spec = json.loads((ROOT / "performance/meta_prior_source_head_association_audit_plan.json").read_text())
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def stop(signum, frame):
        raise TimeoutError("A16 local CPU cap reached")
    signal.signal(signal.SIGXCPU, stop)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    result = {"stage": spec["stage"], "complete": False, "rows": [],
              "scope": spec["separation"], **spec["limits"]}
    try:
        with gzip.open(ROOT / spec["inputs"][0], "rt") as stream:
            archives, reuse = source_archives(json.load(stream)["records"])
        old = json.loads((ROOT / spec["inputs"][1]).read_text())
        families = json.loads((ROOT / spec["inputs"][2]).read_text())["public_families"]
        seeds, groups = spec["cohort"]["seeds"], spec["cohort"]["archive_groups"]
        expected = {(s, g) for s in seeds for g in groups}
        if set(archives) != expected or not old["source_complete"]:
            raise ValueError("saved source archives are incomplete or outside A16 cohort")
        folds = {(f["seed"], f["archive_group"], f["held_task"]): f for f in old["folds"]}
        if len(folds) != len(old["folds"]) or set(folds) != {(s, g, h) for s, g in expected for h in range(8)}:
            raise ValueError("saved source selection holdouts differ from frozen cohort")
        public_indices = [j for indices in families.values() for j in indices]
        if sorted(public_indices) != list(range(68)):
            raise ValueError("public families no longer partition the 68-function dictionary")
        for s in seeds:
            for g in groups:
                info = archives[s, g]
                variance = np.asarray(info[2]["source_function_variance"])
                if variance.shape != (68,) or np.any(variance < 0) or not np.isfinite(variance).all() or variance.sum() <= 0:
                    raise ValueError(f"source risk covariance cannot be normalized: {(s, g)}")
                full = {h: info[i]["selected_indices"] for i, h in enumerate(("f", "g"))}
                full["fg_union"] = sorted(set(full["f"]) | set(full["g"]))
                held = {h: [folds[s, g, t]["fg_selected_indices"][i] for t in range(8)]
                        for i, h in enumerate(("f", "g"))}
                held["fg_union"] = [sorted(set(f) | set(gg)) for f, gg in zip(held["f"], held["g"])]
                if any(j not in range(68) for indices in full.values() for j in indices):
                    raise ValueError("saved function selection is outside the public dictionary")
                alignments = {h: allocation_alignment(variance, indices, families) for h, indices in full.items()}
                stability = {h: selection_stability(full[h], held[h]) for h in full}
                # The full eta fit is reused here: these are descriptions of
                # fg selection stability, never held-out eta-prediction scores.
                held_alignment = {h: [allocation_alignment(variance, f, families) for f in held[h]] for h in full}
                normalized = variance / variance.sum()
                result["rows"].append({"seed": s, "archive_group": g,
                    "source_reuse_count": reuse[s, g], "source_total_learned_eta_variance": float(variance.sum()),
                    "eta_family_variance_mass": {name: float(normalized[ids].sum()) for name, ids in families.items()},
                    "selected_indices": full, "alignment": alignments,
                    "fg_selection_stability": stability, "holdout_fg_alignment_with_full_eta": held_alignment})
        indices = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(len(seeds), size=(20000, len(seeds)))
        summaries = []
        lookup = {(r["seed"], r["archive_group"]): r for r in result["rows"]}
        for g in groups:
            for h in ("f", "g", "fg_union"):
                rows = [lookup[s, g] for s in seeds]
                summaries.append({"archive_group": g, "head": h, "seed_archives": len(rows),
                    **{metric: summarize([r["alignment"][h][metric] for r in rows], indices)
                       for metric in ("risk_variance_mass", "excess_over_uniform", "excess_over_family_matched")},
                    "mean_selected_functions": float(np.mean([r["alignment"][h]["selected_functions"] for r in rows])),
                    "mean_full_vs_holdout_jaccard": float(np.mean([r["fg_selection_stability"][h]["mean_full_vs_holdout_jaccard"] for r in rows])),
                    "mean_holdout_pair_jaccard": float(np.mean([r["fg_selection_stability"][h]["mean_holdout_pair_jaccard"] for r in rows]))})
        separation = [{"head": h, "contrast": "shared_minus_separate_heads",
            **{metric: summarize([lookup[s, "shared"]["alignment"][h][metric] -
                                  lookup[s, "separate_heads"]["alignment"][h][metric] for s in seeds], indices)
               for metric in ("risk_variance_mass", "excess_over_uniform", "excess_over_family_matched")}}
            for h in ("f", "g", "fg_union")]
        result.update(complete=True, groups=summaries, paired_archive_group_contrasts=separation,
                      source_archive_count=len(archives), source_holdout_count=len(folds),
                      reused_source_records=sum(reuse.values()), target_fields_used=[])
    except Exception as exc:
        result["failure"] = str(exc)
        raise
    finally:
        result.update(cpu_seconds=time.process_time() - start, wall_seconds=time.monotonic() - wall_start,
            current_process_vm_hwm_mb=float(next(line.split()[1] for line in
                Path("/proc/self/status").read_text().splitlines() if line.startswith("VmHWM:"))) / 1024)
        (ARTIFACTS / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: result[k] for k in ("complete", "cpu_seconds", "current_process_vm_hwm_mb")}), flush=True)


if __name__ == "__main__":
    main()
