"""Evaluate source-task deletion on fixed target-policy pairs without simulation."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import pickle
import time
import traceback

import numpy as np

from v2_or.dispatch.campaign_runner import build_family_tasks
from v2_or.legacy_solvers.solvers import load_legacy_archive_json
from v2_or.models.tempered_hvd_posterior import TemperedHierarchicalHVDModel
from v2_or.priors.debiased_spectral_prior import DebiasedSpectralMetaPrior


FULL = "HistoryFull-GateKG"
DIAGONAL = "HistoryDiag-GateKG"
HEADS = ("obj", "con")
TIE_TOL = 1e-10


def read_json(path):
    return json.loads(path.read_text())


def fit_deletions(archive_dir, options):
    records, metadata = load_legacy_archive_json(archive_dir / "archive.json")
    task_ids = sorted({r.task_id for r in records})
    if len(task_ids) != 6:
        raise ValueError(f"Expected the sealed six-task archive, got {task_ids}")
    with (archive_dir / "prior_default.pkl").open("rb") as stream:
        original = pickle.load(stream)
    full_refit = DebiasedSpectralMetaPrior(**options).fit(records)
    models = {"original": original}
    for source in task_ids:
        remaining = [row for row in records if row.task_id != source]
        models[source] = DebiasedSpectralMetaPrior(**options).fit(remaining)
    return models, full_refit, metadata


def read_shortlists(target_dir, family):
    shortlists = {}
    for path in sorted(target_dir.glob(f"sv_{family['name']}_*.json")):
        row = read_json(path)
        key = (row["target_task_idx"], row["rep_seed"], row["method_name"])
        shortlists[key] = row["primary_policy"], row["shortlist_policies"]
    wanted = set(itertools.product(range(family["target_config"]["n_tasks"]), family["target_eval_seeds"],
                                   [a["name"] for a in family["arms"]]))
    if shortlists.keys() != wanted:
        raise ValueError(f"Sealed search records do not match the registered panel: {len(shortlists)} vs {len(wanted)}")
    return shortlists


def fixed_menu(task, task_index, seeds, arms, shortlists):
    """Retain only physical candidates, with the status quo at index zero."""
    vectors = []
    index = {}

    def add(value):
        x = np.asarray(value, dtype=float)
        key = tuple(x.tolist())
        if key not in index:
            index[key] = len(vectors)
            vectors.append(x)
        return index[key]

    add(task.domain_heuristic_anchor())
    for seed, arm in itertools.product(seeds, arms):
        primary, slate = shortlists[task_index, seed, arm]
        if not slate or slate[0] != primary:
            raise ValueError("Receipt primary differs from first slate policy")
        for policy in slate:
            add(policy)

    key_pairs = []
    for seed in seeds:
        a, b = (add(shortlists[task_index, seed, arm][0]) for arm in (FULL, DIAGONAL))
        if a != b:
            key_pairs.append((a, b))
    pair_sets = {
        "all": np.array(list(itertools.combinations(range(len(vectors)), 2)), dtype=int).reshape(-1, 2),
        "full_vs_diagonal_primary": np.array(key_pairs, dtype=int).reshape(-1, 2),
    }
    exposures = [task.nominal_exposure(x) for x in vectors]
    return vectors, exposures, pair_sets


def coefficient_moments(prior, exposures, head):
    phi, _, u_hvd, _, _ = prior.transform_exposures(exposures)
    mu = phi @ getattr(prior, f"mu_0_{head}_")
    covariance = getattr(prior, f"cov_0_{head}_")
    return phi, mu, covariance, u_hvd


def reference_anchor(prior, exposures, head, batch_reps):
    phi, mu, covariance, u_hvd = coefficient_moments(prior, exposures, head)
    noise = max(float(u_hvd[0] @ getattr(prior, f"gamma_0_{head}_")), 1e-10) / batch_reps
    sd = np.sqrt(phi[0] @ covariance @ phi[0] + noise)
    return noise, mu[0] - sd, mu[0] + sd


def prior_scenarios(prior, exposures, head, reference):
    phi, mu, covariance, _ = coefficient_moments(prior, exposures, head)
    noise, y_lower, y_upper = reference
    slope = phi @ covariance @ phi[0] / (phi[0] @ covariance @ phi[0] + noise)
    diagonal = np.diag(covariance)
    slope_diagonal = phi @ (diagonal * phi[0]) / (phi[0] @ (diagonal * phi[0]) + noise)
    return {
        "mean": mu,
        "anchor_low": mu + slope * (y_lower - mu[0]),
        "anchor_high": mu + slope * (y_upper - mu[0]),
        "offdiagonal_shift_anchor_high": (slope - slope_diagonal) * (y_upper - mu[0]),
    }, phi, covariance


def posterior_prediction(prior, vectors, exposures, records, kind):
    model = TemperedHierarchicalHVDModel(
        prior, mode="matched_coefficients", use_source_basis=True,
        use_source_hvd_prior=False, use_empirical_r_perp=False,
        coefficient_prior=kind, n0_freeze=len(records),
    )
    for record in records:
        model.observe(record)
    model.freeze_matched_scales(records)
    return model.predict_joint(vectors, exposures)


def sign(values):
    return np.where(np.abs(values) <= TIE_TOL, 0, np.sign(values))


def aggregate(values_by_model):
    """Keep the original denominator, counting ties separately from reversals."""
    base = np.concatenate(values_by_model["original"])
    base_sign = sign(base)
    if not base.size:
        raise ValueError("A registered pair set is empty")
    distinct = base_sign != 0
    stable = distinct.copy()
    reversed_once = np.zeros(base.size, dtype=bool)
    folds = {}
    for source, blocks in values_by_model.items():
        if source == "original":
            continue
        deletion = np.concatenate(blocks)
        deletion_sign = sign(deletion)
        reversed_here = (base_sign * deletion_sign) == -1
        reversed_once |= reversed_here
        stable &= base_sign == deletion_sign
        absolute_change = np.abs(deletion - base)
        folds[source] = {
            "strict_reversals": int(reversed_here.sum()),
            "original_order_became_tie": int(np.count_nonzero(distinct & (deletion_sign == 0))),
            "absolute_gap_change_median": float(np.median(absolute_change)),
            "absolute_gap_change_p90": float(np.quantile(absolute_change, 0.9)),
        }
    return {
        "pairs": int(base.size), "original_strict_pairs": int(distinct.sum()),
        "stable_strict_all_six": int(stable.sum()), "reversed_under_any_deletion": int(reversed_once.sum()),
        "absolute_original_gap_median": float(np.median(np.abs(base))), "deletions": folds,
    }


def coupling_order_changes(full, diagonal):
    signs_full = {source: sign(np.concatenate(values)) for source, values in full.items()}
    signs_diag = {source: sign(np.concatenate(values)) for source, values in diagonal.items()}
    original_flip = signs_full["original"] * signs_diag["original"] < 0
    preserved = original_flip.copy()
    for source in signs_full:
        preserved &= (signs_diag[source] == signs_diag["original"]) & (signs_full[source] == signs_full["original"])
    return {"pairs": int(original_flip.size), "original_coupling_order_changes": int(original_flip.sum()),
            "both_orders_preserved_through_all_six": int(preserved.sum())}


def prior_values(models, full_refit, vectors, exposures, batch_reps):
    values = {}
    spreads = {}
    for head in HEADS:
        reference = reference_anchor(models["original"], exposures, head, batch_reps)
        for source, model in models.items():
            scenarios, phi, cov = prior_scenarios(model, exposures, head, reference)
            values.setdefault(head, {})[source] = scenarios
            spreads.setdefault(head, {})[source] = (phi, cov)
        checked, _, _ = prior_scenarios(full_refit, exposures, head, reference)
        for scenario in checked:
            np.testing.assert_allclose(checked[scenario], values[head]["original"][scenario], rtol=1e-7, atol=1e-8)
    return values, spreads


def posterior_values(models, vectors, exposures, history):
    values = {head: {} for head in HEADS}
    spreads = {head: {} for head in HEADS}
    for source, prior in models.items():
        full = posterior_prediction(prior, vectors, exposures, history, "historical_full")
        diag = posterior_prediction(prior, vectors, exposures, history, "historical_diagonal")
        for head in HEADS:
            mean = getattr(full, f"mean_{head}")
            mean_diag = getattr(diag, f"mean_{head}")
            values[head][source] = {"full": mean, "diagonal": mean_diag, "offdiagonal_shift": mean - mean_diag}
            cov = getattr(full, f"cov_{head}")
            spreads[head][source] = (None, cov)
    return values, spreads


def compare_slate_gaps(family, target_dir, models, full_refit=None, prefix_dir=None):
    shortlists = read_shortlists(target_dir, family)
    arms = [a["name"] for a in family["arms"]]
    seeds = family["target_eval_seeds"]
    if prefix_dir is not None:
        seeds = seeds[:1]
    comparisons = {}
    uncertainty = {}
    examined = []
    for task_index, task in enumerate(build_family_tasks(family, "target")):
        if prefix_dir is not None:
            prefix = prefix_dir / f"{family['name']}_t{task_index:02d}.pkl"
            if not prefix.is_file():
                continue
            with prefix.open("rb") as stream:
                restored = pickle.load(stream)
            if restored["task_index"] != task_index or restored["family"]["name"] != family["name"]:
                raise ValueError(f"Wrong restored history: {prefix.name}")
            history = restored["records"]
            if len(history) != family["n0_init"] or any(r.n_reps != family["batch_reps"] for r in history):
                raise ValueError(f"Invalid restored initialization: {prefix.name}")
        vectors, exposures, pair_sets = fixed_menu(task, task_index, seeds, arms, shortlists)
        examined.append((task_index, len(vectors)))
        if prefix_dir is None:
            values, spreads = prior_values(models, full_refit, vectors, exposures, family["batch_reps"])
        else:
            values, spreads = posterior_values(models, vectors, exposures, history)
        for pair_set, pairs in pair_sets.items():
            left, right = pairs.T
            for head in HEADS:
                for source, scenarios in values[head].items():
                    for scenario, predictions in scenarios.items():
                        key = f"{pair_set}/{head}/{scenario}"
                        comparisons.setdefault(key, {}).setdefault(source, []).append(predictions[left] - predictions[right])
                    phi, cov = spreads[head][source]
                    if phi is None:
                        var = np.diag(cov)[left] + np.diag(cov)[right] - 2 * cov[left, right]
                    else:
                        difference = phi[left] - phi[right]
                        var = np.sum((difference @ cov) * difference, axis=1)
                    key = f"{pair_set}/{head}"
                    uncertainty.setdefault(key, {}).setdefault(source, []).append(np.sqrt(np.maximum(var, 0.0)))
        if any([task.search_calls, task.verification_calls, task.audit_calls]):
            raise RuntimeError("A frozen menu unexpectedly incurred target simulator calls")

    if len(examined) != (family["target_config"]["n_tasks"] if prefix_dir is None else 12):
        raise ValueError(f"Incomplete target roster for {family['name']}: {examined}")
    sd = {}
    for key, blocks_by_source in uncertainty.items():
        base = np.concatenate(blocks_by_source["original"])
        sd[key] = {source: float(np.median(np.abs(np.concatenate(blocks) - base)))
                   for source, blocks in blocks_by_source.items() if source != "original"}
    result = {
        "tasks_and_menu_sizes": examined,
        "registered_primary_pairs": len(examined) * len(seeds),
        "gap_comparisons": {key: aggregate(v) for key, v in comparisons.items()},
        "sd_gap_median_absolute_changes": sd,
    }
    if prefix_dir is not None:
        result["covariance_changes_ranking"] = {
            key.removesuffix("/full"): coupling_order_changes(blocks, comparisons[key.removesuffix("/full") + "/diagonal"])
            for key, blocks in comparisons.items() if key.endswith("/full")
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--prefix-dir", type=Path)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    target_dir = Path(manifest["output_dir"]) / "stages/search_and_verify/items"
    source_root = Path("paper_artifacts") / manifest["reuse_source_campaign"] / "source_bundles"
    args.output.mkdir(parents=True, exist_ok=True)
    for family in manifest["families"][args.start:args.end]:
        name = family["name"]
        mode = "prior" if args.prefix_dir is None else "posterior"
        cached = args.output / f"fitted_{name}.pkl"
        path = args.output / f"{mode}_{name}_attempt{args.attempt}.json"
        if path.exists():
            raise FileExistsError(f"Preserve previous attempt: {path}")
        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        result = {"mode": mode, "family": name, "attempt": args.attempt, "simulator_calls": 0}
        try:
            if args.prefix_dir is None:
                models, refit, metadata = fit_deletions(source_root / name, family["prior_variants"]["default"])
                with cached.open("wb") as stream:
                    pickle.dump(models, stream)
                result.update(source_refits_new=7, source_logs=metadata)
            else:
                with cached.open("rb") as stream:
                    models = pickle.load(stream)
                refit = None
                result["source_refits_new"] = 0
            result.update(compare_slate_gaps(family, target_dir, models, full_refit=refit, prefix_dir=args.prefix_dir))
            result["status"] = "complete"
        except Exception:
            result.update(status="failed", error=traceback.format_exc())
            raise
        finally:
            result["cpu_seconds"] = time.process_time() - cpu_start
            result["wall_seconds"] = time.perf_counter() - wall_start
            path.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({key: result[key] for key in ("family", "mode", "status", "cpu_seconds", "wall_seconds")}), flush=True)


if __name__ == "__main__":
    main()
