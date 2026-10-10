"""Source-only alignment of selected mean functions and learned risk variance."""
import numpy as np


def source_archives(records):
    """Only access source fits and the metadata needed to identify their reuse."""
    archives, reuse = {}, {}
    for record in records:
        row = record["row"]
        if row["arm"] != "risk_ard_mixture":
            continue
        group = "separate_heads" if row["regime"] == "separate_head_structure" else "shared"
        key = row["seed"], group
        info = record["source_info"]
        if key in archives and archives[key] != info:
            raise ValueError(f"reused source archive has different fitted metadata: {key}")
        archives[key] = info
        reuse[key] = reuse.get(key, 0) + 1
    return archives, reuse


def allocation_alignment(variance, selected, families):
    """Control family composition before attributing overlap to exact functions."""
    mass = np.asarray(variance, dtype=float) / np.sum(variance)
    selected = sorted(set(selected))
    observed = float(mass[selected].sum())
    uniform = len(selected) / len(mass)
    matched = sum(len(set(selected).intersection(indices)) / len(indices) * mass[indices].sum()
                  for indices in families.values())
    return {"selected_functions": len(selected), "risk_variance_mass": observed,
            "uniform_reference_mass": uniform, "family_matched_reference_mass": float(matched),
            "excess_over_uniform": observed - uniform, "excess_over_family_matched": observed - float(matched)}


def selection_stability(full, folds):
    full = set(full)
    folds = [set(f) for f in folds]
    def jaccard(a, b):
        return len(a & b) / len(a | b) if a | b else 1.
    frequency = np.zeros(68)
    for fold in folds:
        frequency[list(fold)] += 1. / len(folds)
    return {"mean_full_vs_holdout_jaccard": float(np.mean([jaccard(full, f) for f in folds])),
            "mean_holdout_pair_jaccard": float(np.mean([jaccard(a, b) for i, a in enumerate(folds)
                                                        for b in folds[i + 1:]])),
            "mean_full_selection_retention": float(np.mean([len(full & f) / len(full) for f in folds])) if full else None,
            "source_selection_frequency": frequency.tolist()}
