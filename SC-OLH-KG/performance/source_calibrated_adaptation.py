"""Source-only predictive calibration of risk-adaptation prior odds."""
import numpy as np
from scipy.special import logsumexp, roots_legendre


def prior_adjusted_log_score(source, adapted, null_mass, q):
    """Change model prior odds; retain the earlier-data evidence in null_mass."""
    with np.errstate(divide="ignore"):
        a = np.log1p(-np.asarray(q)) + np.log(null_mass)
        b = np.log(q) + np.log1p(-np.asarray(null_mass))
    return np.logaddexp(a + source, b + adapted) - np.logaddexp(a, b)


def fit_source_calibration(forecasts, nodes=64):
    points, weights = roots_legendre(nodes)
    q = (points + 1.) / 2.
    source = np.array([f["source"] for f in forecasts])
    adapted = np.array([f["adapted"] for f in forecasts])
    null = np.array([f["null_posterior"] for f in forecasts])
    scores = prior_adjusted_log_score(source[:, None], adapted[:, None], null[:, None], q[None, :]).sum(axis=0)
    logs = scores + np.log(weights / 2.)
    masses = np.exp(logs - logsumexp(logs))
    mean = float(masses @ q)
    return {"q_mean": mean, "q_sd": float(np.sqrt(masses @ ((q - mean) ** 2))),
        "source_forecasts": len(forecasts), "source_cv_score_at_zero": float(source.sum()),
        "source_cv_score_at_half": float(prior_adjusted_log_score(source, adapted, null, .5).sum()),
        "source_cv_score_at_one": float(adapted.sum()), "quadrature_nodes": nodes}


def calibrate_existing_paths(source_rows, target_rows, nodes=64):
    groups = {}
    for row in source_rows:
        key = row["seed"], row["archive_group"], row["arm"]
        groups.setdefault(key, []).extend(row["forecasts"])
    fits = {key: fit_source_calibration(forecasts, nodes) for key, forecasts in groups.items()}
    rows = []
    for row in target_rows:
        archive = "separate_heads" if row["regime"] == "separate_head_structure" else "shared"
        q = fits[row["seed"], archive, row["arm"]]["q_mean"]
        forecasts = []
        for old in row["forecasts"]:
            null = old["a9_null_posterior"]
            score = float(prior_adjusted_log_score(old["source"], old["adapted"], null, q))
            adaptation_mass = q * (1. - null) / ((1. - q) * null + q * (1. - null))
            forecasts.append({"step": old["step"], "candidate": old["candidate"], "source": old["source"],
                "adapted": old["adapted"], "a9": old["a9"], "q_source": q,
                "target_adaptation_mass": float(adaptation_mass), "calibrated_log_score": score})
        rows.append({**{k: row[k] for k in ("seed", "regime", "arm")}, "forecasts": forecasts,
            "mean_calibrated_minus_source": float(np.mean([f["calibrated_log_score"] - f["source"] for f in forecasts])),
            "mean_calibrated_minus_a9": float(np.mean([f["calibrated_log_score"] - f["a9"] for f in forecasts]))})
    calibrations = [{"seed": s, "archive_group": a, "arm": arm, **fit} for (s, a, arm), fit in fits.items()]
    return calibrations, rows
