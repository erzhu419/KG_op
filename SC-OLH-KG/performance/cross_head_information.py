"""A13 posterior function energy on archived, already-paid observations."""
import numpy as np

from performance.run_meta_prior_mechanism import observe
from representation.transfer_mixture import TransferMixtureGPR


def coefficient_energy(model, function_dim=68):
    """Exclude the intercept and candidate discrepancies; keep mixture variance."""
    mean = np.asarray(model.a)[1:function_dim + 1]
    variance = np.diag(model.C)[1:function_dim + 1]
    energy = mean ** 2 + variance
    if len(energy) != function_dim or not np.isfinite(energy).all() or energy.sum() <= 0:
        raise ValueError("invalid posterior function second moments")
    return energy / energy.sum()


def posterior_signals(models, function_dim=68):
    heads = [coefficient_energy(m, function_dim) for m in models]
    return {"f": heads[0].tolist(), "g": heads[1].tolist(), "eta": heads[2].tolist(),
            "fg": ((heads[0] + heads[1]) / 2.).tolist()}


def score_support(signals, risk_indices):
    """Simulator support is an evaluation input, absent from posterior fitting."""
    masses = {h: float(np.asarray(v)[risk_indices].sum()) for h, v in signals.items()}
    return {"support_mass": masses, "fg_minus_eta": masses["fg"] - masses["eta"]}


def reconstruct_signals(record, context, protocol, prefixes):
    row = record["row"]
    psi, reference, full, structural, truth, target_seed = context
    candidates = np.arange(len(psi))[:, None]
    learned = row["arm"] == "risk_ard_mixture"
    models = ([TransferMixtureGPR([structural[h], full[h]], psi, reference) for h in range(3)]
              if learned else [p.target_model(psi) for p in full])
    if (len(row["queries"]) != protocol["target_points"] or
            row["queries"][:protocol["initial_points"]] != row["initial_design"]):
        raise ValueError("archived query path changed")
    if any(m.moment.p != 69 if learned else m.p != 69 for m in models):
        raise ValueError("A13 requires the original 68-function basis")
    visits = np.zeros(len(psi), dtype=int)
    snapshots = []
    for step, index in enumerate(row["queries"], 1):
        # Truth here regenerates the original paid response, not support learning.
        response, noise = observe(truth[index], target_seed, index, int(visits[index]),
                                  protocol["replicates"])
        for h, model in enumerate(models):
            model.update(candidates[index], response[h], noise[h])
        visits[index] += 1
        if step in prefixes:
            snapshots.append({"prefix_batches": step, "signals": posterior_signals(models),
                "structural_weights": [m.diagnostics()["structural_weight"] for m in models]
                                       if learned else []})
    return snapshots, models
