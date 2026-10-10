"""Reuse risk spectra across I/F/G/R, two prior arms and quadrature orders."""
import numpy as np
from scipy.special import logsumexp

from performance.risk_family_reweighting import (
    evaluate_covariance_paths, family_covariance, prepare_covariance_paths, rho_quadrature,
)
from performance.scale_preserving_source_covariance import scale_preserving_covariance
from performance.target_risk_direction_mixture import joint_label_log_scores


def fold_forecasts(priors, task, selected, families, orders=(16, 32, 48)):
    psi, labels, noise = task["psi"], task["response"][:, 2], task["noise"][:, 2]
    result = {arm: {str(n): {"relative_log_evidence": [], "component_log_density": []} for n in orders}
              for arm in ("risk_ard_mixture", "fixed_source")}
    endpoints = [[scale_preserving_covariance(p, selected[h], 1.) for h in ("f", "g")] +
                 [family_covariance(p, indices, 1.) for indices in families.values()] for p in priors]
    for t in range(8, 24):
        prepared = [prepare_covariance_paths(p, psi, np.arange(t), labels[:t], noise[:t], t, e)
                    for p, e in zip(priors, endpoints)]
        for nodes in orders:
            log_weights = np.log(rho_quadrature(nodes)[1])
            evaluated = [evaluate_covariance_paths(p, nodes) for p in prepared]
            # Per-component marginal evidence and future joint density;
            # generic uses the same full component computed for the mixture.
            component_evidence, component_joint = [], []
            for prep, paths in zip(prepared, evaluated):
                source_e, mean, variance = prep["source"]
                evidence = [source_e]
                joint = [source_e + float(joint_label_log_scores(np.array([mean]), np.array([variance]), labels[t:t + 1], noise[t:t + 1])[0])]
                for branch in ((0,), (1,), tuple(range(2, 2 + len(families)))):
                    logs, joints = [], []
                    for index in branch:
                        e, m, v = paths[index]
                        log_mass = log_weights - np.log(len(branch))
                        logs.append(e + log_mass)
                        joints.append(e + log_mass + joint_label_log_scores(m, v, labels[t:t + 1], noise[t:t + 1]))
                    evidence.append(float(logsumexp(np.concatenate(logs))))
                    joint.append(float(logsumexp(np.concatenate(joints))))
                component_evidence.append(evidence)
                component_joint.append(joint)
            component_evidence, component_joint = np.array(component_evidence), np.array(component_joint)
            for arm, indices in (("risk_ard_mixture", [0, 1]), ("fixed_source", [1])):
                evidence = logsumexp(component_evidence[indices], axis=0) - np.log(len(indices))
                joint = logsumexp(component_joint[indices], axis=0) - np.log(len(indices))
                if arm == "fixed_source":
                    evidence[1:3], joint[1:3] = evidence[0], joint[0]
                out = result[arm][str(nodes)]
                out["relative_log_evidence"].append((evidence - evidence[0]).tolist())
                out["component_log_density"].append((joint - evidence).tolist())
    return result
