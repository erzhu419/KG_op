"""Nested source sharing calibration from per-archive Gaussian evidence."""
import numpy as np
from scipy.special import logsumexp, roots_legendre

from representation.archive_covariance_prior import sharing_branches


def cached_context(row, archive_keys, excluded_seeds, local_seed):
    indices = [i for i,(s,g) in enumerate(archive_keys) if s not in excluded_seeds]
    local = archive_keys.index([local_seed, row["archive_group"]])
    if local in indices:
        raise ValueError("local archive must be excluded from population")
    indices.append(local)
    columns = [j for i in indices for j in (2*i,2*i+1)]
    risk = {"log_evidence":np.array(row["risk_log_evidence"])[:,columns],
            "log_density":np.array(row["risk_log_density"])[:,columns]}
    auxiliary = np.array(row["auxiliary_log_evidence"])[:,indices]
    weights = np.r_[np.full(len(indices)-1,.5/(len(indices)-1)),.5]
    return risk, auxiliary, weights


def training_profiles(cache, pairs, held):
    evidence, predictive = [], []
    for row in cache["rows"]:
        if row["pair_id"] == held:
            continue
        scored_pair = pairs[row["pair_id"]]
        excluded = pairs[held] + scored_pair
        if len(set(excluded)) != 4:
            raise ValueError("nested outer and inner pairs overlap")
        for local in (row["seed"], next(s for s in scored_pair if s != row["seed"])):
            risk,auxiliary,weights = cached_context(row,cache["archive_keys"],excluded,local)
            if len(weights) != 25:
                raise ValueError("nested calibration requires24 population archives and local")
            branches = sharing_branches(risk,auxiliary,weights)
            evidence.append(branches["log_evidence"])
            predictive.append(branches["log_density"])
    return np.concatenate(evidence), np.concatenate(predictive)


def fit_sharing_prior(evidence, predictive, nodes):
    x,w = roots_legendre(nodes)
    q,quadrature = (x+1)/2,w/2
    logs,complement = np.log(q),np.log1p(-q)
    # Half matched + half transfer, for the same paid source observations.
    score = np.zeros(nodes)
    for start in range(0,len(evidence),512):
        e,p = evidence[start:start+512],predictive[start:start+512]
        shared = (e[:,0]-e[:,1])[:,None]+logs
        denominator = np.logaddexp(shared,complement)
        numerator = np.logaddexp(shared+p[:,:1],complement+p[:,1:2])
        score += .5*(numerator-denominator).sum(axis=0)
    posterior = np.exp(score+np.log(quadrature)-logsumexp(score+np.log(quadrature)))
    moments = [float(posterior @ (q**i)) for i in (1,2,3)]
    return {"q_mean":moments[0],"q_sd":float(np.sqrt(max(moments[1]-moments[0]**2,0.))),
            "q_moments":moments,"reference":"Uniform Beta(1,1)",
            "score_weight_per_condition":.5,"q_nodes":q.tolist(),"posterior_weights":posterior.tolist()}


def posterior_q_moments(branches, prior):
    """Update the continuous source q prior using the task's past evidence."""
    e = branches["log_evidence"]
    relative = np.exp(e-e.max(axis=1,keepdims=True))
    shared,separate = relative[:,0],relative[:,1]
    m1,m2,m3 = prior["q_moments"]
    denominator = shared*m1+separate*(1-m1)
    mean = (shared*m2+separate*(m1-m2))/denominator
    second = (shared*m3+separate*(m2-m3))/denominator
    return mean,np.sqrt(np.maximum(second-mean**2,0.))
