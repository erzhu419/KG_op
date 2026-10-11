"""Marginal upper bounds for the existing Normal chance verifier.

For z>0, convexity gives exp(eta/2)>=exp(t/2)*(1+(eta-t)/2).
The feasible event g+z*exp(eta/2)<=0 is therefore inside a Gaussian
halfspace for every fixed t. Their smallest probability remains an upper
bound. No independence between g and eta, or between policies, is assumed.
"""
import numpy as np
from scipy.special import ndtr, ndtri, roots_hermitenorm, logsumexp

from core.guarded_deployment_kg import conditional_safety_power
from core.positive_mixture_lookahead import raw_eta_proposal


def gaussian_feasibility_upper(mean, covariance, z):
    assert z > 0
    mean, covariance = np.asarray(mean), np.asarray(covariance)
    t = mean[...,1,None]+np.sqrt(np.maximum(covariance[...,1,1,None],0.))*np.asarray([-4.,-2.,-1.,0.,1.,2.,4.])
    exponential = z*np.exp(t/2); slope = exponential/2
    location = mean[...,0,None]+exponential+slope*(mean[...,1,None]-t)
    variance = np.maximum(covariance[...,0,0,None]+2*slope*covariance[...,0,1,None]
        +slope**2*covariance[...,1,1,None],0.)
    with np.errstate(divide='ignore',invalid='ignore'):
        probability = ndtr(-location/np.sqrt(variance))
    probability = np.where(variance>0,probability,(location<=0).astype(float))
    return probability.min(axis=-1)


def condition_pair(mean, covariance, query, policy, eta, label, *, replicates=1):
    """Gaussian (g_policy,eta_policy) given query eta and the raw label."""
    mean, covariance = np.asarray(mean), np.asarray(covariance)
    q, e = 2*query, 2*query+1; target = [2*policy,2*policy+1]
    variance = covariance[...,e,e]
    cross = covariance[...,target,e]
    before_mean = mean[...,target]+cross/variance[...,None]*(eta-mean[...,e])[...,None]
    before_covariance = covariance[...,target,:][...,target]-cross[..., :,None]*cross[...,None,:]/variance[...,None,None]
    query_mean = mean[...,q]+covariance[...,q,e]/variance*(eta-mean[...,e])
    total = covariance[...,q,q]-covariance[...,q,e]**2/variance+np.exp(eta)/replicates
    observation_cross = covariance[...,target,q]-cross*covariance[...,e,q,None]/variance[...,None]
    after_mean = before_mean+observation_cross/total[...,None]*(label-query_mean)[...,None]
    after_covariance = before_covariance-observation_cross[..., :,None]*observation_cross[...,None,:]/total[...,None,None]
    return after_mean,after_covariance


def raw_feasibility_upper(field, updated, query, policy, label, score, alpha, order,
        *, replicates=1, eta_label=None):
    z, gh = roots_hermitenorm(order); gh /= np.sqrt(2*np.pi)
    active = np.flatnonzero(np.isfinite(updated.log_weights))
    components = np.repeat(active,order)
    eta, logs = raw_eta_proposal(field,updated,query,components,np.tile(z,len(active)),label,score,
        replicates=replicates,eta_label=eta_label)
    mean,covariance = condition_pair(field.means[components],field.covariances[components],query,policy,eta,label,
        replicates=replicates)
    log_weights = updated.log_weights[components]+np.tile(np.log(gh),len(active))+logs
    mass = logsumexp(log_weights)
    upper = np.exp(log_weights-mass) @ gaussian_feasibility_upper(mean,covariance,ndtri(1-alpha))
    return float(upper),dict(log_evidence_error=float(mass),components=len(active),order=order,
        points=len(components),all_positive_components_retained=True)


def projected_feasibility_upper(updated, policy, alpha):
    ix = [2*policy,2*policy+1]
    upper = gaussian_feasibility_upper(updated.means[:,ix],updated.covariances[:,ix][:,:,ix],ndtri(1-alpha))
    return float(np.exp(updated.log_weights) @ upper)


def certificate_upper(feasible, budget, shortlist_size, protocol):
    """Use the boundary power of the actual saved SD16 rule, not nominal delta.

    Certificate power decreases in g/sigma. Outside the feasible event its
    maximum is the boundary power, even when numerical SD16 differs from the
    nominal noncentral-t test size. Inside that event its maximum is one.
    """
    boundary = conditional_safety_power(-ndtri(1-protocol['chance_alpha']),1.,budget,
        protocol['chance_alpha'],protocol['familywise_delta']/shortlist_size)
    return float(boundary+(1-boundary)*feasible)


def branch_quality_upper(quality, marginals, incumbent):
    """Fractional-knapsack relaxation of the same mutually exclusive branches."""
    quality, marginals = np.asarray(quality), np.asarray(marginals)
    paired = np.minimum(marginals[...,0],marginals[...,incumbent]) if incumbent else np.zeros(marginals.shape[:-1])
    caps = np.concatenate([marginals,paired[...,None]],axis=-1)
    order = np.argsort(-quality,axis=-1)
    caps = np.take_along_axis(np.broadcast_to(caps,quality.shape),order,axis=-1)
    qualities = np.take_along_axis(quality,order,axis=-1)
    available = np.maximum(1-np.cumsum(caps,axis=-1)+caps,0.)
    return np.sum(qualities*np.minimum(caps,available),axis=-1)
