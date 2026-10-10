"""Use the existing elliptical kernel on global and real source-task blocks."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from performance.check_legacy_f_collapsed_source_raw_update import elliptical_step
from performance.diagnose_legacy_f_source_incremental_calibration import log_density


class SourceBlockLikelihood:
    def __init__(self, eta, indices, model):
        self.eta, self.indices, self.model = eta, indices, model

    def evaluate(self, block):
        proposal = self.eta.copy(); proposal[self.indices] = block
        return self.model.evaluate(proposal)


class SourceRiskLineLikelihood:
    def __init__(self, origin, direction, model):
        self.origin, self.direction, self.model = origin, direction, model

    def evaluate(self, coefficient):
        return self.model.evaluate(self.origin + self.direction * coefficient[0])


def source_block_transition(mean, covariance, joint_direction_sites=()):
    """Gaussian conditioning keeps the old compensated posterior unchanged.

    The supported archive has four genuine12-site source tasks. Each block
    uses q(eta_task | eta_other_tasks), not a marginal Gaussian q block.
    """
    mean, covariance = np.asarray(mean), np.asarray(covariance)
    assert mean.shape == (48,)
    precision = cho_solve(cho_factor(covariance, lower=True), np.eye(48))
    blocks=[]; maximum_error=0.
    for indices in np.split(np.arange(48),4):
        other=np.setdiff1d(np.arange(48),indices)
        block_covariance=cho_solve(cho_factor(precision[np.ix_(indices,indices)],lower=True),np.eye(12))
        loading=-block_covariance@precision[np.ix_(indices,other)]
        dense_loading=cho_solve(cho_factor(covariance[np.ix_(other,other)],lower=True),covariance[np.ix_(other,indices)]).T
        dense_covariance=covariance[np.ix_(indices,indices)]-dense_loading@covariance[np.ix_(other,indices)]
        np.testing.assert_allclose(loading,dense_loading,atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(block_covariance,dense_covariance,atol=1e-10,rtol=1e-10)
        point=mean+.15*np.sqrt(np.diag(covariance));conditional_mean=mean[indices]+loading@(point[other]-mean[other])
        changed=point.copy();changed[indices]+=.1*np.sqrt(np.diag(block_covariance))
        joint_delta=log_density(changed-mean,covariance)-log_density(point-mean,covariance)
        block_delta=(log_density(changed[indices]-conditional_mean,block_covariance)
            -log_density(point[indices]-conditional_mean,block_covariance))
        error=abs(joint_delta-block_delta);assert error<1e-9;maximum_error=max(maximum_error,error)
        blocks.append((indices,other,loading,np.linalg.cholesky(block_covariance)))

    directions=[]; line_error=0.
    for site in joint_direction_sites:
        sd=np.sqrt(covariance[site,site]);direction=covariance[:,site]/sd
        # z=(eta_site-mean_site)/sd is N(0,1), independent of
        # eta-mean-direction*z. The line moves correlated sites together.
        point=mean+.15*np.sqrt(np.diag(covariance));z=(point[site]-mean[site])/sd
        changed=point+.4*direction
        error=abs(log_density(changed-mean,covariance)-log_density(point-mean,covariance)
            +.5*((z+.4)**2-z*z))
        assert error<1e-9;line_error=max(line_error,error)
        directions.append((site,sd,direction))

    def transition(eta, prior_mean, prior_factor, evaluation, model, rng):
        eta,evaluation,count=elliptical_step(eta,prior_mean,prior_factor,evaluation,model,rng)
        for indices,other,loading,root in blocks:
            conditional_mean=mean[indices]+loading@(eta[other]-mean[other])
            block,evaluation,n=elliptical_step(eta[indices],conditional_mean,root,evaluation,
                SourceBlockLikelihood(eta,indices,model),rng)
            eta=eta.copy();eta[indices]=block;count+=n
        for site,sd,direction in directions:
            z=np.array([(eta[site]-mean[site])/sd]);origin=eta-direction*z[0]
            z,evaluation,n=elliptical_step(z,np.zeros(1),np.ones((1,1)),evaluation,
                SourceRiskLineLikelihood(origin,direction,model),rng)
            eta=origin+direction*z[0];count+=n
        return eta,evaluation,count
    return transition,{'four_real_source_blocks':True,'maximum_conditional_density_ratio_error':maximum_error,
        'joint_risk_direction_sites':list(joint_direction_sites),'maximum_joint_risk_line_density_ratio_error':line_error,
        'scientific_population_prior_and_exact_likelihood_unchanged':True}
