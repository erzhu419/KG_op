"""Paid-evidence mixture of complete source response/risk hypotheses.

Each hypothesis has independent (f,eta_f) and (g,eta_g) fields conditional on
its identity. The identity is shared by both heads: their evidence jointly
updates one weight, and terminal values integrate the complete hypotheses.
"""

import numpy as np
from scipy.special import digamma,logsumexp

from core.gpr import normalize_mixture_weights
from representation.joint_function_risk import joint_batch_log_score


class JointResponseMixture:
    def __init__(self,names,components,prior_weights=None):
        self.names=list(names)
        self.components=list(components)
        if prior_weights is None:
            self.log_weights=np.full(len(components),-np.log(len(components)))
        else:
            weights=normalize_mixture_weights(prior_weights)
            if len(weights)!=len(components):
                raise ValueError('source weights and complete hypotheses must align')
            self.log_weights=np.full(len(weights),-np.inf)
            np.log(weights,out=self.log_weights,where=weights>0)
        self.paid_episodes=0
        self.history=[]

    @property
    def weights(self):
        return np.exp(self.log_weights)

    def update_raw(self,candidate,values):
        """One physical episode supplies both values, with no variance label.

        Each field returns its predictive density from before conditioning.
        Conditional head independence makes their joint log density additive.
        Normalization happens only after both scores have been obtained.
        """
        return self._update_statistics(candidate,values,[None,None],1)

    def update_replicates(self,candidate,values):
        """Condition once on a fresh batch of paired (f, centered g) values.

        The sample variance comes only from this batch. Previous paid values
        are already in the starting posterior and are not counted again.
        """
        values=np.asarray(values,dtype=float)
        count=len(values)
        eta=[None,None]
        if count>1:
            k=(count-1)/2
            eta=np.log(np.var(values,axis=0,ddof=1))-digamma(k)+np.log(k)
        return self._update_statistics(candidate,values.mean(axis=0),eta,count)

    def _update_statistics(self,candidate,means,eta,replicates):
        scores=[]
        for f,g in self.components:
            scores.append([f.update_batch(candidate,float(means[0]),eta[0],replicates),
                           g.update_batch(candidate,float(means[1]),eta[1],replicates)])
        scores=np.asarray(scores)
        return self._record_scores(candidate,scores,replicates)

    def _record_scores(self,candidate,scores,replicates):
        """Shared identity update for actual observations and cached fantasies."""
        before=self.log_weights.copy();scores=np.asarray(scores)
        joint=scores.sum(axis=1)
        evidence=float(logsumexp(before+joint))
        self.log_weights=before+joint-evidence
        self.paid_episodes+=replicates
        record={"candidate":int(candidate),"replicates":int(replicates),"head_log_scores":scores.tolist(),
            "joint_log_scores":joint.tolist(),"mixture_log_score":evidence,
            "weights_before":np.exp(before).tolist(),"weights_after":self.weights.tolist()}
        self.history.append(record)
        return record

    def predict_joint(self):
        """Full (f,eta_f,g,eta_g) moments, including identity dependence."""
        means,covariances=[],[]
        for f,g in self.components:
            fm,fc=f.predict(full_cov=True); gm,gc=g.predict(full_cov=True)
            n=len(fm)
            mean=np.concatenate([fm,gm],axis=1).reshape(-1)
            C=np.zeros((4*n,4*n))
            fi=(4*np.arange(n)[:,None]+np.arange(2)).reshape(-1)
            gi=fi+2
            C[np.ix_(fi,fi)]=fc.reshape(2*n,2*n)
            C[np.ix_(gi,gi)]=gc.reshape(2*n,2*n)
            means.append(mean); covariances.append(C)
        means=np.asarray(means)
        mean=self.weights@means
        centered=means-mean
        C=np.einsum("m,mij->ij",self.weights,np.asarray(covariances))
        C+=(centered.T*self.weights)@centered
        return mean.reshape(-1,4),C.reshape(n,4,n,4)

    def batch_log_scores(self,candidate,means,eta,replicates):
        """Score a fresh paired batch without changing the paid posterior.

        Head marginals mix separately; their joint density mixes complete
        identities, preserving dependence induced by the shared identity.
        """
        scores=np.array([[joint_batch_log_score(field.local_components(candidate),
            means[h],eta[h],replicates) for h,field in enumerate(pair)]
            for pair in self.components])
        return {"joint":float(logsumexp(self.log_weights+scores.sum(axis=1))),
                "heads":logsumexp(self.log_weights[:,None]+scores,axis=0).tolist()}

    def terminal_components(self,terminal,remaining_calls,power,seed,*,utility_identities=None):
        """Integrate each hypothesis, retaining the shared f/g identity."""
        rewards,costs=[],[]
        identities=range(len(self.components)) if utility_identities is None else utility_identities
        for identity in identities:
            f,g=self.components[identity]
            r,c,actions=terminal.evaluate_posterior(f,g,remaining_calls,power,seed)
            rewards.append(r); costs.append(c)
        return np.asarray(rewards),np.asarray(costs),actions

    def terminal_values(self,terminal,remaining_calls,power,seed,*,utility_identities=None):
        """Original weights; skipped utilities have a common omitted-mass bound."""
        r,c,actions=self.terminal_components(terminal,remaining_calls,power,seed,
            utility_identities=utility_identities)
        weights=self.weights if utility_identities is None else self.weights[utility_identities]
        return weights@r,weights@c,actions
