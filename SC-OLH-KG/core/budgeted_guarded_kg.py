"""One-step raw-observation value with a hard shared search/verification cap."""

import copy

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from core.guarded_deployment_kg import deployment_values, posterior_draws, _root, _completion


def terminal_covariance_root(covariance):
    """Keep existing full-rank draws; support an exactly shared constant eta."""
    try:
        return np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        return _root(covariance)


class BudgetedTerminalActions:
    """Three guarded allocations, or direct certification at 128/full budget.

    Every terminal action is frozen before verification. Guarded allocations
    reserve both objective-comparison streams and proportionally distribute
    the remaining safety calls using the legacy 80/128/128 proportions.
    Direct certification uses the verifier's one-policy family delta.
    """

    def __init__(self, shortlists, incumbent_positions, protocol):
        self.shortlists = np.asarray(shortlists)
        self.incumbents = np.asarray(incumbent_positions)
        self.protocol = protocol

    def retain_recommendation(self, candidate):
        """Offer the previous recommendation as a fallback at unchanged cost.

        Keep every original action, its primary and objective incumbent. New
        three-policy actions replace the final other support; they still use
        the same familywise delta and allocation, without a fourth verifier.
        The retained policy is selected before the next paid observation.
        """
        additions=[]; incumbents=[]
        seen={(tuple(row),int(incumbent)) for row,incumbent in zip(self.shortlists,self.incumbents)}
        for row,incumbent in zip(self.shortlists,self.incumbents):
            if candidate in row:
                continue
            updated=row.copy()
            position=[i for i in range(len(row)) if i not in (0,incumbent)][-1]
            updated[position]=candidate
            key=(tuple(updated),int(incumbent))
            if key in seen:
                continue
            seen.add(key)
            additions.append(updated); incumbents.append(incumbent)
        if additions:
            self.shortlists=np.vstack([self.shortlists,additions])
            self.incumbents=np.concatenate([self.incumbents,incumbents])

    def allocations(self,remaining_calls):
        reserve = 2*self.protocol["comparison_budget"]
        available = remaining_calls-reserve
        if available < 3*2:
            raise ValueError("guarded action needs at least two safety replicates per policy")
        allocations=[]
        for ratios in ([80,128,128], [128,80,128], [128,128,80]):
            budgets = np.floor(available*np.asarray(ratios)/sum(ratios)).astype(int)
            budgets[-1] += available-budgets.sum()
            allocations.append(("guarded",self.shortlists,self.incumbents,budgets.tolist()))
        singles = np.unique(self.shortlists[:,0])[:,None]
        for budget in dict.fromkeys([min(128,remaining_calls),remaining_calls]):
            allocations.append(("direct",singles,np.zeros(len(singles),dtype=int),[int(budget)]))
        return allocations

    def evaluate(self, draws, remaining_calls):
        rewards,costs,actions=[],[],[]
        for kind,table,inc,budgets in self.allocations(remaining_calls):
            r,c = deployment_values(draws,table,inc,
                                    dict(self.protocol,candidate_budgets=budgets))
            rewards.extend(r); costs.extend(c)
            actions.extend({"kind":kind,"primary":int(s[0]),"shortlist":s.tolist(),
                            "incumbent_position":int(i),"budgets":budgets}
                           for s,i in zip(table,inc))
        return np.asarray(rewards),np.asarray(costs),actions

    def evaluate_posterior(self,f_model,g_model,remaining_calls,power,seed):
        """Integrate each action's exact joint marginal in at most 12 dimensions.

        A terminal action uses at most three policies. Its expected reward
        needs their joint marginal, not hundreds of unrelated policy draws.
        Cross-policy and mean/eta covariance are retained; common Sobol nodes
        give the actions the same numerical integration rule.
        """
        if len(f_model.means)>1 or len(g_model.means)>1:
            from core.positive_mixture_lookahead import evaluate_positive_mixture
            return evaluate_positive_mixture(f_model,g_model,self,remaining_calls,power,seed)
        fm,fc=f_model.predict(full_cov=True); gm,gc=g_model.predict(full_cov=True)
        n=len(fm); fc=fc.reshape(2*n,2*n); gc=gc.reshape(2*n,2*n)
        z=ndtri(qmc.Sobol(12,scramble=True,seed=seed).random_base2(power))
        rewards,costs,actions=[],[],[]
        for kind,table,inc,budgets in self.allocations(remaining_calls):
            for indices,incumbent in zip(table,inc):
                m=len(indices); ix=(2*indices[:,None]+np.arange(2)).reshape(-1)
                fd=fm[indices].reshape(-1)+z[:,:2*m]@terminal_covariance_root(fc[np.ix_(ix,ix)]).T
                gd=gm[indices].reshape(-1)+z[:,6:6+2*m]@terminal_covariance_root(gc[np.ix_(ix,ix)]).T
                draws=np.concatenate([fd.reshape(-1,m,2),gd.reshape(-1,m,2)],axis=-1)
                r,c=deployment_values(draws,np.arange(m)[None],np.array([incumbent]),
                                      dict(self.protocol,candidate_budgets=budgets))
                rewards.append(r[0]); costs.append(c[0])
                actions.append({"kind":kind,"primary":int(indices[0]),"shortlist":indices.tolist(),
                                "incumbent_position":int(incumbent),"budgets":budgets})
        return np.asarray(rewards), np.asarray(costs), actions


def budgeted_raw_lookahead(f_model,g_model,terminal,remaining_calls,search_indices,
                          *,draw_power=9,fantasy_power=3,seed=80519):
    """Compare stopping now with one paid raw episode, including its debit.

    Repeated and new policies use the same likelihood. Moment projection can
    break the Bayesian tower identity, so report its drift separately from
    adaptation value; the chosen one-step action uses the actual projected
    future reward at remaining_calls-1. No observations are executed.
    """
    draws=posterior_draws(f_model,g_model,draw_power,seed)
    values,costs,actions=terminal.evaluate(draws,remaining_calls)
    best=int(np.lexsort((costs,-values))[0])
    next_values,_,_=terminal.evaluate(draws,remaining_calls-1)
    next_best=int(next_values.argmax())
    fm,fc=f_model.predict(full_cov=True); gm,gc=g_model.predict(full_cov=True)
    z=ndtri(qmc.Sobol(6,scramble=True,seed=seed+1).random_base2(fantasy_power))
    rows=[]
    for index in search_indices:
        fs=fm[index]+z[:,:2]@_root(fc[index,:,index,:]).T
        gs=gm[index]+z[:,3:5]@_root(gc[index,:,index,:]).T
        fo=fs[:,0]+np.exp(.5*fs[:,1])*z[:,2]
        go=gs[:,0]+np.exp(.5*gs[:,1])*z[:,5]
        future_best,baseline,eta_shrink=[],[],[]
        for f,g in zip(fo,go):
            fd=_completion(f_model,draws[...,:2],index,f)
            gd=_completion(g_model,draws[...,2:],index,g)
            future,_,_=terminal.evaluate(np.concatenate([fd,gd],axis=-1),remaining_calls-1)
            future_best.append(future.max()); baseline.append(future[next_best])
            eta_shrink.append(1-np.var(gd[:,index,1])/np.var(draws[:,index,3]))
        future_best,baseline=np.asarray(future_best),np.asarray(baseline)
        rows.append({"candidate":int(index),"expected_future_reward":float(future_best.mean()),
            "net_reward_of_paid_episode":float(future_best.mean()-values[best]),
            "adaptation_value":float(np.mean(future_best-baseline)),
            "projection_tower_drift":float(baseline.mean()-next_values[next_best]),
            "verification_budget_opportunity_cost":float(values[best]-next_values[next_best]),
            "expected_eta_variance_reduction_fraction":float(np.mean(eta_shrink))})
    selected=max(rows,key=lambda r:r["net_reward_of_paid_episode"])
    return {"stop_action":actions[best],"stop_expected_reward":float(values[best]),
            "stop_expected_verification_calls":float(costs[best]),"lookahead":rows,
            "choice":"sample" if selected["net_reward_of_paid_episode"]>0 else "stop",
            "sampling_candidate":selected["candidate"] if selected["net_reward_of_paid_episode"]>0 else None,
            "remaining_calls":int(remaining_calls),"draw_count":len(draws),"fantasy_count":len(z)}


def budgeted_marginal_lookahead(f_model,g_model,terminal,remaining_calls,search_indices,
                               *,draw_power=9,fantasy_power=3,seed=80519):
    """Same one-step decision, using exact joint marginals after paid filtering.

    Each fantasy updates a copied posterior through the actual update method.
    Terminal expectations use the same low-dimensional rule before and after
    that update. The predictive fantasies assume the single-component fields
    used by this frozen bridge; no paid simulator call is executed.
    """
    if len(f_model.component_pairs)!=1 or len(g_model.component_pairs)!=1:
        raise ValueError("this predictive sampler is for the single-component bridge")
    values,costs,actions=terminal.evaluate_posterior(f_model,g_model,remaining_calls,draw_power,seed)
    best=int(np.lexsort((costs,-values))[0])
    next_values,_,_=terminal.evaluate_posterior(f_model,g_model,remaining_calls-1,draw_power,seed)
    next_best=int(next_values.argmax())
    fm,fc=f_model.predict(full_cov=True); gm,gc=g_model.predict(full_cov=True)
    z=ndtri(qmc.Sobol(6,scramble=True,seed=seed+1).random_base2(fantasy_power))
    rows=[]
    for index in search_indices:
        fs=fm[index]+z[:,:2]@_root(fc[index,:,index,:]).T
        gs=gm[index]+z[:,3:5]@_root(gc[index,:,index,:]).T
        fo=fs[:,0]+np.exp(.5*fs[:,1])*z[:,2]
        go=gs[:,0]+np.exp(.5*gs[:,1])*z[:,5]
        future_best,baseline,eta_shrink=[],[],[]
        for fv,gv in zip(fo,go):
            f,g=copy.deepcopy(f_model),copy.deepcopy(g_model)
            f.update_batch(index,float(fv),None,1); g.update_batch(index,float(gv),None,1)
            future,_,_=terminal.evaluate_posterior(f,g,remaining_calls-1,draw_power,seed)
            future_best.append(future.max()); baseline.append(future[next_best])
            _,post_cov=g.predict(full_cov=True)
            eta_shrink.append(1-post_cov[index,1,index,1]/gc[index,1,index,1])
        future_best,baseline=np.asarray(future_best),np.asarray(baseline)
        rows.append({"candidate":int(index),"expected_future_reward":float(future_best.mean()),
            "net_reward_of_paid_episode":float(future_best.mean()-values[best]),
            "adaptation_value":float(np.mean(future_best-baseline)),
            "projection_tower_drift":float(baseline.mean()-next_values[next_best]),
            "verification_budget_opportunity_cost":float(values[best]-next_values[next_best]),
            "expected_eta_variance_reduction_fraction":float(np.mean(eta_shrink))})
    selected=max(rows,key=lambda r:r["net_reward_of_paid_episode"])
    return {"stop_action":actions[best],"stop_expected_reward":float(values[best]),
            "stop_expected_verification_calls":float(costs[best]),"lookahead":rows,
            "choice":"sample" if selected["net_reward_of_paid_episode"]>0 else "stop",
            "sampling_candidate":selected["candidate"] if selected["net_reward_of_paid_episode"]>0 else None,
            "remaining_calls":int(remaining_calls),"draw_count":2**draw_power,"fantasy_count":len(z)}
