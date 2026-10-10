"""Joint-hypothesis raw fantasies for the same budgeted terminal utility."""

import copy

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from core.guarded_deployment_kg import _root


def query_risk_variance(model,index):
    """The same eta_g marginal,without assembling unused full covariances."""
    means=[];variances=[];coordinate=2*index+1
    for _,field in model.components:
        weights=np.exp(field.log_weights);values=field.means[:,coordinate]
        mean=float(weights@values);means.append(mean)
        variances.append(float(weights@(field.covariances[:,coordinate,coordinate]+(values-mean)**2)))
    means=np.asarray(means);mean=float(model.weights@means)
    return float(model.weights@(np.asarray(variances)+(means-mean)**2))


def mixture_raw_lookahead(model,terminal,remaining_calls,search_indices,
                           *,draw_power=9,fantasy_power=3,seed=80530,baseline=None,
                           fixed_action_expectations=None,completed_prefix=None):
    """Stratify fantasies by the shared source hypothesis and debit one call.

    Every fantasy uses the actual joint weight/filter update. Terminal values
    integrate model identities instead of independently pooling f and g.
    Gaussian projection and finite integration can violate the tower identity;
    their baseline drift is reported separately from the adaptation value.
    A supplied fixed-action expectation integrates the same debited incumbent
    independently. Use it as a control variate;the adaptive contrast remains
    joint and any actual projected-filter tower drift remains visible.
    A completed same-seed prefix supplies its aggregate expectations;only new
    fantasies need terminal evaluation. Prefix reuse is for a single query.
    """
    if baseline is None:
        values,costs,actions=model.terminal_values(terminal,remaining_calls,draw_power,seed)
        debited,_,_=model.terminal_values(terminal,remaining_calls-1,draw_power,seed)
    else:
        values,costs,actions,debited,_=baseline
    best=int(np.lexsort((costs,-values))[0])
    debited_best=int(debited.argmax())
    z=ndtri(qmc.Sobol(6,scramble=True,seed=seed+1).random_base2(fantasy_power))
    prefix_count=0 if completed_prefix is None else completed_prefix['fantasies_per_component']
    if completed_prefix is not None:
        assert len(search_indices)==1 and completed_prefix['candidate']==search_indices[0]
        assert 0<prefix_count<=len(z)//2
    rows=[]
    for index in search_indices:
        before_variance=query_risk_variance(model,index)
        positive=any(len(field.means)>1 for pair in model.components for field in pair)
        if positive:
            from core.positive_mixture_lookahead import predictive_values
            from representation.vectorized_joint_function_risk import raw_fantasy_moments,apply_raw_moments
            raw_values,_=predictive_values(model,index,fantasy_power,seed+1)
            new_positions=[h*len(z)+p for h in range(len(model.components)) for p in range(prefix_count,len(z))]
            cache=[[raw_fantasy_moments(field,index,raw_values[new_positions,head]) if len(field.means)>1 else None
                for head,field in enumerate(pair)] for pair in model.components]
        future_values,future_baseline,eta_shrink=[],[],[]
        future_weights=[]
        prefix_adaptation=[]
        maximum_error=0. if completed_prefix is None else completed_prefix['maximum_raw_integration_relative_error']
        for h,(f,g) in enumerate(model.components):
            if positive:
                fo,go=raw_values[h*len(z):(h+1)*len(z)].T
            else:
                fm,fc=f.predict(full_cov=True); gm,gc=g.predict(full_cov=True)
                fs=fm[index]+z[:,:2]@_root(fc[index,:,index,:]).T
                gs=gm[index]+z[:,3:5]@_root(gc[index,:,index,:]).T
                fo=fs[:,0]+np.exp(.5*fs[:,1])*z[:,2]
                go=gs[:,0]+np.exp(.5*gs[:,1])*z[:,5]
            component_best,component_baseline,component_shrink,component_weights=[],[],[],[]
            for position,(fv,gv) in enumerate(zip(fo,go)):
                if position<prefix_count:
                    continue
                future=copy.deepcopy(model)
                if positive:
                    scores=[];sample=h*(len(z)-prefix_count)+position-prefix_count
                    for identity,pair in enumerate(future.components):
                        heads=[]
                        for head,field in enumerate(pair):
                            if cache[identity][head] is None:
                                heads.append(field.update_batch(index,[fv,gv][head],None,1))
                            else:
                                moments,gain=cache[identity][head]
                                heads.append(apply_raw_moments(field,index,moments[sample],gain))
                        scores.append(heads)
                    future._record_scores(index,scores,1)
                else:future.update_raw(index,[fv,gv])
                assert future.paid_episodes==model.paid_episodes+1
                maximum_error=max(maximum_error,max(field.max_integration_relative_error
                    for pair in future.components for field in pair))
                r,_,_=future.terminal_values(terminal,remaining_calls-1,draw_power,seed)
                component_best.append(r.max()); component_baseline.append(r[debited_best])
                component_shrink.append(1-query_risk_variance(future,index)/before_variance)
                component_weights.append(future.weights)
            future_values.append(np.mean(component_best)); future_baseline.append(np.mean(component_baseline))
            eta_shrink.append(np.mean(component_shrink)); future_weights.append(np.mean(component_weights,axis=0))
            prefix=max(1,len(z)//2)-prefix_count
            prefix_adaptation.append(float(np.mean(np.asarray(component_best[:prefix])-component_baseline[:prefix])) if prefix else 0.)
        joint_fv=float(model.weights@future_values);joint_base=float(model.weights@future_baseline)
        shrink=float(model.weights@eta_shrink);expected_weights=model.weights@np.asarray(future_weights)
        nested_adaptation=float(model.weights@prefix_adaptation)
        if completed_prefix is not None:
            fraction=prefix_count/len(z);nested_fraction=prefix_count/(len(z)//2)
            joint_fv=fraction*completed_prefix['raw_joint_expected_future_reward']+(1-fraction)*joint_fv
            prior_base=completed_prefix['raw_joint_expected_future_reward']-completed_prefix['adaptation_value']
            joint_base=fraction*prior_base+(1-fraction)*joint_base
            shrink=fraction*completed_prefix['expected_eta_variance_reduction_fraction']+(1-fraction)*shrink
            expected_weights=fraction*np.asarray(completed_prefix['expected_model_weights'])+(1-fraction)*expected_weights
            # A prefix exactly half of the new rule needs no additional
            # evaluations to reproduce that rule's nested adaptation.
            if prefix_count==len(z)//2:
                nested_adaptation=completed_prefix['adaptation_value']
            else:
                nested_adaptation=nested_fraction*completed_prefix['adaptation_value']+(1-nested_fraction)*nested_adaptation
        adaptation=joint_fv-joint_base
        base=(joint_base if fixed_action_expectations is None else float(fixed_action_expectations[int(index)]))
        fv=adaptation+base
        rows.append({"candidate":int(index),"expected_future_reward":fv,
            "net_reward_of_paid_episode":fv-float(values[best]),"adaptation_value":adaptation,
            "nested_prefix_adaptation_value":nested_adaptation,
            "nested_adaptation_difference":abs(adaptation-nested_adaptation),
            "raw_joint_expected_future_reward":joint_fv,
            "raw_joint_baseline_drift":joint_base-float(debited[debited_best]),
            "fixed_action_expected_reward":base,
            "projection_tower_drift":base-float(debited[debited_best]),
            "filter_and_quadrature_baseline_drift":base-float(debited[debited_best]),
            "verification_budget_opportunity_cost":float(values[best]-debited[debited_best]),
            "expected_eta_variance_reduction_fraction":shrink,
            "expected_model_weights":expected_weights.tolist(),
            "model_weight_martingale_residual":(expected_weights-model.weights).tolist(),
            "maximum_raw_integration_relative_error":maximum_error,
            "fantasies_per_component":len(z),"reused_fantasies":prefix_count*len(model.components),
            "new_terminal_fantasies":(len(z)-prefix_count)*len(model.components),
            "positive_source_mixtures_retained":positive,"actual_paid_filter_used_for_every_fantasy":True})
    selected=max(rows,key=lambda r:r["net_reward_of_paid_episode"])
    return {"stop_action":actions[best],"stop_expected_reward":float(values[best]),
        "stop_expected_verification_calls":float(costs[best]),"lookahead":rows,
        "choice":"sample" if selected["net_reward_of_paid_episode"]>0 else "stop",
        "sampling_candidate":selected["candidate"] if selected["net_reward_of_paid_episode"]>0 else None,
        "remaining_calls":int(remaining_calls),"draw_count":2**draw_power,
        "fantasies_per_component":len(z),"fantasy_count":len(z)*len(model.components)}
