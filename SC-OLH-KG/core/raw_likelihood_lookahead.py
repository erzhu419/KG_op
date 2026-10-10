"""One-step raw likelihood reference without projecting the new eta posterior.

The starting mixture is the existing Gaussian moment approximation. Conditional
on each identity, f/eta_f and g/eta_g are independent. Query means can be
conditioned analytically given eta, leaving one-dimensional likelihood weights
per head. Other policies retain their exact Gaussian conditional distribution.
Only this next update avoids projection; previous paid updates stay unchanged.
"""

import numpy as np
from scipy.special import digamma,logsumexp, ndtri
from scipy.stats import chi2,qmc

from core.guarded_deployment_kg import (
    _root, conditional_comparison_power, conditional_safety_power)
from representation.joint_function_risk import local_joint_moments
from representation.physical_risk_prior import log_sample_variance_density


def query_first_draws(mean,covariance,query,indices,z):
    """Put query eta first, keeping its Sobol samples common to all actions."""
    points=list(dict.fromkeys([query]+list(map(int,indices))))
    ix=(2*np.asarray(points)[:,None]+np.arange(2)).reshape(-1)
    local_mean=mean[points].reshape(-1)
    C=covariance.reshape(2*len(mean),2*len(mean))[np.ix_(ix,ix)]
    order=np.r_[1,0,np.arange(2,len(ix))]
    draws=local_mean[order]+z[:,:len(ix)]@np.linalg.cholesky(C[np.ix_(order,order)]).T
    draws=draws[:,np.argsort(order)]
    positions=[points.index(int(i)) for i in indices]
    return local_mean,C,draws,positions


def condition_raw(mean,C,draws,observations,results,*,replicates=1,eta_labels=None):
    """Condition means; sample eta around its posterior, with exact weights.

    A Gaussian proposal uses the adaptive integral's local eta moments. Its
    density ratio retains the non-Gaussian posterior; it is not a moment
    projection. Transporting the entire field along its eta loading keeps
    the original Gaussian conditional residual before observing the mean.
    """
    mu,eta=mean[:2]; ev=C[1,1]; loading=C[0,1]/ev
    cv=C[0,0]-C[0,1]**2/ev
    pm=np.array([r['mean'][1] for r in results])[:,None]
    pv=np.array([r['covariance'][1,1] for r in results])[:,None]
    sampled_eta=pm+np.sqrt(pv/ev)*(draws[None,:,1]-eta)
    shifted=draws[None]+(sampled_eta-draws[None,:,1])[...,None]*C[:,1]/ev
    conditional_mean=mu+loading*(sampled_eta-eta)
    noise=np.exp(sampled_eta)/replicates; total=cv+noise
    residual=observations[:,None]-conditional_mean
    logs=(-.5*(np.log(2*np.pi*total)+residual**2/total)
        -.5*(np.log(2*np.pi*ev)+(sampled_eta-eta)**2/ev)
        +.5*(np.log(2*np.pi*pv)+(sampled_eta-pm)**2/pv))
    if replicates>1:
        k=(replicates-1)/2
        sample_variance=np.exp(np.asarray(eta_labels)+digamma(k)-np.log(k))
        logs+=log_sample_variance_density(sampled_eta,sample_variance[:,None],replicates)
    log_evidence=logsumexp(logs,axis=1)-np.log(len(draws))
    weights=np.exp(logs-logsumexp(logs,axis=1)[:,None])
    posterior_mean=conditional_mean+cv/total*residual
    posterior_query=(posterior_mean+np.sqrt(noise/total)*
                     (shifted[...,0]-conditional_mean))
    gain=(C[:,0]-loading*C[:,1])/cv
    updated=shifted+(posterior_query-shifted[...,0])[...,None]*gain
    return updated,weights,log_evidence


def project_raw(mean,C,draws,results):
    """Same Gaussian completion as the actual paid filter, on the same draws."""
    local=C[:2,:2]
    whitened=np.linalg.solve(_root(local),(draws[:,:2]-mean[:2]).T).T
    gain=np.linalg.solve(local,C[:2]).T
    after=np.stack([r["mean"]+whitened@_root(r["covariance"]).T for r in results])
    return draws[None]+(after-draws[None,:,:2])@gain.T


def objective_terminal_statistics(f,fw,incumbent,protocol):
    """Clipped qualities and the objective-comparison branch within one head."""
    quality=np.clip(1-f[...,0]/protocol["objective_reference"],0,1)
    eq=np.einsum("fs,fsi->fi",fw,quality)
    paired=np.zeros(len(f))
    if incumbent!=0:
        i=incumbent
        comparison=conditional_comparison_power((f[...,0,0]-f[...,i,0]).reshape(-1),
            np.abs(np.exp(.5*f[...,0,1])-np.exp(.5*f[...,i,1])).reshape(-1),
            protocol["comparison_budget"],protocol["comparison_delta"]).reshape(f.shape[:2])
        paired=(fw*(comparison*quality[...,0]+(1-comparison)*quality[...,i])).sum(axis=1)
    return np.column_stack([eq,paired])


def risk_terminal_statistics(g,gw,incumbent,budgets,protocol):
    """Safety branch probabilities and call cost, jointly across policies."""
    delta=protocol["familywise_delta"]/len(budgets)
    powers=np.stack([conditional_safety_power(g[...,i,0],np.exp(.5*g[...,i,1]),n,
        protocol["chance_alpha"],delta) for i,n in enumerate(budgets)],axis=-1)
    return risk_branch_statistics(powers,gw,incumbent,budgets,protocol)


def risk_branch_statistics(powers,gw,incumbent,budgets,protocol):
    """Reuse the verifier branches after integrating any latent mean coordinate."""
    branches=np.zeros((len(powers),len(budgets)+1));cost=np.zeros(len(powers))
    if incumbent==0:
        fail=np.ones_like(powers[...,0])
        for i,n in enumerate(budgets):
            branches[:,i]=(gw*fail*powers[...,i]).sum(axis=1)
            cost+=(gw*fail).sum(axis=1)*n
            fail=fail*(1-powers[...,i])
        return branches,cost
    i=incumbent; p0,pi=powers[...,0],powers[...,i]; both=p0*pi
    branches[:,0]=(gw*p0*(1-pi)).sum(axis=1)
    branches[:,i]=(gw*(1-p0)*pi).sum(axis=1)
    branches[:,-1]=(gw*both).sum(axis=1)
    cost=budgets[0]+budgets[i]+(gw*both).sum(axis=1)*2*protocol["comparison_budget"]
    fail=(1-p0)*(1-pi)
    for j,n in enumerate(budgets):
        if j in (0,i): continue
        branches[:,j]=(gw*fail*powers[...,j]).sum(axis=1)
        cost+=(gw*fail).sum(axis=1)*n
        fail=fail*(1-powers[...,j])
    return branches,cost


def factorized_terminal_values(f,g,fw,gw,incumbent,budgets,protocol):
    """Independent-head statistics within one shared identity."""
    fs=objective_terminal_statistics(f,fw,incumbent,protocol)
    gs,cost=risk_terminal_statistics(g,gw,incumbent,budgets,protocol)
    return (fs*gs).sum(axis=1),cost


def _actions(terminal,remaining):
    return [{"kind":kind,"primary":int(indices[0]),"shortlist":list(map(int,indices)),
             "incumbent_position":int(i),"budgets":budgets}
            for kind,table,inc,budgets in terminal.allocations(remaining)
            for indices,i in zip(table,inc)]


def _posterior_weights(moments,query,fantasies,log_weights,*,replicates=1,eta_labels=None):
    results=[]; scores=[]
    for (fm,fc),(gm,gc) in moments:
        heads=[[local_joint_moments(mean[query],C[query,:,query,:],y,
                None if eta_labels is None else eta_labels[i,head],replicates)
                for i,y in enumerate(fantasies[:,head])] for head,(mean,C) in enumerate(((fm,fc),(gm,gc)))]
        results.append(heads); scores.append(np.array([[r["log_evidence"] for r in head] for head in heads]).T)
    scores=np.asarray(scores)
    logw=log_weights[:,None]+scores.sum(axis=2)
    return results,scores,np.exp(logw-logsumexp(logw,axis=0))


def predictive_raw_values(moments,query,power,seed):
    """Four-dimensional predictive rule after integrating out query means."""
    z=ndtri(qmc.Sobol(4,scramble=True,seed=seed).random_base2(power))
    values=[]
    for heads in moments:
        samples=[]
        for head,(mean,cov) in enumerate(heads):
            m,C=mean[query],cov[query,:,query,:]
            eta=m[1]+np.sqrt(C[1,1])*z[:,2*head]
            loading=C[0,1]/C[1,1]; cv=C[0,0]-C[0,1]**2/C[1,1]
            samples.append(m[0]+loading*(eta-m[1])+np.sqrt(cv+np.exp(eta))*z[:,2*head+1])
        values.extend(zip(*samples))
    return np.asarray(values)


def predictive_batch_statistics(moments,query,replicates,power,seed):
    """Exact marginal law of fresh means and bias-corrected log variances.

    Each head has independent eta, mean-normal and sample-variance chi-square
    coordinates. Batch sizes share random coordinates, not nested raw paths.
    """
    if replicates==1:
        return predictive_raw_values(moments,query,power,seed),None
    u=qmc.Sobol(6,scramble=True,seed=seed).random_base2(power); z=ndtri(u)
    k=(replicates-1)/2; means=[]; etas=[]
    for heads in moments:
        samples=[]; variances=[]
        for head,(mean,cov) in enumerate(heads):
            m,C=mean[query],cov[query,:,query,:]
            eta=m[1]+np.sqrt(C[1,1])*z[:,3*head]
            loading=C[0,1]/C[1,1]; cv=C[0,0]-C[0,1]**2/C[1,1]
            samples.append(m[0]+loading*(eta-m[1])+np.sqrt(cv+np.exp(eta)/replicates)*z[:,3*head+1])
            variances.append(eta+np.log(chi2.ppf(u[:,3*head+2],replicates-1)/(replicates-1))-digamma(k)+np.log(k))
        means.extend(zip(*samples)); etas.extend(zip(*variances))
    return np.asarray(means),np.asarray(etas)


def preposterior_information(model,moments,query,results,postw,fantasy_weights,count,indices):
    """Exact conditional field moments from local adaptive integrals.

    The unchanged Gaussian conditional of other policies transfers local
    information through the learned covariance. Identity mixing follows
    conditioning; the within-identity metric averages only its own fantasies.
    """
    means=[]; variances=[]; within=[]
    for h,heads in enumerate(moments):
        hm=[]; hv=[]
        for head,(m,C) in enumerate(heads):
            local=C[query,:,query,:]; n=len(m)
            gain=np.linalg.solve(local,C[query].reshape(2,2*n)).T
            lm=np.array([r['mean'] for r in results[h][head]])
            lv=np.array([r['covariance'] for r in results[h][head]])
            hm.append((m.reshape(-1)+(lm-m[query])@gain.T).reshape(-1,n,2))
            hv.append((np.diag(C.reshape(2*n,2*n))+np.einsum('ij,fjk,ik->fi',gain,lv-local,gain)).reshape(-1,n,2))
        means.append(np.concatenate(hm,axis=2)); variances.append(np.concatenate(hv,axis=2))
        own=hv[1][h*count:(h+1)*count,:,1].mean(axis=0)
        within.append({str(i):float(1-own[i]/heads[1][1][i,1,i,1]) for i in indices})
    means=np.asarray(means); variances=np.asarray(variances)
    mixed_mean=np.einsum('hf,hfij->fij',postw,means)
    mixed_variance=np.einsum('hf,hfij->fij',postw,variances+(means-mixed_mean[None])**2)
    _,prior=model.predict_joint()
    prior_variance=np.array([np.diag(prior[i,:,i,:]) for i in range(len(mixed_mean[0]))])
    reduction=1-np.einsum('f,fij->ij',fantasy_weights,mixed_variance)/prior_variance
    return {str(i):reduction[i].tolist() for i in indices},dict(zip(model.names,within))


def fixed_action_likelihood_reference(model,action,query,*,draw_power=13,
                                     fantasy_power=7,seed=80563,protocol):
    """Refine only one frozen baseline; block fantasies to bound local memory."""
    if any(len(field.means)>1 for pair in model.components for field in pair):
        from core.positive_mixture_lookahead import fixed_action_positive_likelihood_reference
        return fixed_action_positive_likelihood_reference(model,action,query,draw_power=draw_power,
            fantasy_power=fantasy_power,seed=seed,protocol=protocol)
    moments=[(f.predict(full_cov=True),g.predict(full_cov=True)) for f,g in model.components]
    fantasies=predictive_raw_values(moments,query,fantasy_power,seed+1)
    results,scores,postw=_posterior_weights(moments,query,fantasies,model.log_weights)
    count=2**fantasy_power; fantasy_weights=np.repeat(model.weights/count,count)
    z=ndtri(qmc.Sobol(8,scramble=True,seed=seed).random_base2(draw_power))
    uniform=np.full((1,len(z)),1/len(z)); prior=0.
    exact=np.zeros(len(fantasies)); projected=exact.copy(); evidence_error=0.; ess_min=float(len(z))
    for h,heads in enumerate(moments):
        fields=[query_first_draws(mean,C,query,action['shortlist'],z) for mean,C in heads]
        indices=[(2*np.asarray(d[3])[:,None]+np.arange(2)).reshape(-1) for d in fields]
        fd,gd=[d[2][:,ix].reshape(1,len(z),len(action['shortlist']),2) for d,ix in zip(fields,indices)]
        prior+=model.weights[h]*factorized_terminal_values(fd,gd,uniform,uniform,
            action['incumbent_position'],action['budgets'],protocol)[0][0]
        for start in range(0,len(fantasies),32):
            stop=min(start+32,len(fantasies)); fs=fantasies[start:stop]
            conditioned=[condition_raw(m,C,draws,fs[:,head],results[h][head][start:stop])
                         for head,(m,C,draws,_) in enumerate(fields)]
            for head,c in enumerate(conditioned):
                evidence_error=max(evidence_error,float(np.abs(c[2]-scores[h,start:stop,head]).max()))
                ess_min=min(ess_min,float((1/(c[1]**2).sum(axis=1)).min()))
            m=len(action['shortlist'])
            ff,gg=[c[0][...,ix].reshape(len(fs),len(z),m,2) for c,ix in zip(conditioned,indices)]
            r,_=factorized_terminal_values(ff,gg,conditioned[0][1],conditioned[1][1],
                action['incumbent_position'],action['budgets'],protocol)
            exact[start:stop]+=postw[h,start:stop]*r
            pf,pg=[project_raw(d[0],d[1],d[2],results[h][head][start:stop])[...,ix].reshape(len(fs),len(z),m,2)
                   for head,(d,ix) in enumerate(zip(fields,indices))]
            w=np.broadcast_to(uniform,(len(fs),len(z)))
            r,_=factorized_terminal_values(pf,pg,w,w,action['incumbent_position'],action['budgets'],protocol)
            projected[start:stop]+=postw[h,start:stop]*r
    return {'candidate':int(query),'seed':seed,'action':action,'prior_expected_reward':float(prior),
        'likelihood_tower_drift':float(fantasy_weights@exact-prior),
        'projected_baseline_drift':float(fantasy_weights@projected-prior),
        'projection_minus_likelihood_baseline':float(fantasy_weights@(projected-exact)),
        'predictive_log_evidence_max_error':evidence_error,'effective_sample_size_min':ess_min,
        'model_weight_martingale_residual':(postw@fantasy_weights-model.weights).tolist(),
        'draw_count':len(z),'fantasies_per_component':count,'fantasy_count':len(fantasies)}


def likelihood_query_lookahead(model,terminal,remaining,query,*,draw_power=9,
                               fantasy_power=5,seed=80561,replicates=1):
    """Fresh raw/batch observations and budgeted terminal adaptation."""
    if any(len(field.means)>1 for pair in model.components for field in pair):
        raise ValueError('Positive source mixtures require mixture_raw_lookahead;this reference starts from one Gaussian per head.')
    moments=[(f.predict(full_cov=True),g.predict(full_cov=True)) for f,g in model.components]
    z=ndtri(qmc.Sobol(8,scramble=True,seed=seed).random_base2(draw_power))
    fantasies,eta_labels=predictive_batch_statistics(moments,query,replicates,fantasy_power,seed+1)
    count=2**fantasy_power
    fantasy_weights=np.repeat(model.weights/count,count)
    results,scores,posterior_weights=_posterior_weights(moments,query,fantasies,model.log_weights,
        replicates=replicates,eta_labels=eta_labels)
    now=_actions(terminal,remaining); future_actions=_actions(terminal,remaining-replicates)
    before=np.zeros((len(moments),len(now))); debited=before.copy()
    before_cost=before.copy(); debited_cost=before.copy()
    future=np.zeros((len(fantasies),len(now))); future_cost=future.copy()
    projected_future=future.copy(); projected_cost=future.copy()
    evidence_errors=[]; ess=[]; query_moments=[]
    uniform=np.full((1,len(z)),1/len(z))
    for h,((fm,fc),(gm,gc)) in enumerate(moments):
        groups={}
        for a,action in enumerate(now):
            groups.setdefault(tuple(action["shortlist"]),[]).append(a)
        for key,action_indices in groups.items():
            fields=[query_first_draws(mean,C,query,key,z) for mean,C in ((fm,fc),(gm,gc))]
            conditioned=[condition_raw(mean,C,draws,fantasies[:,head],results[h][head],replicates=replicates,
                eta_labels=None if eta_labels is None else eta_labels[:,head])
                for head,(mean,C,draws,_) in enumerate(fields)]
            fd,gd=[d[2][:,np.reshape(2*np.asarray(d[3])[:,None]+np.arange(2),-1)].reshape(1,len(z),len(key),2) for d in fields]
            ff,gg=[c[0][...,np.reshape(2*np.asarray(d[3])[:,None]+np.arange(2),-1)].reshape(len(fantasies),len(z),len(key),2)
                   for d,c in zip(fields,conditioned)]
            pf,pg=[project_raw(d[0],d[1],d[2],results[h][head])[
                ...,np.reshape(2*np.asarray(d[3])[:,None]+np.arange(2),-1)].reshape(len(fantasies),len(z),len(key),2)
                for head,d in enumerate(fields)]
            projected_weights=np.broadcast_to(uniform,(len(fantasies),len(z)))
            fw,gw=[c[1] for c in conditioned]
            if 0 in action_indices:
                evidence_errors.extend((conditioned[head][2]-scores[h,:,head]).tolist() for head in range(2))
                ess.extend((1/(c[1]**2).sum(axis=1)).tolist() for c in conditioned)
                field_stats=[]
                for c in conditioned:
                    q=c[0][...,:2]; w=c[1]; m=np.einsum("fs,fsi->fi",w,q)
                    v=np.einsum("fs,fsi->fi",w,(q-m[:,None])**2)
                    field_stats.append((m,v))
                query_moments.append(field_stats)
            for a in action_indices:
                action,after=now[a],future_actions[a]; inc=action["incumbent_position"]
                r,c=factorized_terminal_values(fd,gd,uniform,uniform,inc,action["budgets"],terminal.protocol)
                before[h,a],before_cost[h,a]=r[0],c[0]
                r,c=factorized_terminal_values(fd,gd,uniform,uniform,inc,after["budgets"],terminal.protocol)
                debited[h,a],debited_cost[h,a]=r[0],c[0]
                reward,cost=factorized_terminal_values(ff,gg,fw,gw,inc,after["budgets"],terminal.protocol)
                future[:,a]+=posterior_weights[h]*reward; future_cost[:,a]+=posterior_weights[h]*cost
                reward,cost=factorized_terminal_values(pf,pg,projected_weights,projected_weights,inc,after['budgets'],terminal.protocol)
                projected_future[:,a]+=posterior_weights[h]*reward; projected_cost[:,a]+=posterior_weights[h]*cost
    value=model.weights@before; debit=model.weights@debited
    best=int(np.lexsort((model.weights@before_cost,-value))[0])
    dbest=int(np.lexsort((model.weights@debited_cost,-debit))[0]); projected=projected_future[:,dbest]
    selected=np.lexsort((future_cost,-future),axis=1)[:,0]; rewards=future[np.arange(len(fantasies)),selected]
    costs=future_cost[np.arange(len(fantasies)),selected]
    base=future[:,dbest]
    selected_projected=np.lexsort((projected_cost,-projected_future),axis=1)[:,0]
    projected_best=projected_future[np.arange(len(fantasies)),selected_projected]
    projected_best_cost=projected_cost[np.arange(len(fantasies)),selected_projected]
    # Assess the approximate filter's selected action under the unprojected
    # conditional posterior. Its assigned value and achieved model-side
    # reward differ; their gap must not be labeled an action-selection loss.
    projected_choice_reward=future[np.arange(len(fantasies)),selected_projected]
    projected_choice_cost=future_cost[np.arange(len(fantasies)),selected_projected]
    average=lambda v:float(fantasy_weights@v)
    # Law of total variance includes changing identity weights in each fantasy.
    means=np.stack([np.concatenate([s[0] for s in heads],axis=1) for heads in query_moments])
    variances=np.stack([np.concatenate([s[1] for s in heads],axis=1) for heads in query_moments])
    mixture_mean=np.einsum("hf,hfi->fi",posterior_weights,means)
    mixture_var=np.einsum("hf,hfi->fi",posterior_weights,variances+(means-mixture_mean[None])**2)
    prior_mean,prior_cov=model.predict_joint(); prior_var=np.diag(prior_cov[query,:,query,:])
    indices=sorted(set([query]+[i for action in now for i in action['shortlist']]))
    information,within=preposterior_information(model,moments,query,results,posterior_weights,fantasy_weights,count,indices)
    subset=np.concatenate([np.arange(h*count,h*count+min(16,count)) for h in range(len(moments))])
    sw=np.repeat(model.weights/ min(16,count),min(16,count))
    return {"candidate":int(query),"replicates":int(replicates),"remaining_calls":int(remaining),"future_remaining_calls":int(remaining-replicates),
        "stop_action":now[best],"debited_baseline_action":future_actions[dbest],
        "stop_expected_reward":float(value[best]),"debited_expected_reward":float(debit[dbest]),
        "expected_future_reward":average(rewards),"net_reward_of_batch":average(rewards)-float(value[best]),
        "net_reward_of_paid_episode":(average(rewards)-float(value[best]))/replicates,
        "adaptation_value":average(rewards-base),"likelihood_tower_drift":average(base)-float(debit[dbest]),
        "projected_baseline_drift":average(projected)-float(debit[dbest]),
        "projection_minus_likelihood_baseline":average(projected-base),
        "projected_expected_future_reward":average(projected_best),
        "projected_net_reward_of_batch":average(projected_best)-float(value[best]),
        "projected_adaptation_value":average(projected_best-projected),
        "projected_expected_future_verification_calls":average(projected_best_cost),
        "projection_minus_likelihood_best_value":average(projected_best-rewards),
        "likelihood_reward_of_projected_choice":average(projected_choice_reward),
        "likelihood_net_reward_of_projected_choice":average(projected_choice_reward)-float(value[best]),
        "likelihood_adaptation_of_projected_choice":average(projected_choice_reward-base),
        "likelihood_verification_calls_of_projected_choice":average(projected_choice_cost),
        "approximation_choice_loss":average(rewards-projected_choice_reward),
        "projected_value_bias_at_selected_action":average(projected_best-projected_choice_reward),
        "projected_primary_disagreement_probability":average(np.array([
            future_actions[int(a)]['primary']!=future_actions[int(b)]['primary']
            for a,b in zip(selected,selected_projected)],dtype=float)),
        "verification_budget_opportunity_cost":float(value[best]-debit[dbest]),
        "expected_future_verification_calls":average(costs),
        "expected_model_weights":(posterior_weights@fantasy_weights).tolist(),
        "model_weight_martingale_residual":(posterior_weights@fantasy_weights-model.weights).tolist(),
        "expected_variance_reduction_fraction":(1-fantasy_weights@mixture_var/prior_var).tolist(),
        "exact_moment_variance_reduction_by_candidate":information,
        "within_identity_eta_g_variance_reduction":within,
        "effective_sample_size_min":float(np.min(ess)),"effective_sample_size_median":float(np.median(ess)),
        "predictive_log_evidence_max_error":float(np.max(np.abs(evidence_errors))),
        "first_16_fantasies":{"adaptation_value":float(sw@(rewards-base)[subset]),
            "likelihood_tower_drift":float(sw@base[subset]-debit[dbest]),
            "projected_baseline_drift":float(sw@projected[subset]-debit[dbest])},
        "future_decisions":[{"action":future_actions[int(a)],"reward":float(r),
            "projected_action":future_actions[int(selected_projected[i])],
            "likelihood_reward_of_projected_choice":float(projected_choice_reward[i]),
            "origin_hypothesis":int(i//count),**({'raw_observation':y.tolist()} if eta_labels is None else
                {'batch_mean':y.tolist(),'batch_eta_labels':eta_labels[i].tolist()})}
            for i,(a,r,y) in enumerate(zip(selected,rewards,fantasies))],
        "draw_count":len(z),"fantasies_per_component":count,"fantasy_count":len(fantasies)}
