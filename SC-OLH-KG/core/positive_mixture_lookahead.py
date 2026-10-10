"""Reuse guarded terminal rewards and the actual paid filter for source mixtures."""

import copy

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import digamma, logsumexp, ndtr, ndtri, roots_hermitenorm
from scipy.stats import chi2, qmc

from representation.physical_risk_prior import log_sample_variance_density

from core.raw_likelihood_lookahead import (factorized_terminal_values,
    objective_terminal_statistics,risk_terminal_statistics)


def field_draws(field, indices, nodes):
    """One component per joint policy draw preserves cross-policy dependence."""
    components=np.searchsorted(np.cumsum(np.exp(field.log_weights)),nodes[:,0])
    components=np.minimum(components,len(field.means)-1)
    ix=(2*np.asarray(indices)[:,None]+np.arange(2)).reshape(-1)
    means=field.means[:,ix][components]
    covariance=field.covariances[:,ix][:,:,ix][components]
    try:
        root=np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        from core.budgeted_guarded_kg import terminal_covariance_root
        root=np.stack([terminal_covariance_root(c) for c in covariance])
    draws=means+np.einsum('nij,nj->ni',root,ndtri(nodes[:,1:1+len(ix)]))
    return draws.reshape(1,len(nodes),len(indices),2)


def evaluate_positive_mixture(f,g,terminal,remaining,power,seed):
    """Independent-head integration inside the same shared model identity."""
    nodes=qmc.Sobol(7,scramble=True,seed=seed).random_base2(power)
    uniform=np.full((1,len(nodes)),1/len(nodes)); cache={}; rewards=[]; costs=[]; actions=[]
    for kind,table,inc,budgets in terminal.allocations(remaining):
        for indices,incumbent in zip(table,inc):
            key=tuple(indices)
            if key not in cache: cache[key]=(field_draws(f,indices,nodes),field_draws(g,indices,nodes))
            fd,gd=cache[key]
            r,c=factorized_terminal_values(fd,gd,uniform,uniform,int(incumbent),budgets,terminal.protocol)
            rewards.append(r[0]);costs.append(c[0])
            actions.append({'kind':kind,'primary':int(indices[0]),'shortlist':indices.tolist(),
                'incumbent_position':int(incumbent),'budgets':budgets})
    return np.asarray(rewards),np.asarray(costs),actions


class RawPredictiveCDF:
    """Historical conditional Gaussian CDF, averaged over eta and source states.

    F|eta and episode noise are integrated together. The remaining one-
    dimensional Gaussian eta integral retains each positive source weight.
    """
    def __init__(self,field,query,order=256):
        ix=[2*query,2*query+1]
        self.means=field.means[:,ix]
        self.covariance=field.covariances[:,ix][:,:,ix]
        self.weights=np.exp(field.log_weights)
        self.eta_sd=np.sqrt(self.covariance[:,1,1])
        self.loading=self.covariance[:,0,1]/self.eta_sd
        self.conditional=np.maximum(self.covariance[:,0,0]-self.loading**2,0.)
        z,w=roots_hermitenorm(order)
        self.locations=(self.means[:,0,None]+self.loading[:,None]*z).reshape(-1)
        self.scales=np.sqrt(self.conditional[:,None]
            +np.exp(self.means[:,1,None]+self.eta_sd[:,None]*z)).reshape(-1)
        self.integration_weights=(self.weights[:,None]*w/np.sqrt(2*np.pi)).reshape(-1)

    def __call__(self,values):
        values=np.asarray(values)
        flat=values.reshape(-1);result=np.empty(len(flat))
        for start in range(0,len(flat),16):
            result[start:start+16]=(ndtr((flat[start:start+16,None]-self.locations)/self.scales)
                @self.integration_weights)
        return result.reshape(values.shape)

    def log_density(self,values):
        """Collapse the raw likelihood without computing posterior moments."""
        keep=self.integration_weights>0
        locations,scales=self.locations[keep],self.scales[keep]
        logs=np.log(self.integration_weights[keep])-np.log(scales)-.5*np.log(2*np.pi)
        values=np.atleast_1d(values);parts=[]
        for start in range(0,len(values),8):
            residual=(values[start:start+8,None]-locations)/scales
            parts.append(logsumexp(logs-.5*residual**2,axis=1))
        return np.concatenate(parts)

    def adaptive(self,values):
        """Independent adaptive evaluation of the historical CDF statistic."""
        values=np.atleast_1d(values)
        def integrand(z):
            mean=self.means[:,0]+self.loading*z
            scale=np.sqrt(self.conditional+np.exp(self.means[:,1]+self.eta_sd*z))
            return (ndtr((values[:,None]-mean)/scale)@self.weights)*np.exp(-z*z/2)/np.sqrt(2*np.pi)
        result,error,info=quad_vec(integrand,-12.,12.,points=[0.],epsabs=1e-11,
            epsrel=1e-11,norm='max',full_output=True,workers=1)
        if not info.success:raise RuntimeError(info.message)
        return result,float(error)

    def inverse(self,probabilities):
        probabilities=np.asarray(probabilities)
        mean=float(self.weights@self.means[:,0])
        variance=float(self.weights@(self.covariance[:,0,0]+(self.means[:,0]-mean)**2
            +np.exp(self.means[:,1]+self.covariance[:,1,1]/2)))
        radius=np.sqrt(variance)
        while self(mean-radius)>probabilities.min() or self(mean+radius)<probabilities.max():radius*=2
        target=probabilities.reshape(-1)
        left=np.full(len(target),mean-radius);right=np.full(len(target),mean+radius)
        values=np.clip(mean+radius*ndtri(target),left,right)
        active=np.arange(len(target));result=np.empty(len(target))
        tolerance=max(radius*2.**-44,8*np.finfo(float).eps*max(1.,abs(mean)))
        for _ in range(44):
            if not len(active):break
            current=values[active];cdf=np.empty(len(active));density=np.empty(len(active))
            for start in range(0,len(active),16):
                z=(current[start:start+16,None]-self.locations)/self.scales
                cdf[start:start+16]=ndtr(z)@self.integration_weights
                density[start:start+16]=(np.exp(-.5*z*z)/self.scales/np.sqrt(2*np.pi))@self.integration_weights
            error=cdf-target[active]
            step=np.divide(error,density,out=np.full(len(active),np.inf),where=density>0.)
            left[active]=np.where(error<0.,current,left[active])
            right[active]=np.where(error>=0.,current,right[active])
            done=abs(step)<=tolerance
            result[active[done]]=current[done]-step[done]
            pending=active[~done];proposal=current[~done]-step[~done]
            inside=(proposal>left[pending])&(proposal<right[pending])
            values[pending]=np.where(inside,proposal,(left[pending]+right[pending])/2)
            active=pending
        # Retain the original accuracy for any quantiles still unresolved.
        if len(active):
            for _ in range(44):
                middle=(left[active]+right[active])/2;below=self(middle)<target[active]
                left[active]=np.where(below,middle,left[active]);right[active]=np.where(below,right[active],middle)
            result[active]=(left[active]+right[active])/2
        return result.reshape(probabilities.shape)


def predictive_values(model,query,power,seed):
    """Two observed uniforms; integrate latent source/component coordinates."""
    nodes=qmc.Sobol(2,scramble=True,seed=seed).random_base2(power)
    values=[]
    for pair in model.components:
        heads=[]
        for h,field in enumerate(pair):
            cdf=RawPredictiveCDF(field,query);values_h=cdf.inverse(nodes[:,h])
            reference,bound=cdf.adaptive(values_h)
            error=float(np.max(abs(reference-nodes[:,h])))+bound
            if error>1e-8:raise RuntimeError(f'Raw predictive inverse CDF error: {error}')
            heads.append(values_h)
        values.extend(zip(*heads))
    return np.asarray(values),np.repeat(model.weights/len(nodes),len(nodes))


def predictive_positive_batches(model,query,replicates,power,seed):
    """Fresh batch means and log variances from every positive source mixture.

    Each head samples its own component, eta, mean noise and chi-square noise
    within the shared complete identity. No field is collapsed to a Gaussian.
    """
    if replicates == 1:
        values,weights=predictive_values(model,query,power,seed)
        return values,None,weights
    if replicates < 2 or int(replicates) != replicates:
        raise ValueError('replicates must be a positive integer')
    nodes=qmc.Sobol(8,scramble=True,seed=seed).random_base2(power)
    k=(replicates-1)/2; values=[]; etas=[]
    for pair in model.components:
        heads=[]; variances=[]
        for head,field in enumerate(pair):
            u=nodes[:,4*head:4*head+4]
            component=np.minimum(np.searchsorted(np.cumsum(np.exp(field.log_weights)),u[:,0]),len(field.means)-1)
            ix=[2*query,2*query+1]
            mean=field.means[:,ix][component]; C=field.covariances[:,ix][:,:,ix][component]
            eta=mean[:,1]+np.sqrt(C[:,1,1])*ndtri(u[:,1])
            loading=C[:,0,1]/C[:,1,1]
            conditional=np.maximum(C[:,0,0]-loading*C[:,0,1],0.)
            heads.append(mean[:,0]+loading*(eta-mean[:,1])
                +np.sqrt(conditional+np.exp(eta)/replicates)*ndtri(u[:,2]))
            variances.append(eta+np.log(chi2.ppf(u[:,3],replicates-1)/(replicates-1))-digamma(k)+np.log(k))
        values.extend(zip(*heads)); etas.extend(zip(*variances))
    return np.asarray(values),np.asarray(etas),np.repeat(model.weights/len(nodes),len(nodes))


def raw_eta_proposal(field,updated_field,query,component,z,label,head_log_evidence,*,replicates=1,eta_label=None):
    """Historical exact eta likelihood ratio for posterior-component proposals."""
    ix=[2*query,2*query+1]
    mean=field.means[:,ix][component]; C=field.covariances[:,ix][:,:,ix][component]
    mu,eta=mean.T;ev=C[:,1,1];loading=C[:,0,1]/ev
    cv=C[:,0,0]-C[:,0,1]**2/ev
    pm=updated_field.means[component,2*query+1]
    pv=updated_field.covariances[component,2*query+1,2*query+1]
    sampled_eta=pm+np.sqrt(pv)*z
    total=cv+np.exp(sampled_eta)/replicates;residual=label-mu-loading*(sampled_eta-eta)
    logs=(-.5*(np.log(2*np.pi*total)+residual**2/total)
        -.5*(np.log(2*np.pi*ev)+(sampled_eta-eta)**2/ev)
        +.5*(np.log(2*np.pi*pv)+(sampled_eta-pm)**2/pv))
    if replicates>1:
        k=(replicates-1)/2
        sample_variance=np.exp(eta_label+digamma(k)-np.log(k))
        logs+=log_sample_variance_density(sampled_eta,sample_variance,replicates)
    component_evidence=updated_field.log_weights[component]-field.log_weights[component]+head_log_evidence
    return sampled_eta,logs-component_evidence


def condition_positive_raw(field,updated_field,query,points,nodes,label,head_log_evidence,*,replicates=1,eta_label=None):
    """Vectorize historical condition_raw over posterior source components.

    Draw the component from its exact updated weight, propose query eta using
    its adaptive posterior moments, then analytically condition the mean.
    Divide by the component evidence already used for component selection.
    """
    component=np.searchsorted(np.cumsum(np.exp(updated_field.log_weights)),nodes[:,0])
    component=np.minimum(component,len(field.means)-1)
    ix=(2*np.asarray(points)[:,None]+np.arange(2)).reshape(-1)
    means=field.means[:,ix][component];C=field.covariances[:,ix][:,:,ix][component]
    order=np.r_[1,0,np.arange(2,len(ix))]
    ordered=C[:,order][:,:,order]
    try:root=np.linalg.cholesky(ordered)
    except np.linalg.LinAlgError:
        from core.budgeted_guarded_kg import terminal_covariance_root
        root=np.stack([terminal_covariance_root(c) for c in ordered])
    z=ndtri(nodes[:,1:1+len(ix)])
    draws=means[:,order]+np.einsum('nij,nj->ni',root,z);draws=draws[:,np.argsort(order)]
    mu,eta=means[:,:2].T;ev=C[:,1,1];loading=C[:,0,1]/ev
    cv=C[:,0,0]-C[:,0,1]**2/ev
    sampled_eta,logs=raw_eta_proposal(field,updated_field,query,component,z[:,0],label,head_log_evidence,
        replicates=replicates,eta_label=eta_label)
    shifted=draws+(sampled_eta-draws[:,1])[:,None]*C[:,:,1]/ev[:,None]
    conditional_mean=mu+loading*(sampled_eta-eta)
    noise=np.exp(sampled_eta)/replicates;total=cv+noise;residual=label-conditional_mean
    posterior_mean=conditional_mean+cv/total*residual
    posterior_query=posterior_mean+np.sqrt(noise/total)*(shifted[:,0]-conditional_mean)
    gain=(C[:,:,0]-loading[:,None]*C[:,:,1])/cv[:,None]
    transported=shifted+(posterior_query-shifted[:,0])[:,None]*gain
    normalizer=logsumexp(logs);weights=np.exp(logs-normalizer)
    return transported,weights,float(normalizer-np.log(len(nodes)))


def fixed_action_factorized_reference(model,action,query,*,draw_power,terminal_power,
        fantasy_power,seed,protocol):
    """Cancel the predictive identity density before fixed-action integration.

    p(y)w(h|y)=w(h)p_f(y_f|h)p_g(y_g|h). Each terminal branch
    is a product of independent head statistics conditional on h. Integrate
    those two statistics separately, then multiply inside that same h.
    The adaptive maximum is never factorized this way.
    """
    points=list(dict.fromkeys([query]+action['shortlist']))
    positions=[points.index(i) for i in action['shortlist']]
    nodes=qmc.Sobol(1+2*len(points),scramble=True,seed=seed).random_base2(draw_power)
    projected_nodes=qmc.Sobol(7,scramble=True,seed=seed).random_base2(terminal_power)
    uniform=np.full((1,len(nodes)),1/len(nodes))
    projected_uniform=np.full((1,len(projected_nodes)),1/len(projected_nodes))
    probabilities=qmc.Sobol(2,scramble=True,seed=seed+1).random_base2(fantasy_power)
    prior=likelihood=projected=projected_prior=0.;evidence_error=0.;ess_min=float(len(nodes))
    inverse_error=0.;maximum_raw_error=0.;updates=0
    for identity,pair in enumerate(model.components):
        before=[];after=[];projection=[];projection_before=[]
        for h,field in enumerate(pair):
            def statistic(draws,weights):
                if h==0:return objective_terminal_statistics(draws,weights,action['incumbent_position'],protocol)[0]
                return risk_terminal_statistics(draws,weights,action['incumbent_position'],action['budgets'],protocol)[0][0]
            original=field_draws(field,points,nodes)[:,:,positions]
            before.append(statistic(original,uniform))
            original=field_draws(field,action['shortlist'],projected_nodes)
            projection_before.append(statistic(original,projected_uniform))
            cdf=RawPredictiveCDF(field,query);values=cdf.inverse(probabilities[:,h])
            reference,bound=cdf.adaptive(values)
            inverse_error=max(inverse_error,float(np.max(abs(reference-probabilities[:,h])))+bound)
            if inverse_error>1e-8:raise RuntimeError(f'Raw predictive inverse CDF error: {inverse_error}')
            conditional=[];gaussian=[]
            for y in values:
                future=copy.deepcopy(field);evidence=future.update_batch(query,float(y),None,1);updates+=1
                maximum_raw_error=max(maximum_raw_error,future.max_integration_relative_error)
                draws,w,error=condition_positive_raw(field,future,query,points,nodes,y,evidence)
                draws=draws.reshape(1,len(nodes),len(points),2)[:,:,positions]
                conditional.append(statistic(draws,w[None]))
                gaussian.append(statistic(field_draws(future,action['shortlist'],projected_nodes),projected_uniform))
                evidence_error=max(evidence_error,abs(error));ess_min=min(ess_min,float(1/(w@w)))
            after.append(np.mean(conditional,axis=0));projection.append(np.mean(gaussian,axis=0))
        weight=model.weights[identity]
        prior+=weight*(before[0]@before[1]);likelihood+=weight*(after[0]@after[1])
        projected_prior+=weight*(projection_before[0]@projection_before[1])
        projected+=weight*(projection[0]@projection[1])
    return {'candidate':int(query),'action':action,'seed':seed,'fantasy_power':fantasy_power,
        'likelihood_prior_reward':float(prior),'likelihood_expected_reward':float(likelihood),
        'likelihood_tower_drift':float(likelihood-prior),
        'projected_prior_reward':float(projected_prior),'projected_expected_reward':float(projected),
        'projected_baseline_drift':float(projected-projected_prior),
        'projection_minus_likelihood_tower_drift':float(projected-projected_prior-likelihood+prior),
        'predictive_log_evidence_max_error':evidence_error,'effective_sample_size_min':ess_min,
        'maximum_inverse_cdf_error':inverse_error,'maximum_raw_integration_relative_error':maximum_raw_error,
        'head_raw_updates':updates,'terminal_draw_count':len(projected_nodes),'reference_draw_count':len(nodes),
        'fantasies_per_head_identity':len(probabilities),'shared_identity_retained':True,
        'previous_paid_projection_retained':True,'adaptive_maximum_factorized':False}


def fixed_action_positive_likelihood_reference(model,action,query,*,draw_power,fantasy_power,seed,protocol):
    """Reweight unchanged joint latent draws; compare the actual projected filter.

    This is a finite importance reference for one next raw observation, not
    an exact multibatch posterior. Predictive evidence and weight ESS expose
    insufficient particles before interpreting projection differences.
    """
    points=list(dict.fromkeys([query]+action['shortlist']))
    positions=[points.index(i) for i in action['shortlist']]
    nodes=qmc.Sobol(1+2*len(points),scramble=True,seed=seed).random_base2(draw_power)
    uniform=np.full((1,len(nodes)),1/len(nodes));fields=[];prior=0.
    for h,pair in enumerate(model.components):
        draws=[field_draws(field,points,nodes)[0] for field in pair]
        reward,_=factorized_terminal_values(draws[0][None,:,positions],draws[1][None,:,positions],
            uniform,uniform,action['incumbent_position'],action['budgets'],protocol)
        prior+=model.weights[h]*reward[0];fields.append(draws)
    fantasies,fantasy_weights=predictive_values(model,query,fantasy_power,seed+1)
    likelihood=[];projected=[];identity=[];evidence_error=0.;ess_min=float(len(nodes))
    for values in fantasies:
        future=copy.deepcopy(model);record=future.update_raw(query,values)
        exact_scores=np.asarray(record['head_log_scores']);lr=0.;pr=0.
        for h,pair in enumerate(future.components):
            weights=[];conditioned=[]
            for head,current in enumerate(pair):
                draws,w,error=condition_positive_raw(model.components[h][head],current,query,points,nodes,
                    values[head],exact_scores[h,head])
                conditioned.append(draws.reshape(len(nodes),len(points),2));weights.append(w[None])
                ess_min=min(ess_min,float(1./(w@w)));evidence_error=max(evidence_error,abs(error))
            fd,gd=conditioned
            reward,_=factorized_terminal_values(fd[None,:,positions],gd[None,:,positions],weights[0],weights[1],
                action['incumbent_position'],action['budgets'],protocol)
            lr+=future.weights[h]*reward[0]
            fd,gd=[field_draws(field,points,nodes)[:,:,positions] for field in pair]
            reward,_=factorized_terminal_values(fd,gd,uniform,uniform,action['incumbent_position'],action['budgets'],protocol)
            pr+=future.weights[h]*reward[0]
        likelihood.append(lr);projected.append(pr);identity.append(future.weights)
    likelihood,projected=np.asarray(likelihood),np.asarray(projected)
    return {'candidate':int(query),'seed':seed,'action':action,'prior_expected_reward':float(prior),
        'likelihood_tower_drift':float(fantasy_weights@likelihood-prior),
        'projected_baseline_drift':float(fantasy_weights@projected-prior),
        'projection_minus_likelihood_baseline':float(fantasy_weights@(projected-likelihood)),
        'predictive_log_evidence_max_error':float(evidence_error),'effective_sample_size_min':ess_min,
        'model_weight_martingale_residual':(fantasy_weights@np.asarray(identity)-model.weights).tolist(),
        'draw_count':len(nodes),'fantasies_per_component':2**fantasy_power,'fantasy_count':len(fantasies),
        'positive_source_components_retained':True,'reference':'posterior-component and eta proposal with analytic query-mean transport',
        'previous_paid_projection_retained':True}


