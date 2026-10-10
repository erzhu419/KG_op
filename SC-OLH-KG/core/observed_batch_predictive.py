"""Positive-source batch predictions in the two observed coordinates."""

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import digamma, gammainc, logsumexp, ndtr, polygamma, roots_hermitenorm

from representation.physical_risk_prior import log_sample_variance_density


def inverse_observed_cdf(probabilities, evaluate, mean, variance):
    """The historical safeguarded CDF inversion, for a conditional law."""
    target=np.asarray(probabilities);radius=np.sqrt(variance)
    left=np.full(target.shape,mean-radius);right=np.full(target.shape,mean+radius)
    while np.any(evaluate(left)[0]>target) or np.any(evaluate(right)[0]<target):
        radius*=2;left.fill(mean-radius);right.fill(mean+radius)
    current=(left+right)/2
    for _ in range(48):
        cdf,density=evaluate(current);error=cdf-target
        done=abs(error)<=2e-13
        left=np.where(error<0,current,left);right=np.where(error>=0,current,right)
        step=np.divide(error,density,out=np.full_like(error,np.inf),where=density>0.);proposal=current-step
        if np.all(done):return current
        current=np.where(done,current,np.where((proposal>left)&(proposal<right),proposal,(left+right)/2))
    return current


class ObservedBatchPredictive:
    """Variance CDF followed by the conditional mean CDF, same raw law.

    Integrate eta and every positive component; conditional mean and observed
    variance retain their dependence. Shared identities are composed outside.
    """
    def __init__(self,field,query,replicates,order=256):
        self.n=replicates;self.k=(replicates-1)/2
        self.bias=digamma(self.k)-np.log(self.k)
        ix=[2*query,2*query+1];self.means=field.means[:,ix]
        self.covariance=field.covariances[:,ix][:,:,ix];self.weights=np.exp(field.log_weights)
        C=self.covariance;self.sd=np.sqrt(C[:,1,1]);self.loading=C[:,0,1]/self.sd
        self.conditional=np.maximum(C[:,0,0]-self.loading**2,0.)
        z,w=roots_hermitenorm(order)
        self.eta=(self.means[:,1,None]+self.sd[:,None]*z).reshape(-1)
        self.locations=(self.means[:,0,None]+self.loading[:,None]*z).reshape(-1)
        self.scales=np.sqrt(self.conditional[:,None]+np.exp(self.eta.reshape(len(self.weights),order))/replicates).reshape(-1)
        self.integration_weights=(self.weights[:,None]*w/np.sqrt(2*np.pi)).reshape(-1)
        self.log_weights=np.log(self.integration_weights)
        self.variance=np.exp(self.eta)

    def variance_coordinates(self,eta_labels):
        t=np.asarray(eta_labels).reshape(-1);cdf=[];density=[]
        for start in range(0,len(t),16):
            values=t[start:start+16,None];s=np.exp(values+self.bias)
            cdf.extend(gammainc(self.k,self.k*s/self.variance)@self.integration_weights)
            logs=self.log_weights+log_sample_variance_density(self.eta,s,self.n)
            density.extend(np.exp(logsumexp(logs,axis=1)))
        return np.asarray(cdf),np.asarray(density)

    def variance_inverse(self,probabilities):
        mean=float(self.weights@self.means[:,1])
        variance=float(self.weights@(self.covariance[:,1,1]+(self.means[:,1]-mean)**2)+polygamma(1,self.k))
        return inverse_observed_cdf(probabilities,self.variance_coordinates,mean,variance)

    def conditional_coordinates(self,values,eta_labels):
        values=np.asarray(values).reshape(-1);eta_labels=np.asarray(eta_labels).reshape(-1)
        cdf=[];density=[]
        for start in range(0,len(values),16):
            s=np.exp(eta_labels[start:start+16,None]+self.bias)
            logs=self.log_weights+log_sample_variance_density(self.eta,s,self.n)
            weights=np.exp(logs-logsumexp(logs,axis=1)[:,None])
            z=(values[start:start+16,None]-self.locations)/self.scales
            cdf.extend(np.sum(weights*ndtr(z),axis=1))
            density.extend(np.sum(weights*np.exp(-.5*z*z)/self.scales/np.sqrt(2*np.pi),axis=1))
        return np.asarray(cdf),np.asarray(density)

    def inverse(self,probabilities):
        probabilities=np.asarray(probabilities);eta=self.variance_inverse(probabilities[:,1])
        # A single envelope covers the conditional means at the used variances.
        mean=float(self.weights@self.means[:,0])
        variance=float(self.weights@(self.covariance[:,0,0]+(self.means[:,0]-mean)**2
            +np.exp(self.means[:,1]+self.covariance[:,1,1]/2)/self.n))
        values=np.empty(len(eta))
        for start in range(0,len(eta),16):
            s=np.exp(eta[start:start+16,None]+self.bias)
            logs=self.log_weights+log_sample_variance_density(self.eta,s,self.n)
            weights=np.exp(logs-logsumexp(logs,axis=1)[:,None])
            def coordinates(y):
                z=(y[:,None]-self.locations)/self.scales
                return np.sum(weights*ndtr(z),axis=1),np.sum(weights*np.exp(-.5*z*z)/self.scales/np.sqrt(2*np.pi),axis=1)
            values[start:start+16]=inverse_observed_cdf(probabilities[start:start+16,0],coordinates,mean,variance)
        return values,eta

    def log_density(self,values,eta_labels):
        parts=[]
        for start in range(0,len(values),16):
            s=np.exp(np.asarray(eta_labels[start:start+16,None])+self.bias)
            z=(np.asarray(values[start:start+16,None])-self.locations)/self.scales
            logs=self.log_weights+log_sample_variance_density(self.eta,s,self.n)
            parts.extend(logsumexp(logs-.5*z*z-np.log(self.scales)-.5*np.log(2*np.pi),axis=1))
        return np.asarray(parts)

    def adaptive_coordinates(self,values,eta_labels):
        """Independent adaptive CDF check at every used observed coordinate."""
        values=np.asarray(values);eta_labels=np.asarray(eta_labels);s=np.exp(eta_labels[:,None]+self.bias)
        reference_density=self.variance_coordinates(eta_labels)[1]
        def integrand(z):
            eta=self.means[:,1]+self.sd*z;mean=self.means[:,0]+self.loading*z
            scale=np.sqrt(self.conditional+np.exp(eta)/self.n)
            density=np.exp(log_sample_variance_density(eta,s,self.n))/reference_density[:,None]
            phi=np.exp(-.5*z*z)/np.sqrt(2*np.pi)
            return np.stack([gammainc(self.k,self.k*s/np.exp(eta))@self.weights,
                density@self.weights,(density*ndtr((values[:,None]-mean)/scale))@self.weights])*phi
        result,error,info=quad_vec(integrand,-12.,12.,points=[0.],epsabs=1e-12,epsrel=1e-12,
            norm='max',full_output=True,workers=1)
        if not info.success:raise RuntimeError(info.message)
        return result[0],result[2]/result[1],float(2*error/(min(1.,result[1].min())-error))


def predictive_observed_batches(model,query,replicates,power,seed):
    """Full-bank entry: one mean/variance pair per head and complete identity."""
    if replicates==1:
        from core.positive_mixture_lookahead import predictive_values
        values,weights=predictive_values(model,query,power,seed);return values,None,weights
    from scipy.stats import qmc
    nodes=qmc.Sobol(4,scramble=True,seed=seed).random_base2(power);values=[];etas=[]
    for pair in model.components:
        heads=[];variances=[]
        for head,field in enumerate(pair):
            probabilities=nodes[:,2*head:2*head+2]
            for order in (64,128,256):
                rule=ObservedBatchPredictive(field,query,replicates,order);y,t=rule.inverse(probabilities)
                variance_cdf,mean_cdf,bound=rule.adaptive_coordinates(y,t)
                error=max(np.max(abs(variance_cdf-probabilities[:,1])),np.max(abs(mean_cdf-probabilities[:,0])))
                if error<=1e-8 and bound<=1e-9:break
            if error>1e-8 or bound>1e-9:raise RuntimeError('Observed batch CDF did not pass its original precision gates')
            heads.append(y);variances.append(t)
        values.extend(zip(*heads));etas.extend(zip(*variances))
    return np.asarray(values),np.asarray(etas),np.repeat(model.weights/len(nodes),len(nodes))
