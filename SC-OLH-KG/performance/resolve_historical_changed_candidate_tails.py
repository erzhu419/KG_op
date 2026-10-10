"""Historical variance tilting plus a bounded old/new source bridge."""

import signal
import time

import numpy as np
from scipy.special import logsumexp

from performance.complete_historical_source_heads import ROOT,IDENTITY,read,save
from performance.recondition_historical_changed_candidates import candidate_inputs
from performance.resolve_historical_source_variance import VarianceTiltedPosterior,original_posterior_summary,weighted_moments
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior
from performance.check_legacy_f_collapsed_source_raw_update import sample
from performance.check_legacy_f_correlated_source_raw_update import diagnostics
from performance.historical_source_block_transition import source_block_transition
from performance.refine_historical_raw_lookahead import cpu_timeout

PLAN='performance/manifests/historical_changed_candidate_tail_resolution_v1.json'


class PrecisionTiltedPosterior(VarianceTiltedPosterior):
    def __init__(self,old,relative,old_log_tilt,sites,center,scale,log_ratio_scale):
        super().__init__(relative.mean,relative.covariance,relative.labels,relative.s2,sites,center,scale)
        self.old,self.relative,self.old_log_tilt=old,relative,old_log_tilt
        self.log_ratio_scale=log_ratio_scale

    def reverse_ratio(self,eta):
        evaluation=self.old.evaluate(eta)
        return evaluation[0]-self.old_log_tilt(evaluation)-self.relative.evaluate(eta)[0]

    def log_tilt_at(self,eta,evaluation=None):
        base=VarianceReferencePosterior.evaluate(self,eta) if evaluation is None else evaluation
        variance=VarianceTiltedPosterior.log_tilt(self,base)
        return float(np.logaddexp(variance,self.reverse_ratio(eta)-self.log_ratio_scale))

    def evaluate(self,eta):
        base=VarianceReferencePosterior.evaluate(self,eta)
        return base[0]+self.log_tilt_at(eta,base),base[1],base[2]


def register():
    if (ROOT/PLAN).exists():return read(PLAN)
    parent=read('performance/manifests/historical_changed_candidate_reconditioning_v1.json')
    result=read(parent['output']+'/results.json');cases=[]
    if not result['all_registered_cases_complete']:raise RuntimeError('Complete fresh candidates first')
    for row in result['rows']:
        if row['new_source_posterior_precision_passed']:continue
        sites=np.argsort(row['half_variance_relative_differences'])[-3:][::-1]
        cases.append({**{k:row[k] for k in (*IDENTITY,'head','source_case')},'tilt_sites':sites.tolist(),
            'tilt_center':row['posterior_mean'],'tilt_scale':np.diag(row['posterior_covariance']).tolist(),
            'log_ratio_scale':-row['log_source_evidence_ratio_new_vs_old']})
    combined=ROOT/parent['output']/'complete_posterior_precision.json'
    save(combined,{'rows':result['rows']+result['reused_passed_cases']})
    spec={**parent,'cases':cases,'status':'registered before the fixed variance/evidence bridge',
        'previous_posterior_precision':str(combined.relative_to(ROOT)),
        'sampler':{**parent['sampler'],'retained_sweeps_per_chain':2048,'seed':202610090},
        'maximum_single_thread_cpu_seconds':90,
        'scope':'Only four remaining fixed new candidates;28 PASS cases reused. Fixed h=1+mean(centered conditional f second moment/scale)+exp(log pi_old/pi_new-log_ratio_scale). Historical compensated elliptical sampler with the four actual source-task blocks and three joint source risk directions. Inverse-h weights restore the new law;both bridge weights are bounded. Frozen source-only sites and scales,no seed search or target reads. Original32 passed source cases are untouched.',
        'output':'paper_artifacts/historical_changed_candidate_tail_resolution_v1'}
    save(ROOT/PLAN,spec);return spec


def run(plan=PLAN):
    spec=register() if plan==PLAN else read(plan)
    out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True);path=out/'results.json'
    previous=read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_registered_cases_complete'):return previous
    parent=read(spec['source_plan']);laws=read(parent['source_laws'])['rows'];selection=read(parent['selection_plan'])
    learning=read(spec['learning_results']);baseline=read(spec['previous_posterior_precision']);rows=previous.get('rows',[])
    started=time.process_time()
    for item in spec['cases']:
        if any(all(r[k]==item[k] for k in (*IDENTITY,'head')) for r in rows):continue
        case=next(r for r in learning['evidence_rows'] if all(r[k]==item[k] for k in (*IDENTITY,'head')))
        one=time.process_time();handler=signal.signal(signal.SIGPROF,cpu_timeout)
        signal.setitimer(signal.ITIMER_PROF,spec['maximum_single_thread_cpu_seconds']);row={k:item[k] for k in (*IDENTITY,'head','source_case')}
        try:
            old,relative,old_tilt,field,reference=candidate_inputs(case,laws,selection)
            model=PrecisionTiltedPosterior(old,relative,old_tilt,item['tilt_sites'],item['tilt_center'],item['tilt_scale'],item['log_ratio_scale'])
            rm,rc=reference['reference_mean'],reference['reference_covariance']
            transition,check=source_block_transition(rm,rc,item['tilt_sites'])
            seed=spec['sampler']['seed']+1000*(case['selector']=='additive')+10*(case['support']=='joint')+10000*(case['head']=='g')
            paired={**spec,'sampler':{**spec['sampler'],'seed':seed}};eta=[];moments=[]
            sample(field,0,paired,one,model,rm,rc,eta,transition,moments)
            states=eta[0];ds,cs=moments[0];flat=states.reshape(-1,48)
            if 'prefix_component_file' in item:
                prefix=read(item['prefix_component_file'])['positive_eta_mixture']['eta_states_by_chain']
                np.testing.assert_allclose(states[:,:2048:8],prefix,atol=1e-10,rtol=0.)
            logh=np.asarray([model.log_tilt_at(p) for p in flat]).reshape(states.shape[:2])
            reverse=np.asarray([model.reverse_ratio(p) for p in flat]).reshape(logh.shape)
            weights=np.exp(-logh);summary=original_posterior_summary(model,states,ds,cs,weights,spec)
            w=weights/weights.sum();count=states.shape[1];stride=count//256
            kept=states[:,::stride];kw=w[:,::stride];kw/=kw.sum()
            from performance.resolve_historical_source_variance import weighted_function_moments
            delta,cov=weighted_function_moments(ds[:,::stride],cs[:,::stride],kw)
            variance=np.diag(summary['posterior_covariance'])
            ve=float(np.max(abs(np.diag(cov)-variance)/variance))
            me=float(np.sqrt(np.mean((model.mean+delta-summary['posterior_mean'])**2/variance)))
            retain=bool(ve<=spec['retained_mixture']['maximum_source_f_diagonal_variance_relative_difference_vs_all_draws']
                and me<=spec['retained_mixture']['maximum_source_f_mean_rms_difference_in_full_posterior_sd'])
            old_weights=np.exp(reverse-logh-item['log_ratio_scale'])
            logratio=float(-item['log_ratio_scale']+np.log(weights.sum())-np.log(old_weights.sum()))
            halves=[float(-item['log_ratio_scale']+np.log(weights[:,s].sum())-np.log(old_weights[:,s].sum()))
                for s in (slice(0,count//2),slice(count//2,None))]
            influence=old_weights/old_weights.mean()-weights/weights.mean()
            diagnostic=diagnostics(influence[...,None]);gate=spec['monte_carlo_gate']
            ess=float(old_weights.sum()**2/np.sum(old_weights**2));gap=abs(halves[0]-halves[1])
            evidence=bool(ess>=spec['evidence_precision']['minimum_importance_ESS']
                and gap<=spec['evidence_precision']['maximum_split_log_ratio_difference']
                and diagnostic['split_rhat'][0]<=gate['maximum_split_rhat_eta_and_rao_blackwell_f_moments']
                and diagnostic['autocorrelation_ess'][0]>=gate['minimum_autocorrelation_ess_eta_and_rao_blackwell_f_moments'])
            passed=bool(summary['monte_carlo_gate_passed'] and retain and evidence)
            component=out/('candidate_'+'_'.join(case[k] for k in (*IDENTITY,'head'))+'.json')
            save(component,{'execution_plan':plan,'fixed_candidate':{k:case[k] for k in (*IDENTITY,'head',
                    'f_common_task_scales','eta_common_task_scales')},
                'posterior':summary,'positive_eta_mixture':{'eta_states_by_chain':kept.tolist(),
                    'positive_component_weights':kw.reshape(-1).tolist(),'weights_restore_new_candidate_law':True}})
            row.update(**summary,new_source_posterior_precision_passed=passed,posterior_functional_precision_passed=summary['monte_carlo_gate_passed'],
                point_mixture_retention_passed=retain,source_posterior_and_retention_passed=passed,evidence_ratio_precision_passed=evidence,
                log_source_evidence_ratio_new_vs_old=logratio,split_log_ratio_difference=gap,reverse_importance_ESS=ess,
                reverse_ratio_rhat=diagnostic['split_rhat'][0],reverse_ratio_autocorrelation_ESS=diagnostic['autocorrelation_ess'][0],
                point_mixture_maximum_f_variance_relative_error=ve,point_mixture_mean_error=me,
                candidate_component_file=str(component.relative_to(ROOT)),source_block_check=check,
                new_candidate_MCMC_sweeps=4*(spec['sampler']['warmup_sweeps_per_chain']+count))
        except TimeoutError:
            row.update(new_source_posterior_precision_passed=False,source_posterior_and_retention_passed=False,
                numerical_failure='registered90CPU second tilted-candidate cap')
        finally:
            signal.setitimer(signal.ITIMER_PROF,0.);signal.signal(signal.SIGPROF,handler)
        row['cpu_seconds']=time.process_time()-one;rows.append(row)
        save(path,{'execution_plan':plan,'rows':rows,'all_registered_cases_complete':False})
        print('tilted candidate',*[row[k] for k in (*IDENTITY,'head')],'pass',row['new_source_posterior_precision_passed'],
            'variance split',round(row.get('maximum_first_second_half_f_variance_relative_difference',-1),4),
            'evidence',row.get('evidence_ratio_precision_passed'),'cpu',round(row['cpu_seconds'],2),flush=True)
    reused=[r for r in baseline['rows'] if r['new_source_posterior_precision_passed']]
    result={'execution_plan':plan,'rows':rows,'reused_passed_cases':reused,'all_registered_cases_complete':len(rows)==len(spec['cases']),
        'posterior_precision_cases_passed':len(reused)+sum(r['new_source_posterior_precision_passed'] for r in rows),
        'cpu_seconds':time.process_time()-started,'cost':spec['cost'],'new_candidate_MCMC_sweeps':sum(r.get('new_candidate_MCMC_sweeps',0) for r in rows),
        'original32_passed_source_cases_unchanged':True,'no_model_adopted':True,'independent_confirmation':False}
    save(path,result);return result


if __name__=='__main__':run()
