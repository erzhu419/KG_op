"""Read-only source support and covariance audit on the frozen common pool."""
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from baselines.transfer_archive import FrozenTransferArchive
from performance.mainline_legacy_source_models import _query_design
from performance.run_mainline_legacy_backend_bridge import load_pickle
from representation.legacy_function_prior import legacy_public_coordinates, legacy_source_labels
from representation.legacy_observable_prior import legacy_exposure_descriptors
from representation.observable_exposure import observable_state_descriptor_names
from representation.source_kernel_population import rbf_cross, shared_kernel

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / 'paper_artifacts/mainline_legacy_backend_bridge_v1'
OUTPUT = ROOT / 'paper_artifacts/mainline_factorshock_diagnosis_v1'
FAMILY = 'FactorShockStatePolicyRZDT1'


def spectrum(covariance):
    eigen = np.linalg.eigvalsh((covariance+covariance.T)/2)
    positive = eigen[eigen > max(1e-12, max(eigen)*1e-10)]
    return {'rank': len(positive), 'dimension': len(eigen),
            'effective_rank': float(positive.sum()**2/(positive@positive)) if len(positive) else 0.,
            'largest_eigenvalue': float(eigen[-1]), 'trace': float(eigen.sum())}


def source_support(source, target):
    distances = cdist(target, source)
    near = distances.argmin(axis=1)
    source_distances = cdist(source, source)
    np.fill_diagonal(source_distances, np.inf)
    distinct_source_distances = source_distances.copy()
    distinct_source_distances[distinct_source_distances < 1e-10] = np.inf
    target_distances = cdist(target, target)
    np.fill_diagonal(target_distances, np.inf)
    return {'source_unique_coordinate_count': len(np.unique(source,axis=0)),
            'source_axis_min': source.min(axis=0).tolist(),
            'source_axis_max': source.max(axis=0).tolist(),
            'target_axis_min': target.min(axis=0).tolist(),
            'target_axis_max': target.max(axis=0).tolist(),
            'outside_source_box_axis_count': np.sum((target < source.min(axis=0)) | (target > source.max(axis=0)),axis=1).tolist(),
            'nearest_source_indices': near.tolist(),
            'nearest_source_distance': distances[np.arange(len(target)),near].tolist(),
            'source_nearest_distance_quantiles': np.quantile(source_distances.min(axis=1),[0,.5,.95,1]).tolist(),
            'source_nearest_distinct_distance_quantiles': np.quantile(distinct_source_distances.min(axis=1),[0,.5,.95,1]).tolist(),
            'target_nearest_other_distance': target_distances.min(axis=1).tolist(),
            'exact_target_coordinate_alias_pairs': np.argwhere(np.triu(target_distances < 1e-10,1)).tolist()}


def run():
    started = time.process_time()
    inputs = json.loads((OUTPUT/'inputs.json').read_text())
    bank = np.asarray(inputs['bank'])
    bundle = load_pickle(BRIDGE/'source_models'/FAMILY/'source_models.pkl')
    engine = load_pickle(BRIDGE/FAMILY/'same_source_gp/search_engine.pkl')
    mean = (legacy_public_coordinates(bank/engine.problem.L)-bundle['mean_center'])/bundle['mean_scale']
    descriptors=legacy_exposure_descriptors(engine.problem,bank,50)
    risk = (bundle['encoder'].transform(descriptors)
            -bundle['risk_center'])/bundle['risk_scale']
    archive = FrozenTransferArchive.load(ROOT/'paper_artifacts/legacy_joint_bridge_v1/inputs'/FAMILY/'archive.json')
    labels = legacy_source_labels(archive)
    response = np.vstack([row['response'] for row in labels])
    source_mean = np.asarray(bundle['gp']['heads'][1]['joint_common']['source_features'])
    source_risk = np.asarray(bundle['gp']['heads'][2]['joint_common']['source_features'])
    mean_support=source_support(source_mean,mean)
    nearest_source_labels=[{'bank_index':i,'source_site':j%64,
        'source_response_rows':[{'task':row['name'],'f_g_eta_g_eta_f':row['response'][j%64].tolist()}
                                for row in labels]}
        for i,j in enumerate(mean_support['nearest_source_indices'])]
    coordinate=bundle['encoder'].g_mean
    names=observable_state_descriptor_names()
    library_names=['signed_log1p_'+name for name in names]+['log1p_squared_'+name for name in names]
    library_audit=[]
    for i,descriptor in enumerate(descriptors):
        library=coordinate._input_library(descriptor)
        standardized=(library-coordinate.feature_mean)/coordinate.feature_scale
        indices=np.argsort(abs(standardized))[-3:][::-1]
        library_audit.append({'bank_index':i,'largest_standardized_source_library_coordinates':[
            {'library_index':int(j),'name':library_names[j], 'standardized_value':float(standardized[j]),
             'source_mean':float(coordinate.feature_mean[j]),'source_scale':float(coordinate.feature_scale[j]),
             'target_value':float(library[j])} for j in indices]})
    gp = []
    for head,record in enumerate(bundle['gp']['heads']):
        fit = record['joint_common']; anchor = fit['anchor']
        target = mean if head < 2 else risk
        source = np.asarray(fit['source_features'])
        cross = shared_kernel(target,source,anchor)
        scale = anchor['normalization']['scale']
        common = scale**2*(shared_kernel(target,target,anchor)-cross@np.asarray(fit['common_precision'])@cross.T)
        amplitude,intercept,white = fit['residual_parameters']
        task = scale**2*(amplitude*rbf_cross(target,target,anchor['lengthscale'])+intercept)
        total = common+task+scale**2*white*np.eye(len(target))
        prior_mean = anchor['normalization']['mean']+scale*(anchor['constant']+cross@np.asarray(fit['common_alpha']))
        gp.append({'head': ['f','g','eta_g','eta_f'][head],
                   'source_response_range': [float(response[:,head].min()),float(response[:,head].max())],
                   'source_response_scale': scale,'lengthscale': anchor['lengthscale'],
                   'common_kernel_variance': float(scale**2*anchor['signal_variance']),
                   'task_kernel_variance': float(scale**2*amplitude),
                   'task_intercept_variance': float(scale**2*intercept),'white_variance':float(scale**2*white),
                   'maximum_source_rbf_correlation': rbf_cross(target,source,anchor['lengthscale']).max(axis=1).tolist(),
                   'prior_mean': prior_mean.tolist(),'common_posterior_variance':np.diag(common).tolist(),
                   'new_task_variance': np.diag(task).tolist(),'total_prior_covariance':spectrum(total)})
    structural=[]
    for identity,pair,weight in zip(bundle['identities'],bundle['components'],np.exp(bundle['log_weights'])):
        row={'identity':identity,'source_weight':float(weight),'heads':[]}
        for head,record in enumerate(pair):
            law,risk_law=record['law'],record['risk']
            D=_query_design(bundle,identity,head,(mean,risk),law)
            E=_query_design(bundle,identity,3 if head==0 else 2,(mean,risk),risk_law)
            weights=np.exp(record['log_weights'])
            component_means=record['mean']@D.T
            mix_mean=weights@component_means
            centered=component_means-mix_mean
            common=D@np.einsum('k,kij->ij',weights,record['covariance'])@D.T
            between=centered.T@(weights[:,None]*centered)
            task=D@law['task']@D.T
            total=common+between+task+law['white']*np.eye(len(bank))
            eta_var=np.diag(E@(record['eta_covariance']+risk_law['task'])@E.T)+risk_law['white']
            eta_means=record['eta_mean']@E.T
            noise=weights@np.exp(eta_means+eta_var[None,:]/2)
            row['heads'].append({'head':['f','g'][head], 'design_dimension':D.shape[1],
                'source_task_coefficient_covariance':spectrum(law['task']),
                'source_common_coefficient_covariance':spectrum(law['common']),
                'prior_mean':mix_mean.tolist(),'common_posterior_variance':np.diag(common).tolist(),
                'source_mixture_between_mean_variance':np.diag(between).tolist(),
                'new_task_variance':np.diag(task).tolist(),'white_variance':float(law['white']),
                'total_prior_covariance':spectrum(total),'episode_variance':noise.tolist()})
        structural.append(row)
    encoder_diagnostics=bundle['encoder'].diagnostics()
    result={'source_bundle':'mainline_legacy_backend_bridge_v1/source_models/'+FAMILY+'/source_models.pkl',
            'bank_input':'inputs.json','bank_count':len(bank),'source_tasks':bundle['diagnostics']['source_task_names'],
            'source_variance_coordinate_rank_correlations': [v['source_variance_rank_correlation'] for v in encoder_diagnostics['variance_maps']],
            'mean_coordinates':mean_support,'risk_coordinates':source_support(source_risk,risk),
            'nearest_public_coordinate_source_responses':nearest_source_labels,
            'observable_library_extrapolation':library_audit,
            'gp':gp,'structural':structural,
            'cost':{'cpu_seconds':time.process_time()-started,'new_source_fits':0,'new_source_simulations':0,
                    'new_target_simulations':0,'truth_queries':0,'remote_downloads':0}}
    (OUTPUT/'source_geometry.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'source_geometry_complete':True,'cost':result['cost']}))


if __name__=='__main__':
    run()
