"""Freeze real d1000 public coordinates and audit dimension/response units.

Only public controls, simulator configuration and frozen source records enter
this audit. No nominal target outcomes or new simulator calls are requested.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.mainline_cross_dimension_prior import query_coordinates
from representation.legacy_observable_prior import legacy_exposure_descriptors
from representation.public_control_mean import public_control_mean_coordinates


OUTPUT = ROOT/'paper_artifacts/mainline_d1000_two_round_v1'


def read(path):
    path = Path(path)
    return json.loads((path if path.is_absolute() else ROOT/path).read_text())


def save(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')


def difference_summary(before, after):
    delta = np.asarray(after)-np.asarray(before)
    return {'policy_count': len(delta), 'coordinate_count': delta.shape[1],
        'changed_policy_count_at_1e_12': int(np.sum(np.any(abs(delta)>1e-12, axis=1))),
        'max_absolute_delta': float(np.max(abs(delta))),
        'max_absolute_delta_by_coordinate': np.max(abs(delta), axis=0).tolist(),
        'median_absolute_delta_by_coordinate': np.median(abs(delta), axis=0).tolist()}


def run(bank_path=OUTPUT/'original_bank.json', output=OUTPUT, layout='5'):
    cpu, wall = time.process_time(), time.perf_counter()
    bank_path = Path(bank_path)
    if not bank_path.is_absolute():
        bank_path = ROOT/bank_path
    original = read(bank_path)
    lineage = original['lineage']
    legacy = read(lineage['legacy_plan'])
    public = read(lineage['source50_bank_path'])
    source_config = public['plan']['task_distribution']
    target_config = {key: legacy[key] for key in ('L', 'sigma', 'alpha', 'weights')}
    source_parameters = {key: source_config[key] for key in target_config}
    # A mismatch changes response/noise units and must be resolved before reuse.
    if source_parameters != target_config:
        raise ValueError('Source and original target response parameters differ')
    bank = np.asarray(original['bank'], dtype=int)
    source_bank = np.asarray(original['source50_bank'], dtype=int)
    if bank.shape != (73, 1000) or source_bank.shape != (73, 50):
        raise ValueError('This audit requires the original paired 73-policy d1000/d50 bank')
    if legacy['target_dimension'] != 1000 or not lineage['downsampled_index_identity']:
        raise ValueError('The bank must retain original d1000 policy lineage and ordering')
    providers = {d: no_outcomes(build_scalarized_problem(
        'InventorySupplyChain', d, **target_config)) for d in (50, 1000)}
    output = Path(output)
    coordinates = query_coordinates(layout, providers[1000], bank.tolist(),
        output_path=output/'query_coordinates.json')
    diagnostics = coordinates['diagnostics']
    mean_source = read(diagnostics['source_mean_inputs'])
    risk_source = read(diagnostics['source_risk_inputs'])
    raw_source = public_control_mean_coordinates(providers[50], source_bank)
    raw_target = public_control_mean_coordinates(providers[1000], bank)
    source_desc = legacy_exposure_descriptors(providers[50], source_bank, 50)
    target_desc = legacy_exposure_descriptors(providers[1000], bank, 50)
    representation = {
        **diagnostics,
        'original_bank': str(Path(bank_path).relative_to(ROOT)),
        'raw_mean_names': ['stock_control_mean', 'reorder_control_mean',
                           'safety_control_mean', 'policy_sd'],
        'raw_mean_dimension_shift': difference_summary(raw_source, raw_target),
        'standardized_mean_dimension_shift': difference_summary(
            mean_source['public_mean_reference'], coordinates['mean']),
        'risk_dimension_shift': difference_summary(
            risk_source['public_reference_psi'], coordinates['risk']),
        'common_grid_descriptor_dimension_shift': difference_summary(source_desc, target_desc),
        'mean_source_center': mean_source['mean_coordinate_center'],
        'mean_source_scale': mean_source['mean_coordinate_scale'],
        'dynamics_grid_dimension': 50,
        'descriptor_rule': 'Actual policy channel means/scales and occupancy; only curve dynamics resampled to the frozen 50-point grid.',
        'coordinate_rule': 'Fresh actual-bank mean4 and frozen-source-encoder risk10; no reused d50 rows and no legacy_public_coordinates substitution.',
        'reference_policy_deltas': {str(i): {
            'raw_mean_delta': (raw_target[i]-raw_source[i]).tolist(),
            'standardized_mean_delta': (np.asarray(coordinates['mean'])[i]-np.asarray(mean_source['public_mean_reference'])[i]).tolist(),
            'risk_delta': (np.asarray(coordinates['risk'])[i]-np.asarray(risk_source['public_reference_psi'])[i]).tolist(),
        } for i in (10, 45, 63, 71)},
    }
    audit = {
        'source_dimension': 50, 'target_dimension': 1000,
        'source_public_parameters': source_parameters,
        'target_public_parameters': target_config,
        'source_tau': float(providers[50].tau), 'target_tau': float(providers[1000].tau),
        'source_channel_sizes': [len(a) for a in np.array_split(np.arange(50), 3)],
        'target_channel_sizes': [len(a) for a in np.array_split(np.arange(1000), 3)],
        'response_scale_mapping': {
            'f_multiplier': 1.0, 'f_offset': 0.0,
            'g_minus_tau_multiplier': 1.0, 'g_minus_tau_offset': 0.0,
            'sigma_f_multiplier': 1.0, 'sigma_g_multiplier': 1.0,
            'eta_f_offset': 0.0, 'eta_g_offset': 0.0,
            'basis': 'No explicit dimension multiplier in Inventory mean or noise laws. At fixed task parameters, identical three channel means and policy SD imply identical response distributions.',
            'interpretation': 'Identity maps units, not policy values. A paired d50/d1000 policy can have changed summary inputs and hence different responses.',
            'source_geometry': 'The frozen training tasks have their existing geometry shifts; the target remains original nominal geometry. No geometry or target outcomes fit this scale mapping.',
            'scalarized_noise': 'sqrt(w0**2+w1**2) times the common base objective sigma; constraint sigma is unchanged by objective scalarization.',
            'log_variance_rule': 'eta = log(sigma**2); any public response multiplier a would add 2*log(abs(a)), not multiply eta. Here a=1.',
        },
        'static_formula_evidence': {
            'public_summary': 'problems/rzdt.py:1019-1037',
            'objective_and_constraint': 'problems/rzdt.py:1040-1064',
            'risk_and_noise': 'problems/rzdt.py:1074-1135',
            'scalarization': 'problems/single_objective.py:52-71',
            'observed_source_log_variance': 'representation/legacy_function_prior.py:17-33',
        },
        'scientific_scope': '1000 control coordinates, with mean and noise laws determined by four public summary statistics; this case tests high-dimensional input transfer with low-dimensional sufficient statistics.',
        'representation_audit': representation,
        'target_outcomes_used': 0, 'target_scoring_truth_queries': 0,
        'source_scoring_truth_queries': 0, 'new_simulator_calls': 0,
        'new_verifier_calls': 0, 'paired_outcome_probes': 0,
        'source_law_optimizations': 0, 'gp_optimizations': 0,
        'cpu_seconds': time.process_time()-cpu,
        'wall_seconds': time.perf_counter()-wall,
    }
    save(output/'bridge_inputs.json', {'bank': bank.tolist(),
        'mean_coordinates': coordinates['mean'], 'risk_coordinates': coordinates['risk'],
        'representation_audit': representation})
    save(output/'dimension_audit.json', audit)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', type=Path, default=OUTPUT/'original_bank.json')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--layout', default='5')
    args = parser.parse_args()
    result = run(args.bank, args.output, args.layout)
    print(json.dumps({key: result[key] for key in ('source_dimension', 'target_dimension',
        'new_simulator_calls', 'target_scoring_truth_queries', 'cpu_seconds', 'wall_seconds')}))
