"""Recover the original public d1000 bank without outcomes or simulation."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.transfer_archive import resample_normalized_profiles
from core.designs import load_frozen_source_informed_design


LEGACY_PLAN = 'performance/manifests/legacy_joint_bridge_v1.json'
INPUTS = 'paper_artifacts/legacy_joint_bridge_v1/inputs/InventorySupplyChain'
SOURCE50_INPUTS = 'paper_artifacts/legacy_related_inventory_source_v1/frozen_inputs.json'
OUTPUT = 'paper_artifacts/mainline_d1000_two_round_v1/original_bank.json'


def _read(relative):
    return json.loads((ROOT / relative).read_text())


def _integer_policies(profiles, level_count):
    """The existing Inventory [0,L] continuous_to_int rounding rule."""
    values = np.clip(np.round(np.asarray(profiles) * level_count), 0, level_count)
    return [tuple(map(int, row)) for row in values]


def reconstruct_original_bank():
    """Return all original policies in order and their public construction lineage."""
    plan = _read(LEGACY_PLAN)
    archive = _read(f'{INPUTS}/archive.json')
    initial, _ = load_frozen_source_informed_design(
        ROOT / INPUTS / 'design.json', heldout='InventorySupplyChain',
        seed=plan['target_seed'], n0=plan['n0'], dimension=plan['target_dimension'])
    profiles = np.vstack([task['X'] for task in archive['tasks']])
    lifted = resample_normalized_profiles(profiles, plan['target_dimension'])
    bank = list(dict.fromkeys(list(initial) + _integer_policies(lifted, plan['L'])))
    source50 = [tuple(map(int, row)) for row in _read(SOURCE50_INPUTS)['bank']]
    downsampled = _integer_policies(resample_normalized_profiles(
        np.asarray(bank) / plan['L'], plan['source_dimension']), plan['L'])
    if len(bank) != 73 or any(len(point) != 1000 for point in bank):
        raise ValueError('Original Inventory support must contain 73 d1000 policies')
    if downsampled != source50 or len(set(downsampled)) != 73:
        raise ValueError('Original d1000 policies no longer match source50 bank indices')
    lineage = {
        'construction': 'ten frozen initial policies plus all interpolated source profiles; stable deduplication',
        'legacy_plan': LEGACY_PLAN,
        'design_path': f'{INPUTS}/design.json',
        'archive_path': f'{INPUTS}/archive.json',
        'source50_bank_path': SOURCE50_INPUTS,
        'design_seed': plan['target_seed'],
        'archive_source_seed': archive['source_seed'],
        'initial_policy_count': len(initial),
        'source_profile_count': len(profiles),
        'source_domains': [task['name'] for task in archive['tasks']],
        'original_policy_count': len(bank),
        'target_dimension': plan['target_dimension'],
        'source_dimension': plan['source_dimension'],
        'L': plan['L'],
        'downsampled_index_identity': True,
        'target_outcomes_used': False,
        'simulation_calls': 0,
        'new_source_fits': 0,
        'reverse_lift_from_source50': False,
    }
    return bank, lineage


def main():
    bank, lineage = reconstruct_original_bank()
    source50 = _integer_policies(resample_normalized_profiles(
        np.asarray(bank) / lineage['L'], lineage['source_dimension']), lineage['L'])
    path = ROOT / OUTPUT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'bank': bank, 'source50_bank': source50, 'lineage': lineage},
                               separators=(',', ':')) + '\n')
    print(json.dumps({'output': OUTPUT, 'bytes': path.stat().st_size, **lineage}))


if __name__ == '__main__':
    main()
