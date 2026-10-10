"""Check the frozen source-learned shape budget on cached outer task1."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_predictive_shape_pilot import execute
from performance.check_legacy_public_mean_risk_uncertainty import read

EXECUTION = 'performance/manifests/legacy_joint_population_f_predictive_shape_scale_pilot_execution_v1.json'


def run():
    spec = read(EXECUTION)
    fitted, package = read(spec['reference_results']), read(spec['source_rules'])
    assert fitted['all_source_predictive_fits_and_rules_complete']
    assert not fitted['outer_task1_query_outcomes_used']
    assert package['both_source5_laws_frozen_without_task1_values']
    assert len(package['rows']) == len(spec['layouts']) == 2
    execute(spec, package)


if __name__ == '__main__':
    run()
