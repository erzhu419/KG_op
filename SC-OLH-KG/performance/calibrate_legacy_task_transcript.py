"""Calibrate the whole source observation transcript, including paid evidence."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.calibrate_legacy_task_transfer import mean_density, run

EXECUTION = 'performance/manifests/legacy_joint_population_task_transcript_calibration_execution_v1.json'


def transcript_loss(predictions, branch):
    paid = predictions['raw_paid']; count = len(branch['sites'])
    return -(paid['paid_predictive_log_score']+count*mean_density(paid, branch))/(count+len(branch['paid_sites']))


if __name__ == '__main__':
    run(EXECUTION, transcript_loss)
