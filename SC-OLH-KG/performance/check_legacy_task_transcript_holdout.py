"""Original source gate for the fixed transcript-calibrated meta-prior."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_task_transfer_holdout import run


if __name__ == '__main__':
    run('performance/manifests/legacy_joint_population_task_transcript_holdout_execution_v1.json')
