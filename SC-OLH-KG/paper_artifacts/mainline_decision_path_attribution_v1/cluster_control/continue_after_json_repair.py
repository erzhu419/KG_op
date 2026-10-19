"""Continue only the failed zero-call audit after its recorded JSON output repair."""
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[4]/'scripts'))
from run_mainline_decision_path_cluster import AUDIT, Controller, base


def main():
    control = Controller()
    try:
        validation = str(Path(control.plan['output_dir'])/'cluster_control/run_stage77_recovery_attempt2.py')
        control.job('validation/attempt2', 1, [base.PYTHON, '-u', validation])
        control.job('audit/tasks/attempt2', 24, base.module(AUDIT, 'batch', '--manifest', control.manifest,
            '--start', '{start}', '--end', '{end}', '--workers', '{workers}'))
        control.job('audit/collect', 1, base.module(AUDIT, 'collect', '--manifest', control.manifest))
        control.update(status='complete', waiting_for=None, error=None)
    except Exception as error:
        control.update(status='blocked', error=str(error))
        raise


if __name__ == '__main__':
    main()
