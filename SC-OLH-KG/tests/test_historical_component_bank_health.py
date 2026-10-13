import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from performance import launch_historical_component_bank as bank


class LockBusy(TimeoutError):
    pass


class ComponentBankHealthTest(unittest.TestCase):
    def test_busy_lock_preserves_worker_state_then_refreshes_on_next_poll(self):
        task={'id':'t1','status':'running'}
        scheduler=Mock(SchedulerLockTimeout=LockBusy)
        scheduler.load_state.return_value={'tasks':[task]}
        attempts=iter([True,False])

        @contextlib.contextmanager
        def lock(**kwargs):
            if next(attempts):
                raise LockBusy('scheduler state lock busy')
            yield

        scheduler.state_lock.side_effect=lock
        scheduler._batch_check_running.side_effect=lambda state,**kwargs: state['tasks'][0].update(status='done')
        with tempfile.TemporaryDirectory() as tmp:
            queue=Path(tmp)/'queue.json'
            queue.write_text(json.dumps({'tasks':[task]}))
            with patch.object(bank,'QUEUE',queue),patch.dict(sys.modules,{'scheduler':scheduler}),contextlib.redirect_stdout(io.StringIO()):
                bank.refresh_jobs({'t1'})
                self.assertEqual(task['status'],'running')
                scheduler.save_state.assert_not_called()
                bank.refresh_jobs({'t1'})
        self.assertEqual(task['status'],'done')
        scheduler.save_state.assert_called_once_with({'tasks':[task]})

    def test_other_refresh_errors_still_stop_the_controller(self):
        scheduler=Mock(SchedulerLockTimeout=LockBusy)
        scheduler._running_probe_snapshot_deps.side_effect=RuntimeError('probe failed')
        with tempfile.TemporaryDirectory() as tmp:
            queue=Path(tmp)/'queue.json'
            queue.write_text(json.dumps({'tasks':[{'id':'t1','status':'running'}]}))
            with patch.object(bank,'QUEUE',queue),patch.dict(sys.modules,{'scheduler':scheduler}):
                with self.assertRaisesRegex(RuntimeError,'probe failed'):
                    bank.refresh_jobs({'t1'})


if __name__=='__main__':
    unittest.main()
