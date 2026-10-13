import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from performance import probe_historical_low_gauss_reference as pilot
from performance.diagnose_historical_component_guarded import compressed_write,compressed_read
from performance import screen_historical_observed_batch_risk as screen


class LowGaussPrepareTest(unittest.TestCase):
    def test_low_gauss_pricing_retains_primary_rules_and_uses_historical_cost_floor(self):
        rows={name:dict(cpu_seconds=value) for name,value in
            [('E8',1.),('F8',1.),('H8',2.),('E9',2.),('F9',2.),('K9',4.)]}
        low={'E8':['E8','F8','L8','E9'],'E9':['E8','E9','F9','L9']}
        groups=[dict(group=0,kind='guarded')];row=dict(record='r',kind='raw',group=0)
        with patch.object(screen,'read',return_value=dict(rows=rows)):
            for fine,expected,candidate in [(set(),7.7,'E8'),({(0,'raw')},9.8,'E9')]:
                selected,jobs=screen.price_reference_group(row,groups,[],fine,low)
                self.assertEqual(selected['candidate'],candidate)
                self.assertEqual(selected['rules'],low[candidate])
                self.assertAlmostEqual(jobs[0]['estimated_cpu_seconds'],expected)
                old,original=screen.price_reference_group(row,groups,[],fine)
                self.assertEqual(old['candidate'],candidate)
                self.assertLess(jobs[0]['estimated_cpu_seconds'],original[0]['estimated_cpu_seconds'])

    def test_current_bank_paths_and_cached_cpu_are_preserved_for_new_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def save(path,data):
                p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data))
            def read(path):return json.loads((root/path).read_text())
            jobs=[dict(identity=h,estimated_cpu_seconds=20) for h in (4,5)]
            save('plan',dict(output='pilot',source_bound_plan='bound',previous_phase_completion='completion',
                maximum_estimated_cpu_seconds=3600,maximum_worker_cpu_seconds=422,
                registered_cases=[dict(source_plan='source_plan',record='r',kind='raw',group=34,jobs=jobs)]))
            save('completion',dict(state='phase_complete',all_reference_precision_passed=True))
            save('bound',dict(bank_plan='bank_plan'));save('bank_plan',dict(output='bank'))
            compressed_write(root/'bank/context.json.gz',dict(prior_weights=[.125]*8,query=63,
                origins={'4':dict(risk=[dict(record='r',head_log_scores=[0.]*8)])}))
            save('source_plan',dict(output='source'))
            save('source/compact_summary.json',dict(groups=[dict(record='r',kind='raw',candidate_passed=True)]))
            save('source/preparation.json',dict(selected=[dict(record='r',action_indices=[34,107,180])]))
            bundle=dict(prior_weights=[.125]*8,query=63,replicates=4,opposite=[{}]*16,head_log_scores=[0.]*8)
            compressed_write(root/'source/marginals/r.json.gz',bundle)
            for h in (4,5):save(f'source/parts/r/{h}_raw.json',dict(complete=True,cpu_seconds=100,
                cumulative_cpu_seconds=120,rows={'E9':dict(statistics=[[.2]],costs=[1])}))
            with patch.object(pilot,'ROOT',root),patch.object(pilot,'read',read),patch.object(pilot,'save',save),\
                contextlib.redirect_stdout(io.StringIO()):pilot.prepare('plan')
            prep=read('pilot/preparation.json')
            self.assertEqual(prep['estimated_cpu_seconds_including_reserves'],640)
            self.assertEqual(compressed_read(root/'pilot/marginals/r.json.gz'),bundle)
            for h in (4,5):
                source=read(f'source/parts/r/{h}_raw.json');cached=read(f'pilot/parts/r/{h}_raw.json')
                self.assertEqual(source['cumulative_cpu_seconds'],120)
                self.assertEqual(cached['cumulative_cpu_seconds'],0)
                self.assertEqual(cached['rows'],source['rows'])


if __name__=='__main__':unittest.main()
