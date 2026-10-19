"""Frozen policy identities with new shared raws, without real task calls."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from performance import run_mainline_role_fresh as runner
from performance.mainline_legacy_native import materialize_v69_task


class FrozenBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol=runner.read(runner.ROOT/'performance/manifests/mainline_observation_seed_replication_v1.json')

    def test_saved_policies_match_the_frozen_native_design(self):
        # An incompatible saved design would fail the existing source-informed
        # initialization contract before any of the new observations can run.
        designs={}
        for task in self.protocol['tasks']:
            domain=task['domain']
            if domain not in designs:
                designs[domain]=materialize_v69_task(domain)['args']['initial_design_points']
            rows=runner.read(runner.ROOT/task['initial_policy_file'])['history']
            self.assertEqual(len(rows),10)
            self.assertEqual([row['point'] for row in rows],designs[domain])

    def test_original_bank_and_new_raws_are_shared_by_all_histories(self):
        task=self.protocol['tasks'][0]
        seed=task['geometry_task_seed']
        initial_source=runner.ROOT/task['initial_policy_file']
        neighbor_source=runner.ROOT/task['shared_neighbors_file']
        frozen_rows=runner.read(initial_source)['history']
        fixed_points=[row['point'] for row in frozen_rows]
        geometry=task['geometry']['problem_kwargs']
        plan={'domain':task['domain'],'target_seeds':[seed],
              'geometry_by_seed':{str(seed):task['geometry']},
              'fixed_initialization_file_by_task':{str(seed):str(initial_source)},
              'fixed_neighbors_file_by_task':{str(seed):str(neighbor_source)}}
        built=[]

        class FakeEngine:
            def __init__(self, arm, initial_rows):
                self.arm=arm
                self.initial_rows=initial_rows
                self.history=[]
                self.iteration_log=[]
                self.config=SimpleNamespace(initial_design_points=(),
                    truth_pool_diagnostics=False,observation_seed=2026102211)
                target=SimpleNamespace(**geometry)
                self.problem=SimpleNamespace(d=1000,base=SimpleNamespace(base=target),
                    simulate=lambda point,rng=None:np.array([1000+len(self.history),-2000-len(self.history)]))
                self.frozen_terminal={'primary':fixed_points[0],'terminal':{}}

            def run_search(self):
                for i,point in enumerate(self.config.initial_design_points):
                    raw=(np.asarray(self.initial_rows[i]['raw']) if self.initial_rows
                         else self.problem.simulate(point))
                    self.history.append((point,raw))
                if self.arm!=runner.STOP:
                    for selected_id in range(3):
                        point=self.audit['points'][selected_id]
                        self.audit['stages'].append({'paid_prefix':len(self.history),
                            'selected_id':selected_id,'action_ids':self.audit['shared_ids']})
                        raw=self.problem.simulate(point)
                        self.history.append((point,raw))
                        self.iteration_log.append({'x_selected':list(point)})

        def build(plan,seed,arm,initial_rows):
            engine=FakeEngine(arm,initial_rows)
            built.append(engine)
            return engine

        @contextmanager
        def neighbors(engine,points):
            audit={'points':[tuple(map(int,p)) for p in points],
                   'shared_ids':list(range(len(points))),'stages':[]}
            engine.audit=audit
            yield audit

        # Fail on source restoration or regeneration. The only simulate method
        # exercised below is the deterministic local substitute above.
        with TemporaryDirectory() as tmp, \
                patch.object(runner,'build_fresh_backend',side_effect=build), \
                patch.object(runner,'shared_neighbor_candidates',neighbors), \
                patch.object(runner,'save_engine'), \
                patch.object(runner,'load_pickle',side_effect=AssertionError('pickle restore')), \
                patch.object(runner,'make_shared_neighbors',side_effect=AssertionError('bank regeneration')):
            output=Path(tmp)
            runner.search(output,plan,seed,runner.STOP)
            runner.prepare(output,plan)
            copied=output/f'seed_{seed}'
            self.assertEqual((copied/'neighbors.npz').read_bytes(),neighbor_source.read_bytes())
            self.assertEqual((copied/'neighbors.json').read_bytes(),neighbor_source.with_suffix('.json').read_bytes())
            with np.load(copied/'neighbors.npz') as data:
                self.assertEqual(data['points'].shape,(64,1000))
            fresh=runner.read(copied/'initialization.json')
            self.assertEqual([r['point'] for r in fresh['history']],fixed_points)
            self.assertNotEqual([r['raw'] for r in fresh['history']],[r['raw'] for r in frozen_rows])
            self.assertEqual(fresh['observation_seed'],2026102211)
            self.assertEqual(built[0].initial_rows,[])
            for arm in self.protocol['search_arms']:
                if arm==runner.STOP:
                    continue
                runner.search(output,plan,seed,arm)
                result=runner.read(runner.case_dir(output,seed,arm)/'search.json')
                self.assertEqual(result['history'][:10],fresh['history'])
                self.assertEqual((result['new_initial_calls'],result['new_search_calls']),(0,3))
                self.assertEqual(result['observation_seed'],2026102211)
                self.assertEqual(built[-1].initial_rows,fresh['history'])
            stop=runner.read(runner.case_dir(output,seed,runner.STOP)/'search.json')
            self.assertEqual((stop['new_initial_calls'],stop['new_search_calls']),(10,0))
            self.assertEqual(len(built),7)


if __name__=='__main__':
    unittest.main()
