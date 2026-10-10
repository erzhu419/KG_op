import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.budgeted_guarded_kg import BudgetedTerminalActions
from performance.validate_historical_component_bank import action_groups
from performance.validate_historical_component_transfer import part_path
from performance.validate_historical_cdf_bank import select_pilot, check, collect, prepare_pilot, prepare_decision, RULES


def row(value,actions):
    return dict(statistics=np.full((actions,4),value).tolist(),costs=[200.]*actions,
        diagnostics=dict(predictive_log_evidence_error=0.),cpu_seconds=3.)


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))


class CdfBankTests(unittest.TestCase):
    def test_decision_extension_preserves_worker_indices_and_cumulative_cost_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'out';plan=root/'plan.json'
            write(plan,dict(output='out',maximum_estimated_cpu_seconds_per_stage=1000,maximum_job_cpu_seconds=600))
            original=dict(selected=[dict(record='interior',group=0,kind='raw')],estimated_cpu_seconds=600.,
                maximum_estimated_worker_cpu_seconds=200.,scope='initial')
            write(out/'pilot_plan.json',original);write(out/'summary.json',dict(complete=True))
            write(out/'contexts.json',dict(interior=dict(utility_identities=[4,5])))
            write(out/'bank_layout.json',dict(groups=[dict(group=0,action_indices=[0,2]),dict(group=1,action_indices=[1,3])]))
            write(out/'coarse_plan.json',dict(selected=[dict(record='interior',group=g,kind='raw',anchor_rule='G7') for g in (0,1)]))
            write(out/'decision_relevance.json',dict(records=dict(interior={k:dict(winner_actions={'3':{}},runner_up_actions={'1':{}})
                for k in ('raw','projected')})))
            with patch('performance.validate_historical_cdf_bank.ROOT',root),patch(
                'performance.validate_historical_cdf_bank.price_rows',return_value=(300.,250.)),contextlib.redirect_stdout(io.StringIO()):
                result=prepare_decision(str(plan))
            self.assertTrue(result['estimated_cost_admitted']);self.assertEqual(result['estimated_cpu_seconds'],900.)
            self.assertEqual(result['selected'][0],original['selected'][0])
            self.assertEqual([(r['group'],r['kind']) for r in result['selected']],[(0,'raw'),(1,'raw')])
            self.assertEqual(json.loads((out/'initial_pilot_plan.json').read_text()),original)
            write(out/'pilot_plan.json',original)
            with patch('performance.validate_historical_cdf_bank.ROOT',root),patch(
                'performance.validate_historical_cdf_bank.price_rows',return_value=(401.,250.)),contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(prepare_decision(str(plan))['estimated_cost_admitted'])
            self.assertEqual(json.loads((out/'pilot_plan.json').read_text()),original)

    def test_pilot_pricing_excludes_cache_and_enforces_stage_and_worker_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'out';base=root/'base';plan=root/'plan.json'
            spec=dict(output='out',parent_plan=str(root/'parent.json'),maximum_estimated_cpu_seconds_per_stage=155.,maximum_job_cpu_seconds=27.)
            write(plan,spec);write(root/'parent.json',dict(output='base'));write(out/'coarse_summary.json',dict(complete=True))
            write(out/'contexts.json',dict(interior=dict(utility_identities=[4,5])))
            write(out/'bank_layout.json',dict(groups=[dict(kind='guarded'),dict(kind='direct')]))
            rows=[dict(record='interior',group=g,kind=k,anchor_rule='G7',target_precision_admitted=False,
                valid_unclipped_ranges=True,minimum_selected_statistic=0.,baseline_maximum_action_difference=.1)
                for g,k in [(0,'raw'),(1,'raw'),(0,'projected')]]
            write(out/'coarse_groups.json',rows);write(out/'coarse_plan.json',dict(selected=rows))
            times={k:{str(h):{r:3. for r in RULES if r!='E9'} for h in (4,5)} for k in ('raw','projected')}
            write(root/'paper_artifacts/historical_component_transfer_v1/cost_probe.json',dict(groups={'0':dict(times=times)},
                resolution_units={f'{k}_{h}':dict(E9=3.) for k in ('raw','projected') for h in (4,5)}))
            for r in rows:
                for h in (4,5):
                    write(base/'parts'/f'{r["group"]:03d}_{h}_{r["kind"]}.json',dict(rows={n:row(.2,3) for n in ('E7','F7','G7')}))
                    write(part_path(out,r,h),dict(rows={}))
            with patch('performance.validate_historical_cdf_bank.ROOT',root),contextlib.redirect_stdout(io.StringIO()):
                held=prepare_pilot(str(plan));self.assertFalse(held['estimated_cost_admitted'])
                self.assertAlmostEqual(held['estimated_cpu_seconds'],158.4)
                write(part_path(out,rows[0],4),dict(rows=dict(E7=row(.2,3))))
                admitted=prepare_pilot(str(plan));self.assertTrue(admitted['estimated_cost_admitted'])
                self.assertAlmostEqual(admitted['estimated_cpu_seconds'],154.8)
                spec['maximum_job_cpu_seconds']=26.;write(plan,spec)
                self.assertFalse(prepare_pilot(str(plan))['estimated_cost_admitted'])

    def bundle(self):
        rng=np.random.default_rng(87)
        return dict(prior_weights=[.125]*8,utility_identities=[4,5],actions=[dict(kind='guarded')]*3,
            head_log_scores=rng.normal(size=8).tolist(),opposite=[dict(raw=rng.uniform(size=(8,3,4)).tolist(),
                head_log_scores=rng.normal(size=8).tolist()) for _ in range(4)])

    def test_selection_prioritizes_invalid_ranges_then_baseline_disagreement(self):
        layout=[dict(kind='guarded'),dict(kind='guarded'),dict(kind='direct')]
        rows=[dict(record='interior',group=g,kind=k,target_precision_admitted=False,
            valid_unclipped_ranges=not (g==1 and k=='raw'),minimum_selected_statistic=-.01 if g==1 else 0.,
            baseline_maximum_action_difference=10.-g) for g,k in [(0,'raw'),(1,'raw'),(2,'raw'),(0,'projected'),(1,'projected')]]
        selected=select_pilot(rows,layout)
        self.assertEqual([(r['kind'],r['group']) for r in selected],[('raw',1),('raw',2),('projected',0)])
        self.assertEqual(selected[0]['selection_reason'],'invalid_coarse_range')

    def test_direct_reference_selects_actual_fallback_and_holds_a_failed_reference(self):
        bundle=self.bundle();anchor={h:{r:row(.2,3) for r in ('E5','E7','F7','G7')} for h in (4,5)}
        target={h:{r:row(.3,3) for r in ('E5',*RULES)} for h in (4,5)}
        result,data=check(bundle,anchor,target,'G7','raw',1e-4)
        self.assertEqual(result['selected_method'],'control32')
        target[5]['E5']['statistics'][1][0]+=.1
        self.assertEqual(check(bundle,anchor,target,'G7','raw',1e-4)[0]['selected_method'],'control128')
        target[5]['E7']['statistics'][1][0]+=.1
        result,data=check(bundle,anchor,target,'G7','raw',1e-4)
        self.assertEqual(result['selected_method'],'direct256');self.assertAlmostEqual(data[0][5,1,0],.3)
        target[5]['E9']['statistics'][2][0]+=.1
        result,data=check(bundle,anchor,target,'G7','raw',1e-4)
        self.assertEqual(result['selected_method'],'HOLD');self.assertIsNone(data)

    def test_direct_fixed_cost_is_analytic_in_all_reference_rules(self):
        bundle=self.bundle();bundle['actions']=[dict(kind='direct',budgets=[b]) for b in (128,200,359)]
        anchor={h:{r:row(.2,3) for r in ('E5','E7','F7','G7')} for h in (4,5)}
        target={h:{r:row(.3,3) for r in ('E5',*RULES)} for h in (4,5)}
        for values in target.values():
            for value in values.values():value['costs']=[128.,200.,359.0000000000001]
        result,data=check(bundle,anchor,target,'G7','raw',1e-4)
        self.assertEqual(result['selected_method'],'control32')
        np.testing.assert_array_equal(data[1][4:6],[[128.,200.,359.],[128.,200.,359.]])

    def test_partial_bank_patch_keeps_other_actions_and_does_not_admit_full_bank(self):
        terminal=BudgetedTerminalActions([[0,2,3],[1,2,3]],[1,1],dict(comparison_budget=8))
        actions=[dict(kind=k,shortlist=s.tolist(),incumbent_position=int(i),budgets=b)
            for k,table,inc,b in terminal.allocations(359) for s,i in zip(table,inc)]
        layout=action_groups(actions)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'out';base=root/'base';contexts={};banks={};groups=[];selected=[];all_rows=[]
            for record in ('lower','upper'):
                context=dict(utility_identities=[4,5],prior_weights=[.125]*8,predictive_cdf=.2,
                    label=-.2,head_log_scores=[0.]*8,groups={})
                banks[record]={name:{k:np.full((8,len(actions),4) if name=='statistics' else (8,len(actions)),.1).tolist()
                    for k in ('raw','projected')} for name in ('statistics','costs')}
                for g in layout:
                    gid=g['group'];a=len(g['action_indices']);context['groups'][str(gid)]=dict(opposite=[])
                    for kind in ('raw','projected'):
                        groups.append(dict(record=record,group=gid,kind=kind,action_indices=g['action_indices'],
                            target_precision_admitted=kind=='projected' and g['kind']=='direct'))
                        if kind=='projected' and g['kind']=='direct':continue
                        r=dict(record=record,group=gid,kind=kind,anchor_rule='G7',rules=RULES,selection_reason='test')
                        all_rows.append(r)
                        pilot=(kind,gid) in (('raw',0),('raw',2),('projected',1))
                        if pilot:selected.append(r)
                        for h in (4,5):
                            write(base/'parts'/f'{gid:03d}_{h}_{kind}.json',dict(rows={n:row(.2,a) for n in ('E7','F7')}))
                            write(part_path(out,r,h),dict(rows={n:row(.3,a) for n in RULES} if pilot else {}))
                contexts[record]=context
            write(out/'contexts.json',contexts);write(out/'bank_layout.json',dict(actions=actions,groups=layout))
            write(out/'coarse_groups.json',groups);write(out/'coarse_risk_bank.json',banks)
            write(out/'coarse_plan.json',dict(selected=all_rows));write(out/'pilot_plan.json',dict(selected=selected,estimated_cpu_seconds=400.))
            write(root/'parent.json',dict(output='base'));plan=root/'plan.json'
            write(plan,dict(output='out',parent_plan=str(root/'parent.json'),maximum_action_difference=1e-4,
                maximum_estimated_cpu_seconds_per_stage=3600))
            for kind in ('raw','projected'):
                for shard in range(sum(r['kind']==kind for r in selected)):
                    for h in (4,5):write(out/'tasks/pilot'/f'{h}_{kind}_{shard}.json',dict(complete=True,cpu_seconds=1.,peak_rss_mb=100.))
            def checked(bundle,*args):
                a=len(bundle['actions']);return dict(selected_method='direct256'),(np.full((8,a,4),.3),np.full((8,a),200.))
            with patch('performance.validate_historical_cdf_bank.ROOT',root),patch(
                'performance.validate_historical_cdf_bank.check',side_effect=checked),contextlib.redirect_stdout(io.StringIO()):
                result=collect(str(plan))
            self.assertTrue(result['all_pilot_references_admitted']);self.assertFalse(result['full_predictive_integration_admitted'])
            self.assertEqual(result['target_precision_actions_admitted'],{r:dict(raw=5,projected=7) for r in contexts})
            bank=json.loads((out/'selected_risk_bank.json').read_text())
            for r in selected:
                indices=layout[r['group']]['action_indices']
                np.testing.assert_array_equal(np.asarray(bank[r['record']]['statistics'][r['kind']])[:,indices],.3)
            np.testing.assert_array_equal(np.asarray(bank['lower']['statistics']['raw'])[:,layout[1]['action_indices']],.1)


if __name__=='__main__':unittest.main()
