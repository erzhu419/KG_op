import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from performance.diagnose_historical_decision_regions import region_metrics, observed_query_matrix, overlay_direct, objective_upper, risk_competitors
from performance.validate_historical_direct_body import prepare as prepare_direct, run as run_direct, PARENT, OUTPUT
from performance.validate_historical_component_transfer import body_cdf
from scipy.special import roots_legendre


class DecisionRegionsTests(unittest.TestCase):
    def test_competitors_keep_close_agreeing_groups_and_exclude_only_existing_coverage_or_bound(self):
        high=np.array([[.7,.699,.68,.6999,.9],[.8,.79,.799,.7999,.9]])
        low=high.copy();low[0,2]-=.03
        masks=dict(direct_target=np.array([False,False,False,True,True]))
        dominated=np.zeros_like(high,dtype=bool);dominated[:,0]=True
        groups=[dict(group=i,action_indices=[i]) for i in range(5)]
        rows=risk_competitors(high,np.zeros_like(high),low,dominated,masks,groups,[10,11])
        self.assertEqual([r['group'] for r in rows],[2,1])
        self.assertAlmostEqual(rows[0]['minimum_gap'],.101)
        self.assertEqual(rows[0]['objective_position'],11)
        self.assertEqual(rows[0]['winner_at_minimum_gap'],4)
        self.assertEqual(rows[0]['empirical_overlap_action_head_pairs'],0)
        self.assertEqual(rows[1]['maximum_risk_rule_difference'],0.)
        # A close competitor is reported even if its two old rules agree.
        high[:,0]=.6;high[:,2:4]=.6;high[:,4]=high[:,1]+.00005;low=high.copy()
        rows=risk_competitors(high,np.zeros_like(high),low,dominated,masks,groups,[10,11])
        self.assertEqual(rows[0]['group'],1)
        self.assertEqual(rows[0]['empirical_overlap_action_head_pairs'],2)

    def test_objective_upper_keeps_missing_identity_mass_and_shared_weights(self):
        f=np.zeros((2,8,3,4));f[:,4]=.2;f[:,5]=.8
        posterior=np.zeros((2,3,8));posterior[:,:,4]=.25;posterior[:,:,5]=.5;posterior[:,:,0]=.25
        upper=objective_upper(f,posterior,[4,5])
        np.testing.assert_allclose(upper,.7)
        posterior[:,:,4]=.5;posterior[:,:,5]=.25
        np.testing.assert_allclose(objective_upper(f,posterior,[4,5]),.55)

    def test_objective_upper_bounds_actual_verifier_branches(self):
        from core.raw_likelihood_lookahead import risk_branch_statistics
        rng=np.random.default_rng(43);f=rng.uniform(size=(2,8,3,4))
        posterior=rng.dirichlet(np.ones(8),size=(2,3));g=np.zeros((3,8,3,4))
        protocol=dict(comparison_budget=8)
        for risk in range(3):
            for h in range(8):
                for a in range(3):
                    branches,_=risk_branch_statistics(rng.uniform(size=(5,1,3)),np.ones((5,1)),1,[20,20,20],protocol)
                    g[risk,h,a]=branches.mean(axis=0)
        actual=np.einsum('fgk,fkab,gkab->fga',posterior,f,g)
        self.assertTrue(np.all(actual<=objective_upper(f,posterior,list(range(8)))+1e-14))

    def test_completed_cached_worker_retains_its_measured_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/OUTPUT;(out/'direct_tasks').mkdir(parents=True)
            names=['already_complete'];cached=dict(complete=True,records=names,new_integral_cpu_seconds=12.3,cpu_seconds=18.7)
            (out/'direct_plan.json').write_text(json.dumps(dict(estimated_cost_admitted=True,jobs=[dict(identity=4,origin=4,records=names)])))
            (out/'direct_tasks/4_4.json').write_text(json.dumps(cached))
            with patch('performance.validate_historical_direct_body.ROOT',root),patch(
                'performance.validate_historical_direct_body.compressed_read',side_effect=AssertionError('cached worker restored fields')):
                self.assertEqual(run_direct(4,4),cached)

    def test_cdf_coordinate_uses_the_record_16_or_32_point_rule(self):
        rules={'16':dict(positions=list(range(176,192)),labels=list(range(16))),
            '32':dict(positions=list(range(192,224)),labels=list(range(32)))}
        self.assertAlmostEqual(body_cdf(rules,176,0),.9*(roots_legendre(16)[0][0]+1)/2)
        self.assertAlmostEqual(body_cdf(rules,207,15),.9*(roots_legendre(32)[0][15]+1)/2)
        with self.assertRaises(AssertionError):body_cdf(rules,176,1)

    def test_overlay_keeps_other_actions_identities_and_likelihood_scores(self):
        row=dict(head_log_scores=list(range(8)),raw={str(p):dict(statistics=np.zeros((8,5,4)).tolist(),
            costs=np.zeros((8,5)).tolist()) for p in (12,24)},projected={str(p):dict(statistics=np.zeros((8,5,4)).tolist(),
            costs=np.zeros((8,5)).tolist()) for p in (10,12)})
        parts={4:dict(rows={r:dict(statistics=[[v]*4,[v]*4],costs=[128.,359.]) for r,v in [('G7',.2),('J8',.3)]})}
        overlay_direct(row,parts,[1,4])
        self.assertEqual(row['head_log_scores'],list(range(8)))
        for power,value in [('24',.2),('12',.3)]:
            expected=np.zeros((8,5,4));expected[4,[1,4]]=value
            np.testing.assert_array_equal(row['raw'][power]['statistics'],expected)
            self.assertEqual(row['raw'][power]['costs'][4],[0.,128.,0.,0.,359.])
        overlay_direct(row,parts,[1,4],kind='projected',pair=('G7','J8'))
        self.assertEqual(row['projected']['10']['statistics'][4][1],[.2]*4)
        self.assertEqual(row['projected']['12']['statistics'][4][1],[.3]*4)
        self.assertEqual(row['raw']['24']['statistics'][4][1],[.2]*4)

    def test_query_direct_pricing_uses_gh_cost_and_reuses_the_source_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base';(base/'parts').mkdir(parents=True)
            for h in (4,5):(base/'parts'/f'136_{h}_raw.json').write_text(json.dumps(dict(rows=dict(G7=dict(cpu_seconds=.1)))))
            fields=[None]*8;fields[4]=dict(means=[0]*512);fields[5]=dict(means=[0]*1024)
            contexts={f'source_n1_o{o}_h1_p{p}':dict(origin=o) for o in (4,5) for p in range(176,224)}
            contexts.pop('source_n1_o5_h1_p207')
            shared=dict(fields=fields,name='source_n1_o5_h1_p207',label=-.1,replicates=1,
                head_log_scores=[0.]*8,utility_identities=[4,5],prior_weights=[.125]*8,opposite=[])
            def read(path):
                if path==PARENT:return dict(output='base')
                return json.loads(Path(path).read_text())
            with patch('performance.validate_historical_direct_body.ROOT',root),patch(
                'performance.validate_historical_direct_body.read',side_effect=read),patch(
                'performance.validate_historical_direct_body.prepare_contexts',return_value=(contexts,shared,
                    dict(name='source_n1_o5_h1_p207',utility_identities=[4,5]))),contextlib.redirect_stdout(io.StringIO()):
                plan=prepare_direct()
            self.assertEqual(len(plan['jobs']),4);self.assertEqual(plan['reused_rule_rows'],2)
            self.assertAlmostEqual(plan['estimated_cpu_seconds'],299.28)
            self.assertTrue(plan['estimated_cost_admitted'])
            self.assertEqual(plan['source_components'],{'4':512,'5':1024})
            self.assertTrue((root/OUTPUT/'direct_parts/source_n1_o4_h1_p192/4.json').exists())
            (root/OUTPUT/'direct_summary.json').write_text(json.dumps(dict(records=[{}]*64)))
            (root/OUTPUT/'direct_plan.json').write_text(json.dumps(dict(estimated_cpu_seconds=3500.)))
            with patch('performance.validate_historical_direct_body.ROOT',root),patch(
                'performance.validate_historical_direct_body.read',side_effect=read),patch(
                'performance.validate_historical_direct_body.prepare_contexts',return_value=(contexts,shared,
                    dict(name='source_n1_o5_h1_p207',utility_identities=[4,5]))),contextlib.redirect_stdout(io.StringIO()):
                held=prepare_direct()
            self.assertFalse(held['estimated_cost_admitted'])
            self.assertAlmostEqual(held['estimated_cpu_seconds'],3799.28)

    def test_original_mass_negative_achieved_gain_and_cost_ties(self):
        raw=np.array([[.6,.8,.2],[.5,.5,.1]])
        native=np.array([[.2,.1,.9],[.5,.5,.1]])
        costs=np.array([[359.,128.,128.],[359.,128.,128.]])
        result=region_metrics(raw,costs,native,costs,np.array([.03,.07]),0)
        self.assertAlmostEqual(result['probability_mass'],.1)
        self.assertAlmostEqual(result['raw_adaptation'],.006)
        self.assertAlmostEqual(result['achieved_adaptation'],-.012)
        self.assertAlmostEqual(result['choice_loss'],.018)
        self.assertEqual(result['raw_winners'],[1])
        self.assertEqual(result['native_winners'],[1,2])

    def test_observed_switches_compare_same_state_and_ignore_budget_only_changes(self):
        rows=[dict(name='a',arm='source',task_id='t0',policy='stop',action=dict(primary=63)),
            dict(name='b',arm='source',task_id='t1',policy='stop',action=dict(primary=71)),
            dict(name='a',arm='source',task_id='t0',policy='batch1',candidate=63,action=dict(primary=63)),
            dict(name='b',arm='source',task_id='t1',policy='batch1',candidate=63,action=dict(primary=4))]
        result=observed_query_matrix(rows)
        self.assertEqual(result[0]['records'],2)
        self.assertEqual(result[0]['primary_changed'],1)
        self.assertEqual(result[0]['changed_tasks'],['t1'])


if __name__=='__main__':unittest.main()
