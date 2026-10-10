import copy
import unittest

import numpy as np

from performance.assemble_historical_risk_references import (
    add_entry, apply_entry, reference_masks, remaining_risk_checks, bank_group_tables)


class RiskReferenceAssemblyTests(unittest.TestCase):
    def fixture(self):
        group=dict(group=3, kind='guarded', action_indices=[1,4])
        context=dict(name='source_n1_o4_h1_p207', label=.2, head_log_scores=list(range(8)),
            utility_identities=[4,5], replicates=1, prior_weights=[.125]*8,
            groups={'3':dict(action_indices=[1,4], opposite=[dict(position=0,
                head_log_scores=[0.]*8, raw=np.zeros((8,2,4)).tolist())])})
        values={str(p):dict(statistics=np.zeros((8,5,4)).tolist(),costs=np.zeros((8,5)).tolist()) for p in (12,24)}
        row=copy.deepcopy({k:context[k] for k in ('label','head_log_scores','utility_identities','replicates')})
        row.update(position=207,head=1,eta_label=None,raw=values)
        objectives=[dict(position=0,head_log_scores=[0.]*8,raw=copy.deepcopy(values))]
        statistics=np.zeros((8,2,4));statistics[4]=.2;statistics[5]=.3
        costs=np.zeros((8,2));costs[[4,5]]=[128.,359.]
        entries={}
        add_entry(entries,context,group,'raw',statistics,costs,'direct_target','direct.json','E8')
        return row,objectives,entries,dict(actions=[{}]*5,groups=[group])

    def test_noncontiguous_overlay_preserves_low_values_and_all_identity_scores(self):
        row,objectives,entries,layout=self.fixture();before=copy.deepcopy(row)
        apply_entry(row,objectives,[.125]*8,next(iter(entries.values())),4)
        self.assertEqual(row['raw']['12'],before['raw']['12'])
        self.assertEqual(row['head_log_scores'],list(range(8)))
        high=np.asarray(row['raw']['24']['statistics'])
        self.assertTrue(np.all(high[4,[1,4]]==.2) and np.all(high[5,[1,4]]==.3))
        self.assertTrue(np.all(high[[0,1,2,3,6,7]]==0))
        self.assertTrue(np.all(high[:,[0,2,3]]==0))
        self.assertEqual(row['raw']['24']['costs'][4],[0.,128.,0.,0.,359.])

    def test_references_require_matching_origin_prior_likelihood_and_opposite_head(self):
        for mismatch in ('origin','prior','likelihood','opposite'):
            row,objectives,entries,_=self.fixture();prior=[.125]*8;origin=4
            if mismatch=='origin':origin=5
            elif mismatch=='prior':prior[0]=.1
            elif mismatch=='likelihood':row['head_log_scores'][0]=99.
            else:objectives[0]['raw']['24']['statistics'][4][1][0]=.001
            with self.subTest(mismatch=mismatch), self.assertRaises(AssertionError):
                apply_entry(row,objectives,prior,next(iter(entries.values())),origin)

    def test_direct_reference_keeps_precedence_and_both_sources(self):
        _,_,entries,_=self.fixture();entry=next(iter(entries.values()))
        add_entry(entries,entry['context'],entry['group'],'raw',np.ones((8,2,4)),
            np.zeros((8,2)),'paired_residual','residual.json','control32')
        chosen=next(iter(entries.values()))
        self.assertEqual(chosen['scope'],'direct_target')
        self.assertEqual(chosen['sources'],['direct.json','residual.json'])
        self.assertEqual(chosen['statistics'][4,0,0],.2)

    def test_direct_bank_cost_uses_existing_exact_budget_without_changing_guarded_cost(self):
        row,_,_,_=self.fixture()
        actions=[dict(kind='guarded'),dict(kind='direct',budgets=[359])]
        costs=np.zeros((8,2));costs[4]=[358.9,359.0000000000001]
        bank=dict(statistics={'raw':np.zeros((8,2,4))},costs={'raw':costs})
        _,cost=bank_group_tables(dict(utility_identities=[4,5],prior_weights=[.125]*8),
            dict(action_indices=[0,1]),bank,'raw',actions)
        self.assertEqual(cost[4,0],358.9)
        self.assertEqual(cost[4,1],359.)
        self.assertEqual(cost[5,1],359.)

    def test_residual_scope_does_not_become_direct_target_and_uncovered_failures_remain(self):
        _,_,entries,layout=self.fixture();entry=next(iter(entries.values()))
        entry['scope']='paired_residual'
        masks=reference_masks(layout,[entry])
        result=remaining_risk_checks(np.full((2,5),.01),np.array([[False]*5,[True]*5]),masks)
        self.assertEqual(result['reference_actions'],dict(direct_target=0,paired_residual=2,native_analytic=0))
        self.assertEqual(result['unreferenced_actions'],3)
        self.assertEqual(result['unreferenced_imprecise_action_head_pairs'],6)
        self.assertEqual(result['unreferenced_relevant_action_head_pairs'],3)
        self.assertEqual(result['unreferenced_relevant_action_indices'],[0,2,3])


if __name__=='__main__':unittest.main()
