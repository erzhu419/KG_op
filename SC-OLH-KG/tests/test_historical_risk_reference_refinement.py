import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from performance import check_historical_observed_batch_risk as risk
from performance.check_historical_observed_batch_risk import reference_comparisons,selected_rule
from performance.validate_historical_component_bank import RULES,group_checks,selected_reference_rule
from performance.diagnose_historical_component_guarded import compressed_write


class RiskReferenceRefinementTest(unittest.TestCase):
    def test_lower_gauss_pilot_retains_original_comparisons(self):
        case=dict(record='r',kind='raw',candidate='E9',rules=['E8','E9','F9','K9','L9'])
        self.assertEqual(reference_comparisons(dict(refined_cases=[case]),'r','raw'),
            [('E8','E9'),('E9','F9'),('E9','K9'),('L9','E9')])
        case['rules'].remove('K9')
        self.assertEqual(reference_comparisons(dict(refined_cases=[case]),'r','raw'),
            [('E8','E9'),('E9','F9'),('L9','E9')])
        case.update(candidate='E8',rules=['E8','F8','L8','E9'])
        self.assertEqual(reference_comparisons(dict(refined_cases=[case]),'r','raw'),
            [('E8','F8'),('L8','E8'),('E8','E9')])

    def test_refinement_keeps_candidate_and_other_record_gates(self):
        spec=dict(refined_cases=[dict(record='failed',kind='raw',candidate='E9',independent_reference='F10'),
            dict(record='passed',kind='raw',candidate='E8')])
        self.assertEqual(selected_rule(spec,'failed','raw'),'E9')
        self.assertEqual(reference_comparisons(spec,'failed','raw'),
            [('E8','E9'),('E9','F10'),('E9','K9'),('F9','F10')])
        self.assertEqual(reference_comparisons(spec,'passed','raw'),[('E8','F8'),('E8','H8'),('E8','E9')])
        self.assertEqual(RULES['F10'],(32,10,101))
        self.assertEqual(RULES['F9'][0::2],RULES['F10'][0::2])

    def test_finer_independent_rule_can_resolve_original_failed_difference(self):
        f=np.zeros((8,3,4));f[:,:,0]=1.
        g=np.zeros((3,4));g[:,0]=.4
        bundle=dict(prior_weights=[.125]*8,head_log_scores=[0.]*8,actions=[{}]*3,
            opposite=[dict(head_log_scores=[0.]*8,raw=f.tolist())])
        parts=[]
        for identity in (4,5):
            rows={}
            for name,shift in [('E8',0.),('E9',0.),('K9',0.),('F9',.0006),('F10',.0003)]:
                stats=g.copy();stats[1,0]+=shift
                rows[name]=dict(statistics=stats.tolist(),costs=[0.]*3,
                    diagnostics=dict(predictive_log_evidence_error=0.))
            parts.append(dict(identity=identity,rows=rows))
        original=[('E8','E9'),('E9','F9'),('E9','K9')]
        refined=reference_comparisons(dict(refined_cases=[dict(record='r',kind='raw',candidate='E9',
            independent_reference='F10')]),'r','raw')
        self.assertFalse(group_checks(bundle,parts,'raw',original,1e-4)['passed'])
        self.assertTrue(group_checks(bundle,parts,'raw',refined,1e-4)['passed'])
        unresolved=copy.deepcopy(parts)
        for p in unresolved:p['rows']['F10']=copy.deepcopy(p['rows']['E9'])
        self.assertFalse(group_checks(bundle,unresolved,'raw',refined,1e-4)['passed'])

    def test_collector_uses_recorded_identity_resolution_and_finer_validation(self):
        f=np.zeros((8,3,4));f[:,:,0]=1.
        g=np.zeros((3,4));g[:,0]=.4
        bundle=dict(prior_weights=[.125]*8,head_log_scores=[0.]*8,actions=[{}]*3,
            opposite=[dict(head_log_scores=[0.]*8,raw=f.tolist())])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';out.mkdir()
            case=dict(record='r',kind='raw',candidate='E9',independent_reference='F10',
                identity_candidates={'4':'E10','5':'E9'})
            (root/'plan.json').write_text(json.dumps(dict(output='out',refined_cases=[case])))
            (out/'preparation.json').write_text(json.dumps(dict(selected=[dict(record='r',group=34)],
                jobs=[dict(record='r',kind='raw')],estimated_cpu_seconds_including_reserves=3576)))
            compressed_write(out/'marginals/r.json.gz',bundle)
            for h in (4,5):
                rows={}
                for name,shift in [('E8',0.),('E9',0.),('K9',0.),('F9',.0006),('F10',.0003)]+([('E10',.0003)] if h==4 else []):
                    stats=g.copy();stats[1,0]+=shift
                    rows[name]=dict(statistics=stats.tolist(),costs=[0.]*3,
                        diagnostics=dict(predictive_log_evidence_error=0.))
                part=dict(complete=True,identity=h,rows=rows,cpu_seconds=1.,maximum_branch_above_marginal_bound=0.,
                    marginal_bounds=[dict(raw64_128_difference=0.,maximum_log_evidence_error=0.)])
                p=out/'parts/r'/f'{h}_raw.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(part))
            def read(path):return json.loads((root/path).read_text())
            def save(path,data):Path(path).write_text(json.dumps(data))
            with patch.object(risk,'ROOT',root),patch.object(risk,'read',read),patch.object(risk,'save',save),\
                contextlib.redirect_stdout(io.StringIO()):
                result=risk.collect('plan.json')
            self.assertTrue(result['local_risk_candidate_passed'])
            check=result['groups'][0]
            self.assertEqual(selected_reference_rule(check,4),'E10')
            self.assertEqual(selected_reference_rule(check,5),'E9')
            self.assertEqual(check['validation_rule'],'F10')
            self.assertEqual(check['checks']['comparisons'][1]['high'],'selected')


if __name__=='__main__':unittest.main()
