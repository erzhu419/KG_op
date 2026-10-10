import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from performance.validate_historical_label_control import controlled_row, control_check
from performance import validate_historical_label_control as control


def row(value, cost, mass=1.):
    return dict(statistics=np.full((3,4),value).tolist(),costs=[cost]*3,
        diagnostics=dict(predictive_log_evidence_error=np.log(mass)))


class LabelControlTests(unittest.TestCase):
    def test_prepare_creates_output_and_keeps_the_previous_budget_separate(self):
        spec=dict(output='new',parent_plan='parent',transfer_plan='transfer',maximum_estimated_cpu_seconds=.1)
        contexts={n:dict(utility_identities=[4,5]) for n in ('low','source_n1_o5_h1_p223')}
        def read(path):
            if path=='plan':return spec
            if path in ('parent','transfer'):return dict(output=path)
            name=Path(path).name
            if name=='contexts.json':return contexts
            if name=='adaptive_summary.json':return dict(complete=True,cpu_seconds=2000.,failed_groups=[dict(record='low',group=12,kind='raw')])
            if name=='screen_summary.json':return dict(cpu_seconds=491.)
            if name=='cost_probe.json':return dict(resolution_units={'raw_'+str(h):dict(E9=.1) for h in (4,5)})
            if Path(path).is_file():return json.loads(Path(path).read_text())
            return dict(rows={n:dict(cpu_seconds=.1) for n in ('E7','E9','F9','L9','E10')})
        with tempfile.TemporaryDirectory() as tmp,patch.object(control,'ROOT',Path(tmp)),patch.object(control,'read',side_effect=read),\
            patch.object(control,'compressed_read',return_value=dict(fields=[dict(means=[0,1])]*8)),contextlib.redirect_stdout(io.StringIO()):
            result=control.prepare('plan')
            self.assertTrue((Path(tmp)/'new/resolution_plan.json').exists())
            self.assertFalse(result['estimated_cost_admitted'])
            self.assertEqual(result['previous_experiment_cpu_seconds'],2491.)
            self.assertTrue(result['previous_cost_hold_preserved'])
            # Completed rules still count against this experiment's cap on resume.
            cached=Path(tmp)/'new/parts/low/012_4_raw.json';cached.parent.mkdir(parents=True)
            cached.write_text(json.dumps(dict(rows={'E9':dict(cpu_seconds=200.)})))
            spec['maximum_estimated_cpu_seconds']=100.
            resumed=control.prepare('plan')
            self.assertFalse(resumed['estimated_cost_admitted'])
            self.assertGreaterEqual(resumed['estimated_cpu_seconds'],200.)

    def test_unnormalized_difference_uses_each_rules_own_mass(self):
        result=controlled_row(row(.1,100.,2.),row(.2,200.,3.),row(.05,100.,4.))
        np.testing.assert_allclose(result['statistics'],.2,atol=1e-15)
        np.testing.assert_allclose(result['costs'],400./3.,atol=1e-12)
        anchor=row(.1,120.,.98);target=row(.23,230.,1.03)
        result=controlled_row(anchor,target,anchor)
        np.testing.assert_allclose(result['statistics'],target['statistics'],atol=1e-15)
        np.testing.assert_allclose(result['costs'],target['costs'],atol=1e-12)

    def fixture(self):
        rng=np.random.default_rng(83)
        bundle=dict(utility_identities=[4,5],prior_weights=[.125]*8,actions=[{},{},{}],
            head_log_scores=rng.normal(size=8).tolist(),
            opposite=[dict(raw=rng.uniform(size=(8,3,4)).tolist(),
                head_log_scores=rng.normal(size=8).tolist()) for _ in range(4)])
        rules=['E7','F7','G7','E8','E9','F9','L9','E10']
        anchor={h:{n:row(.1,300.) for n in rules} for h in (4,5)}
        target={h:{n:row(.2,300.) for n in rules} for h in (4,5)}
        return bundle,anchor,target

    def test_one_identity_budget_direct_disagreement_is_rejected(self):
        bundle,anchor,target=self.fixture()
        self.assertTrue(control_check(bundle,anchor,target,1e-4)['passed'])
        target[5]['E10']['statistics'][2][0]+=.02
        result=control_check(bundle,anchor,target,1e-4)
        self.assertGreater(result['comparisons']['direct_E10'],1e-4)
        self.assertFalse(result['passed'])

    def test_negative_control_probability_is_not_clipped(self):
        bundle,anchor,target=self.fixture();original=copy.deepcopy(target)
        target[4]['E7']['statistics'][1][0]=-1.
        result=control_check(bundle,anchor,target,1e-4)
        self.assertFalse(result['valid_unclipped_ranges'])
        self.assertFalse(result['passed'])
        self.assertEqual(original[5],target[5])


if __name__=='__main__':unittest.main()
