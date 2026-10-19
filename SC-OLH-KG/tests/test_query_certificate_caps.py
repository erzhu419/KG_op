import copy
import unittest

import numpy as np

from core.conditional_raw_reference import component_risk_statistics
from core.direct_terminal_reference import expected_certificate
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.vectorized_joint_function_risk import update_raw_observation
from performance.refine_historical_query_certificate_caps import query_caps


class QueryCertificateCapsTest(unittest.TestCase):
    def test_guarded_delta_raw_likelihood_and_projected_mixture_match_historical_references(self):
        before=JointFunctionRiskPosterior.from_moments(np.array([[-.2,-3.]]),
            np.array([[.06,.01],[.01,.15]]).reshape(1,2,1,2))
        before.means=np.array([[-.2,-3.],[-.16,-2.9]])
        before.covariances=np.repeat(before.covariances,2,axis=0);before.log_weights=np.log([.4,.6])
        future=copy.deepcopy(before);row=dict(label=-.13,eta_label=-3.2)
        score=update_raw_observation(future,0,row['label'],row['eta_label'],4)
        protocol=dict(chance_alpha=.1,familywise_delta=.05,objective_reference=2.)
        original=protocol.copy();budgets=[('raw',81,3),('projected',81,3),('raw',356,1)]
        rows,diagnostic=query_caps(before,future,row,score,budgets,protocol,1e-4)
        self.assertEqual(protocol,original);self.assertTrue(diagnostic['precision_passed'])
        for r in rows:
            matched=dict(protocol,familywise_delta=.05/r['shortlist_size'])
            if r['kind']=='raw':
                action=dict(kind='direct',primary=0,shortlist=[0],incumbent_position=0,budgets=[r['budget']])
                stats,_,_=component_risk_statistics(before,future,action,row['label'],score,matched,
                    query=0,raw=True,order=128,power=0,seed=20261014,replicates=4,eta_label=row['eta_label'])
                actual=stats[0]
            else:actual=expected_certificate(future,0,r['budget'],matched)[0]
            self.assertAlmostEqual(r['point'],actual,places=10)
            self.assertGreaterEqual(r['upper'],actual)
            self.assertLessEqual(r['upper']-actual,1.00001e-4)
        self.assertEqual(len(before.log_weights),2);self.assertEqual(before.paid_episodes,0)


if __name__=='__main__':unittest.main()
