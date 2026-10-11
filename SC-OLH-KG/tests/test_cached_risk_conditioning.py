import copy
import unittest

import numpy as np

from performance.refine_historical_observed_prediction import field_from_snapshot
from performance.screen_historical_observed_batch_risk import condition_risk
from representation.vectorized_joint_function_risk import local_raw_moments, update_raw_observation


class CachedRiskConditioningTest(unittest.TestCase):
    def test_cached_query_update_preserves_correlated_policy_posterior(self):
        loading=np.array([[.6,0.,0.,0.],[.2,.5,0.,0.],[.3,-.1,.4,0.],[.1,.2,.15,.3]])
        covariance=loading@loading.T
        payload=dict(means=[[.1,-.4,.5,-.2],[-.2,-.1,.3,-.5]],
            covariances=[covariance.tolist(),(covariance*1.3).tolist()],
            log_weights=np.log([.35,.65]).tolist(),paid_batches=2,paid_episodes=2)
        field=field_from_snapshot(payload);row=dict(record='new_positive_coordinate',label=.7,eta_label=-.3)
        query=dict(payload,means=np.asarray(payload['means'])[:,:2].tolist(),
            covariances=np.asarray(payload['covariances'])[:,:2,:2].tolist())
        local=local_raw_moments(field.means[:,:2],field.covariances[:,:2,:2],row['label'],row['eta_label'],4)
        saved={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in local.items()}
        context=dict(risk_query_fields={'4':query},risk_local_moments={row['record']:{'4':saved}})
        native=copy.deepcopy(field);expected=update_raw_observation(native,0,row['label'],row['eta_label'],4)
        score=condition_risk(field,row,4,context)
        self.assertEqual(score,expected)
        np.testing.assert_array_equal(field.means,native.means)
        np.testing.assert_array_equal(field.covariances,native.covariances)
        np.testing.assert_array_equal(field.log_weights,native.log_weights)
        self.assertEqual(field.paid_episodes,6)
        self.assertEqual(field.paid_batches,3)


if __name__=='__main__':unittest.main()
