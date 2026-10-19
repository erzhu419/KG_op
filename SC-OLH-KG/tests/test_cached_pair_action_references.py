import unittest

import numpy as np

from core.shared_certificate_pair import guard_statistics_from_pair_states
from performance.reconstruct_historical_pair_action_references import reward_precision_checks


class CachedPairActionReferenceTest(unittest.TestCase):
    def test_reward_admission_preserves_state_hold_and_checks_each_head(self):
        states=np.tile([.2,.2,.2,.4],(2,1));shifted=states.copy()
        shifted[:,1]+=2.5e-4;shifted[:,3]-=2.5e-4
        source_precision_passed=np.max(abs(states-shifted))<=1e-4
        self.assertFalse(source_precision_passed)
        profiles={'E9':states,'E10':shifted,'F10':shifted,'L10':shifted}
        budgets=[[129,80,131],[129,129,82]];protocol=dict(comparison_budget=8)
        quality=np.tile([.7,.8,.3,.8],(16,2,1));omission=np.zeros((16,2))
        statistics={name:guard_statistics_from_pair_states(p,budgets,protocol)[0] for name,p in profiles.items()}
        mixed={name:np.sum(quality*s,axis=-1) for name,s in statistics.items()}
        self.assertTrue(reward_precision_checks(mixed,omission,0.)['passed'])
        self.assertFalse(source_precision_passed)
        self.assertFalse(reward_precision_checks(mixed,np.full((16,2),1.01e-4),0.)['passed'])
        self.assertFalse(reward_precision_checks(mixed,omission,1.01e-4)['passed'])
        quality[7,1,1]=.9;quality[7,1,3]=.4
        mixed={name:np.sum(quality*s,axis=-1) for name,s in statistics.items()}
        check=reward_precision_checks(mixed,omission,0.)
        self.assertFalse(check['passed'])
        self.assertAlmostEqual(check['maximum_mixed_reward_rule_difference'],1.25e-4,places=12)


if __name__=='__main__':unittest.main()
