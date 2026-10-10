import json
import tempfile
import unittest
from pathlib import Path

from performance.validate_historical_upper_decisions import make_jobs


class UpperDecisionTests(unittest.TestCase):
    def test_pilot_and_extension_keep_both_origins_and_cumulative_cost(self):
        spec=dict(pilot_positions=[155,168],positions=[153,155,168],origins=[4,5],groups=[107,144],
            rules=['E7','F7','G7'],analysis_reserve_cpu_seconds=120.,maximum_estimated_cpu_seconds=3600.,maximum_job_cpu_seconds=600.)
        contexts={f'source_n1_o{o}_h1_p{p}':dict(utility_identities=[4,5]) for o in (4,5) for p in spec['positions']}
        source={(g,h):{r:dict(cpu_seconds=.1) for r in spec['rules']} for g in spec['groups'] for h in (4,5)}
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);pilot=make_jobs(spec,contexts,source,out,'pilot',analysis_cpu=3.)
            self.assertEqual(len(pilot['jobs']),8)
            self.assertTrue(all(len(j['records'])==2 for j in pilot['jobs']))
            self.assertAlmostEqual(pilot['estimated_integral_cpu_seconds'],24.96)
            self.assertAlmostEqual(pilot['cumulative_estimated_cpu_seconds'],147.96)
            path=out/'parts/source_n1_o4_h1_p153/107_4.json';path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(rows=dict(E7=dict(cpu_seconds=15.)))))
            full=make_jobs(spec,contexts,source,out,'full',pilot['estimated_integral_cpu_seconds'],5.)
            self.assertTrue(all(j['records']==[f'source_n1_o{j["origin"]}_h1_p153'] for j in full['jobs']))
            self.assertAlmostEqual(full['incremental_estimated_cpu_seconds'],12.36)
            self.assertAlmostEqual(full['cumulative_estimated_cpu_seconds'],162.32)
            self.assertTrue(full['estimated_cost_admitted'])
            held=make_jobs(spec,contexts,source,out,'full',3500.,5.)
            self.assertFalse(held['estimated_cost_admitted'])

    def test_expensive_worker_is_held_even_when_total_is_below_stage_cap(self):
        spec=dict(pilot_positions=[155],positions=[155],origins=[4],groups=[107],rules=['E7'],
            analysis_reserve_cpu_seconds=0.,maximum_estimated_cpu_seconds=3600.,maximum_job_cpu_seconds=600.)
        contexts={'source_n1_o4_h1_p155':dict(utility_identities=[4,5])}
        source={(107,h):dict(E7=dict(cpu_seconds=500.)) for h in (4,5)}
        with tempfile.TemporaryDirectory() as directory:
            plan=make_jobs(spec,contexts,source,Path(directory),'pilot')
        self.assertLess(plan['cumulative_estimated_cpu_seconds'],3600.)
        self.assertFalse(plan['estimated_cost_admitted'])


if __name__=='__main__':unittest.main()
