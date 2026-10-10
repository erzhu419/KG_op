import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from performance.validate_historical_risk_competition import select_competitors,prepare,PLAN


class RiskCompetitionTests(unittest.TestCase):
    def test_repeated_preparation_preserves_completed_parts_and_the_original_cost_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'out';out.mkdir()
            saved=dict(complete=True,cumulative_estimated_cpu_seconds=1612.)
            (out/'preparation.json').write_text(json.dumps(saved))
            part=out/'completed_part.json';part.write_text('{"E9":1}')
            def read(path):
                return dict(output='out') if path==PLAN else json.loads(Path(path).read_text())
            with (patch('performance.validate_historical_risk_competition.ROOT',root),patch(
                'performance.validate_historical_risk_competition.read',side_effect=read),patch(
                'performance.validate_historical_risk_competition.prepare_contexts',side_effect=AssertionError('recomputed preparation')),
                contextlib.redirect_stdout(io.StringIO())):
                self.assertEqual(prepare(),saved)
            self.assertEqual(part.read_text(),'{"E9":1}')

    def test_selects_closest_in_each_kind_and_more_distant_overlap_without_duplicates(self):
        def group(gid,gap,overlap=0):
            return dict(group=gid,minimum_gap=gap,empirical_overlap_action_head_pairs=overlap)
        def report(rows):
            return dict(closest_groups=rows,empirical_overlap_groups=sum(r['empirical_overlap_action_head_pairs']>0 for r in rows))
        records=[dict(record='a',cdf=.1,raw_competitors=report([group(0,.01),group(1,.1,1)]),
            projected_competitors=report([group(2,.02,1)])),
            dict(record='b',cdf=.2,raw_competitors=report([group(3,.03)]),
            projected_competitors=report([group(4,.04)]))]
        selected=select_competitors(dict(records=records))
        self.assertEqual([(r['record'],r['kind'],r['group']) for r in selected],
            [('a','raw',0),('a','raw',1),('a','projected',2)])
        self.assertEqual(selected[-1]['selection_reasons'],['closest_uncovered_competitor','empirical_gap_rule_overlap'])

    def test_does_not_silently_omit_overlaps_outside_the_compact_report(self):
        summary=dict(closest_groups=[dict(group=0,minimum_gap=.1,empirical_overlap_action_head_pairs=0)],
            empirical_overlap_groups=1)
        with self.assertRaises(AssertionError):
            select_competitors(dict(records=[dict(record='a',cdf=.1,raw_competitors=summary)]))


if __name__=='__main__':unittest.main()
