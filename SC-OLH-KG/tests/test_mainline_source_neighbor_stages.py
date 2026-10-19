"""The source-label stage must wait for the complete public-pair stage."""
import pytest

from performance import analyze_mainline_source_neighbor_coverage as runner


def test_scoring_stops_before_labels_when_geometry_stage_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    runner.save(tmp_path / 'out/geometry_completion.json', {'status': 'failed'})
    with pytest.raises(ValueError, match='sealed before source scoring'):
        runner.execute({'output_dir': 'out'}, 'score')
