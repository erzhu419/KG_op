"""The one scheduled audit must follow successful substitutes and never retry."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/run_mainline_source_numerical_audit_cluster.py'
spec = importlib.util.spec_from_file_location('source_numerical_audit_cluster', SCRIPT)
cluster = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cluster)


@pytest.mark.parametrize('codes,expected_labels', [
    ([1], ['substitute_checks']),
    ([0, 0], ['substitute_checks', 'saved_record_audit']),
    ([0, 2], ['substitute_checks', 'saved_record_audit']),
])
def test_gate_preserves_actual_failure_without_repetition(tmp_path, monkeypatch, codes, expected_labels):
    monkeypatch.setattr(cluster, 'ROOT', tmp_path)
    monkeypatch.setattr(cluster.socket, 'gethostname', lambda: 'node004')
    (tmp_path / 'manifest.json').write_text(json.dumps({'output_dir': 'audit'}))
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=codes[len(calls) - 1], stdout='3 passed\n', stderr='')
    monkeypatch.setattr(cluster.subprocess, 'run', run)
    result = cluster.run_worker('manifest.json')
    sequence = json.loads((tmp_path / 'audit/cluster_control/validation/stage97_attempt1_sequence.json').read_text())
    assert result == codes[-1]
    assert [phase['phase'] for phase in sequence['phases']] == expected_labels
    assert len(calls) == len(codes)
    assert sequence['status'] == ('complete' if codes[-1] == 0 else 'failed')
    assert sequence['automatic_retries'] == 0
    if len(calls) == 2:
        assert calls[-1][calls[-1].index('-m') + 1] == cluster.AUDIT
    with pytest.raises(FileExistsError):
        cluster.run_worker('manifest.json')
    assert len(calls) == len(codes)
