"""Three actual semantics: aliases are not identity; reference scores stay reused."""
from performance import mainline_three_stream_query_attribution as attribution


def test_equal_history_labels_with_different_physical_policies_stay_distinct():
    registry,catalog = {},{}
    first = attribution.register_alias(registry,catalog,[1,2,3],'dynamic',
        dict(stream='original',arm='AFTER',label='AFTER:history:11'))
    second = attribution.register_alias(registry,catalog,[1,2,4],'dynamic',
        dict(stream='observation_1',arm='AFTER',label='AFTER:history:11'))
    assert first=='P000' and second=='P001'
    assert len(catalog)==2


def test_equal_physical_policy_under_different_stream_and_arm_labels_joins():
    registry,catalog = {},{}
    first = attribution.register_alias(registry,catalog,[1,2,3],'dynamic',
        dict(stream='original',arm='FULL',label='FULL:history:10'))
    second = attribution.register_alias(registry,catalog,(1,2,3),'dynamic',
        dict(stream='observation_2',arm='AFTER',label='AFTER:history:12'))
    assert first==second=='P000'
    assert len(catalog)==1 and len(catalog[first]['aliases'])==2
    assert attribution.register_policy(registry,[1,3,3])=='P001'
    assert attribution.register_policy(registry,[1,2,3])==first


def test_original_completed_query_attribution_is_read_without_running_audit(monkeypatch):
    saved = dict(reconstruction=dict(recorded_choices=9,max_score_error=0.))
    reads = []
    def read(path):
        reads.append(str(path))
        return dict(query=saved)
    def forbidden(*args,**kwargs):
        raise AssertionError('The opened original audit must not be reconstructed')
    monkeypatch.setattr(attribution,'read',read)
    monkeypatch.setattr(attribution.single,'analyze_task',forbidden)
    result,reused = attribution.load_query('/repo',dict(name='original',
        reuse_attribution_dir='stage66'),2026104003)
    assert reused and result is saved
    assert reads==['/repo/stage66/cases/seed_2026104003/analysis.json']
