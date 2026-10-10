import copy
import json
import pickle
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import performance.historical_full_bank_batch_kg as batch_kg
import performance.refine_historical_achievable_policies as policies
import performance.extend_historical_predictive_heads as extension
from performance.extend_historical_predictive_heads import head_labels, merge_records
from core.conditional_terminal_kg import ConditionalTerminalActions
from core.raw_terminal_head_grid import evaluate_head_records
from performance.run_legacy_joint_population_matched_baselines import paired_fields
from representation.joint_response_mixture import JointResponseMixture


class PredictiveExtensionTests(unittest.TestCase):
    def test_preparation_reuses_snapshot_and_only_adds_new_risk_positions(self):
        self.check_preparation_reuse(1)

    def test_batch_preparation_reuses_observed_coordinates_but_recomputes_old_proxies(self):
        for legacy_proxy in (False, True):
            with self.subTest(legacy_proxy=legacy_proxy):
                self.check_preparation_reuse(4, legacy_proxy)

    def check_preparation_reuse(self, n, legacy_proxy=False):
        fields=[(np.full(3,v),np.eye(3)*.02+.003) for v in (1.,-.2,-4.,-4.)]
        model=JointResponseMixture(['first','second'],[paired_fields(fields),paired_fields(fields)])
        terminal=ConditionalTerminalActions([[1,0,2]],[1],dict(objective_reference=2.,chance_alpha=.05,
            familywise_delta=.05,comparison_budget=8,comparison_delta=.05))
        labels,eta,_=extension.predictive_observed_batches(model,0,n,2,714)
        actions=extension._actions(terminal,360-n)
        heads={(h,o):[dict(position=p,label=labels[o*4+p,h],
            eta_label=None if eta is None else eta[o*4+p,h]+(0.5 if legacy_proxy else 1e-10),
            raw={'18':dict(value=99+p)})
            for p in range(2)] for o in (0,1) for h in (0,1)}
        case=dict(name='source_n'+str(n),state='source',n=n,predictive_power=2,head_counts=[2,4])
        spec=dict(output='next',reuse_precision_plan='old_plan',posterior_inputs={'source':'cached.pkl'},cases=[case])
        payload={'plan':spec,'old_plan':dict(parent_plan='parent',output='old',cases=[case]),
            'parent':dict(output='original',query=0,predictive_seed=714),
            'old/registered_jobs.json':dict(jobs=[]),
            'original/source/bank.json':dict(origins=[0,1],utility_identities=[0,1])}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);snapshot=root/'cached.pkl'
            with snapshot.open('wb') as f:pickle.dump((model,terminal,{},{}),f)
            before=snapshot.read_bytes()
            with patch.object(extension,'ROOT',root),patch.object(batch_kg,'ROOT',root), \
                    patch.object(extension,'read',side_effect=payload.__getitem__), \
                    patch.object(extension,'assemble_case',return_value=(heads,{},[])), \
                    patch.object(extension,'original_heads',return_value=[dict(actions=actions)]), \
                    patch.object(batch_kg,'context',side_effect=AssertionError('Source replay is unnecessary')):
                extension.prepare('source','plan')
            self.assertEqual(snapshot.read_bytes(),before)
            self.assertFalse((root/'next/source/posterior.pkl').exists())
            result=json.loads((root/'next/source/preparation.json').read_text())
            self.assertTrue(result['posterior_snapshot_reused'])
            self.assertEqual(len(result['jobs']),12 if legacy_proxy else 4)
            if not legacy_proxy:
                self.assertEqual({j['position'] for j in result['jobs']},{2,3})
                self.assertTrue(all(j['head']==1 for j in result['jobs']))
            bank=json.loads((root/'next/source/bank.json').read_text())['batches'][str(n)]
            for (h,o),records in heads.items():
                saved=json.loads((root/('next/source/n'+str(n)+'/head/'+str(h)+'_'+str(o)+'/head_records.json')).read_text())
                self.assertEqual(saved['records'],[] if legacy_proxy else records)
                if eta is not None and not legacy_proxy:
                    for row in records:
                        self.assertEqual(bank['eta'][o*4+row['position']][h],row['eta_label'])

    def test_risk_extension_only_registers_unused_cached_quantiles(self):
        spec=dict(output='bank',cases=[dict(name='source_n1',state='source',n=1,head_counts=[2,4])])
        old=dict(jobs=[dict(name='source_n1_o0_h1_p1')],cpu_seconds=12.)
        payload={'plan':spec,'bank/source/preparation.json':old,
            'bank/source/bank.json':dict(origins=[0],batches={'1':dict(count=4)}),
            'bank/source/n1/head/0_0/head_records.json':dict(records=[{},{}]),
            'bank/source/n1/head/1_0/head_records.json':dict(records=[{},{}])}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'bank/source').mkdir(parents=True)
            with patch.object(extension,'ROOT',root), \
                    patch.object(extension,'read',side_effect=lambda p:payload[str(Path(p).relative_to(root))] if Path(p).is_absolute() else payload[p]), \
                    patch.object(extension,'restore') as restore:
                extension.extend('plan');restore.assert_not_called()
            result=json.loads((root/'bank/risk32_registered_jobs.json').read_text())
            self.assertEqual([j['position'] for j in result['jobs']],[2,3])
            prepared=json.loads((root/'bank/source/preparation.json').read_text())
            self.assertEqual(prepared['cpu_seconds'],12.);self.assertEqual(prepared['jobs'][0],old['jobs'][0])

    def test_batch_labels_use_bank_count_and_keep_identity_head_pairs(self):
        bank = dict(batches={'1':dict(count=4,labels=np.arange(16).reshape(8,2).tolist(),eta=None),
            '4':dict(count=8,labels=np.arange(32).reshape(16,2).tolist(),
                eta=(100+np.arange(32).reshape(16,2)).tolist())})
        y, eta = head_labels(bank, 1, 1, 0)
        np.testing.assert_array_equal(y, [8,10,12,14]);self.assertIsNone(eta)
        y, eta = head_labels(bank, 4, 1, 1)
        np.testing.assert_array_equal(y, np.arange(17,32,2))
        np.testing.assert_array_equal(eta, y+100)

    def test_assemble_preserves_admitted_rules_and_rejects_overwritten_positions(self):
        actions = [{'kind':'direct','primary':0}]
        old = [dict(position=0,label=.2,raw={'18':{'statistics':[[[.7,0,0,0]]]}})]
        snapshot = copy.deepcopy(old)
        piece = dict(actions=actions,records=[dict(position=1,label=.3,raw={'10':{}})])
        merged = merge_records(old, [piece], actions, 2)
        self.assertEqual(merged[0], snapshot[0]);self.assertEqual(old, snapshot)
        with self.assertRaises(AssertionError): merge_records(old,[piece,piece],actions,2)
        with self.assertRaises(AssertionError): merge_records(old,[],actions,2)

    def test_server_snapshot_restores_paid_state_without_source_replay(self):
        payload = (dict(paid_episodes=2,weights=[.7,.3]),dict(shortlists=[[0,1,2]]),
            dict(arm='source_structural_mixture'),dict(source_projection_error=0.))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'posterior.pkl'
            with path.open('wb') as f:pickle.dump(payload,f)
            with patch.object(batch_kg,'ROOT',Path(directory)), patch.object(batch_kg,'context') as context:
                result = batch_kg.restore(dict(posterior_inputs={'source':'posterior.pkl'}),'source')
                self.assertEqual(result,payload);context.assert_not_called()

    def test_policy_refinement_registers_the_objective_head_too(self):
        def row(values):
            rules={str(p):dict(statistics=[[[v,0,0,0]]],costs=[[356]],
                predictive_log_evidence_max_error=0.) for p,v in values.items()}
            return dict(raw=rules,projected=copy.deepcopy(rules),head_log_scores=[0.])
        job=dict(name='source_n4_o0_h0_p0',case='source_n4',origin=0,head=0,ordinal=0)
        other=dict(job,name='source_n4_o0_h1_p0',head=1)
        case=dict(name='source_n4',state='source',objective_prefixes=[1,1])
        data=[dict(records=[row({14:.6,16:.7})]),dict(records=[row({14:.9,16:.9})])]
        spec=dict(parent_plan='parent',output='precision')
        parent=dict(output='original',predictive_prefixes=[1,1],maximum_action_resolution_difference=1e-4)
        payload={'plan':spec,'parent':parent,'precision/registered_jobs.json':dict(jobs=[job,other]),
            'precision/shards/'+job['name']+'/results.json':dict(complete=True),
            'precision/shards/'+other['name']+'/results.json':dict(complete=True),
            'original/source/bank.json':dict(prior_weights=[1.])}
        with tempfile.TemporaryDirectory() as directory, patch.object(policies,'ROOT',Path(directory)), \
                patch.object(policies,'read',side_effect=payload.__getitem__), \
                patch.object(policies,'current_head',return_value=(case,data)):
            jobs=policies.prepare(18,'plan')
        self.assertEqual(len(jobs),1);self.assertEqual(jobs[0]['head'],0)
        self.assertTrue(jobs[0]['raw']);self.assertEqual(jobs[0]['target_power'],18)

    def test_parallel_policy_groups_match_serial_raw_and_native_batch_records(self):
        fields=[(np.full(3,v),np.eye(3)*.02+.003) for v in (1.,-.2,-4.,-4.)]
        model=JointResponseMixture(['first','second'],[paired_fields(fields),paired_fields(fields)])
        protocol=dict(objective_reference=2.,chance_alpha=.05,familywise_delta=.05,
            comparison_budget=8,comparison_delta=.05)
        terminal=ConditionalTerminalActions([[1,0,2],[2,0,1]],[1,2],protocol)
        for head,raw_needed in ((0,True),(1,True),(1,False)):
            labels={0:1.1 if head==0 else -.13,1:.9 if head==0 else -.25};eta={0:-4.1,1:-3.8}
            actions=[]
            old=evaluate_head_records(model,terminal,360,0,head,labels,[0,1],powers=[4,5],
                projected_powers=[4,5],seed=714,guarded_mean_position=0,replicates=4,eta_labels=eta,
                checkpoint=lambda r,a:actions.extend(a) if not actions else None)
            expected=evaluate_head_records(model,terminal,360,0,head,labels,[0,1],powers=[6] if raw_needed else [],
                projected_powers=[6],seed=714,guarded_mean_position=0,replicates=4,eta_labels=eta,completed=old)
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);jobs=[]
                for p,row in enumerate(old):
                    job=dict(name='record'+str(p),case='source_n4',origin=0,head=head,ordinal=p,
                        raw=raw_needed,native=True,target_power=6);jobs.append(job)
                    path=root/'precision/shards'/job['name'];path.mkdir(parents=True)
                    (path/'head_records.json').write_text(json.dumps(dict(records=[row],actions=actions)))
                path=root/'precision/policy6';path.mkdir()
                (path/'registered_jobs.json').write_text(json.dumps(dict(shards=2,groups=[dict(name='group0',jobs=jobs)])))
                spec=dict(parent_plan='parent',output='precision',maximum_case_cpu_seconds=420,
                    cases=[dict(name='source_n4',state='source',n=4)])
                payload={'plan':spec,'parent':dict(query=0,seed=714)}
                def read(path):
                    return payload[path] if path in payload else json.loads((root/path).read_text())
                from scipy.stats import qmc
                constructor=qmc.Sobol
                def caller_sobol(*args,**kwargs):
                    self.assertIs(threading.current_thread(),threading.main_thread())
                    return constructor(*args,**kwargs)
                with patch.object(policies,'ROOT',root),patch.object(policies,'read',side_effect=read), \
                        patch.object(policies,'restore',return_value=(model,terminal,None,None)), \
                        patch.object(qmc,'Sobol',side_effect=caller_sobol):
                    for shard in (0,1):policies.run_group('group0',6,shard,2,'plan')
                for p,row in enumerate(expected):
                    pieces=[json.loads((root/'precision/policy6'/('record'+str(p))/('shard'+str(s))/'head_records.json').read_text()) for s in (0,1)]
                    for piece in pieces:self.assertEqual(piece['records'][0]['head_log_scores'],row['head_log_scores'])
                    for kind in (('raw','projected') if raw_needed else ('projected',)):
                        got=policies.merge_rule(actions,pieces,kind,6);target=row[kind]['6']
                        np.testing.assert_allclose(got['statistics'],target['statistics'],atol=1e-13,rtol=0.)
                        np.testing.assert_allclose(got['costs'],target['costs'],atol=1e-13,rtol=0.)
                        self.assertEqual({k:v for k,v in got.items() if k not in ('statistics','costs')},
                            {k:v for k,v in target.items() if k not in ('statistics','costs')})
                for s in (0,1):
                    account=json.loads((root/'precision/policy6/groups/group0'/('shard'+str(s))/'results.json').read_text())
                    self.assertTrue(account['complete']);self.assertEqual(account['completed_records'],2)
        self.assertEqual(model.paid_episodes,0)


if __name__=='__main__':unittest.main()
