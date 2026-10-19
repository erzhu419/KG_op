"""Register two new observation streams on the frozen Stage65 geometries."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

NAME = 'mainline_terminal_two_comparison_seed_replication_v1'
MANIFEST = ROOT / f'performance/manifests/{NAME}.json'
REFERENCE = 'performance/manifests/mainline_terminal_two_comparison_confirmation_v1.json'
PROTOCOL = 'docs/mainline_terminal_two_comparison_seed_replication_protocol_20261017.md'


def plans(reference, registered_utc):
    """Build metadata only; geometry, optimizer seeds and policies are not redrawn."""
    output = f'paper_artifacts/{NAME}'
    children, streams = [], []
    initial_source = Path(reference['replicates'][0]['output_dir'])
    for replicate, observation_base, verification_base in ((1,2026105100,2026105200),
                                                          (2,2026106100,2026106200)):
        name = f'mainline_terminal_two_comparison_seed_replication_stream_{replicate}_v1'
        path = f'performance/manifests/{name}.json'
        child = deepcopy(reference)
        for key in ('execution_started_utc','completed_utc','results_document','summary_file',
                    'completion_file','next_step','execution_failures'):
            child.pop(key,None)
        directory = f'{output}/stream_{replicate}'
        seed_map = {}
        for number,task in enumerate(child['tasks'],1):
            seed = task['geometry_task_seed']
            task_directory = initial_source / task['domain'] / 'search' / f'seed_{seed}'
            task.update(initial_policy_file=str(task_directory/'initialization.json'),
                shared_neighbors_file=str(task_directory/'neighbors.npz'))
            seed_map[str(seed)] = dict(observation_seed=observation_base+number,
                verification_seed=verification_base+number)
        child.update(name=name,status='registered_not_started',registered_utc=registered_utc,
            execution_started=False,parent_manifest=REFERENCE,output_dir=directory,
            reference_manifest=REFERENCE,observation_replicate=replicate,
            objective='Confirm frozen AFTER propagation and paired terminal gains under a new observation/verification stream on the same24 Stage65 geometries; no model, threshold or geometry selection.',
            scope='Within-geometry observation/verification replication; not24 new geometries.',
            geometry_description='Reuse every Stage65 physical geometry, optimizer seed, initial10 policy vector and shared64 neighbor vector; draw only the newly registered observation/verification streams.',
            replicates=[dict(replicate=replicate,output_dir=f'{directory}/replicate_{replicate}',
                seed_by_task=seed_map)],
            statistical_unit='24 fixed geometries. The outer analysis averages the two new-stream effects within each geometry before uncertainty summaries; stream and arm cells are not independent geometry samples.',
            historical_accounts=[*reference['historical_accounts'],
                f"{reference['output_dir']}/completion.json",
                'paper_artifacts/mainline_confirmation_attribution_v1/completion.json'],
            protocol_document=PROTOCOL,new_target_calls_at_registration=0,truth_used_at_registration=False)
        child['frozen'].update(initial_policies='Reuse exact Stage65 initial10 physical policies from initial_policy_file; ignore their old raw responses and draw10 fresh shared observations.',
            shared_neighbors='Reuse exact Stage65 shared64 physical neighbor bank and metadata from shared_neighbors_file; no new bank draw.',
            initial_design_seed=80)
        child['randomness']['optimizer_seed'] = 'Preserve each Stage65 geometry_task_seed=optimizer_seed4001..4024 as config.seed; observation seeds never replace optimizer seeds.'
        child['barriers'].update(before_sampling='Freeze both new stream maps, original geometry/optimizer identities, model references and reused initial/neighbor policy files before any sampling.',
            before_online='Both streams must complete all48 initialization histories and freeze all48 reused neighbor banks before any adaptive search.',
            before_verification='Both streams must seal all336 histories or retained failures and all432 paired terminal inputs before terminal simulation.',
            before_scoring='Both streams must seal all864 old/new terminal outcomes or failures before any new scoring or target-truth access.')
        child['execution_preparation'] = dict(status='pending',workers=None,
            entrypoint='performance/run_mainline_two_comparison_confirmation.py',
            execution_policy='scheduleurm submit-cpu-batch node001-node006, vram=0, one BLAS thread per worker',
            inherited_preflight=deepcopy(reference['execution_preparation']['inherited_preflight']),
            inherited_seed_routing=deepcopy(reference['execution_preparation']['inherited_seed_routing']),
            inherited_reuse_checks=deepcopy(reference['execution_preparation']['inherited_reuse_checks']))
        child['analysis'].update(fixed_size='Complete this24-task stream and the paired second stream without interim scoring, optional expansion or model/threshold changes.',
            truth='Only after both streams are globally sealed, merge deterministic Stage65 and already scored new-stream truth for the same geometry and exact physical policy. Fill task truth caches only during scoring; evaluate only missing terminal-union policies and charge new analytic truth calls separately.',
            interpretation='Frozen-geometry noise replication; source and opened Stage65 outcomes are excluded from fresh-effect means. Same352 caps do not imply same actual costs.')
        children.append((path,child))
        streams.append(dict(name=f'observation_{replicate}',replicate=replicate,manifest=path))
    outer = dict(name=NAME,status='registered_not_started',registered_utc=registered_utc,
        execution_started=False,reference_manifest=REFERENCE,output_dir=output,streams=streams,
        tasks=deepcopy(reference['tasks']),terminal_arms=deepcopy(reference['terminal_arms']),
        search_arms=deepcopy(reference['search_arms']),
        objective='Test whether frozen AFTER new-minus-old terminal gains and the secondary propagation contrasts survive two fresh observation/verification streams.',
        scope='24 fixed Stage65 geometries,8per known family,two newly registered streams; no new geometry sample.',
        statistical_unit=children[0][1]['statistical_unit'],
        frozen=dict(model_changes=False,source_fits=0,new_source_calls=0,new_geometry_draws=0,
            d=1000,initial_design_seed=80,source_protocol=REFERENCE,
            optimizer_seeds='Exact original geometry_task_seed4001..4024 in both streams',
            old_and_new_terminals='Unchanged Stage65 rules, shortlisting and352-call ceiling'),
        budget=dict(geometry_count=24,observation_replicates=2,task_stream_units=48,
            unique_search_histories=336,terminal_cells_per_protocol=432,registered_terminal_cells=864,
            initial_physical_calls=480,online_search_physical_calls=864,
            maximum_verification_calls_per_cell=352,maximum_unique_terminal_calls_per_paired_cell=368,
            maximum_verification_physical_calls=158976,maximum_target_physical_calls=160320,
            maximum_logical_target_calls_both_protocols=315072,
            new_source_calls=0,new_source_fits=0),
        barriers=deepcopy(children[0][1]['barriers']),
        analysis=dict(primary='AFTER paired new-minus-old terminal C and signedG, averaged across two new streams within each geometry; report both without substitution and retain each stream separately.',
            secondary='AFTER-minus-POINT and FULL-minus-AFTER under each frozen terminal and changes in these contrasts; report shared-neighbor and dynamic-policy contributions separately.',
            paired_terminal='AFTER new-minus-old C and signedG, including tradeoffs and largest-task concentration.',
            controls='Retain all9 arms, FULL-minus-AFTER, same-source GP, native terminal crosses, STOP, all failures and fees.',
            contributions='Report shared-neighbor and dynamic policy contributions separately, including zero/negative effects and uncertified/false-certified outcomes.',
            truth=children[0][1]['analysis']['truth'],
            missing='Keep undefined G null; distinguish complete24-geometry estimates from the explicitly defined subset. Never fill an undefined baseline with zero.',
            uncertainty='Descriptive geometry-level paired95%t intervals; streams are paired repetitions, not48 independent geometry samples.'),
        historical_accounts=[f"{reference['output_dir']}/completion.json",
            'paper_artifacts/mainline_confirmation_attribution_v1/completion.json'],
        protocol_document=PROTOCOL,new_target_calls_at_registration=0,truth_used_at_registration=False)
    return outer,children


def register():
    assert not MANIFEST.exists(), 'Preserve the existing registration'
    reference = read(ROOT/REFERENCE)
    assert reference['status']=='complete'
    outer,children = plans(reference,datetime.now(timezone.utc).isoformat())
    new_seeds = [value for _,child in children for row in child['replicates'][0]['seed_by_task'].values()
                 for value in row.values()]
    assert len(new_seeds)==len(set(new_seeds))==96
    proposed = set(new_seeds)

    def check_seeds(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if 'seed' in key and isinstance(item,int):
                    assert item not in proposed, f'Observation/verification seed already registered: {item}'
                check_seeds(item)
        elif isinstance(value,list):
            for item in value:
                check_seeds(item)
    for path in (ROOT/'performance/manifests').glob('mainline*.json'):
        check_seeds(read(path))
    for path,child in children:
        assert not (ROOT/path).exists()
        for task in child['tasks']:
            assert task['optimizer_seed']==task['geometry_task_seed']
            for field in ('initial_policy_file','shared_neighbors_file'):
                assert (ROOT/task[field]).is_file(),task[field]
    output = ROOT/outer['output_dir']
    with attempt(output/'registration','register-frozen-geometry-seed-replication') as cost:
        cost.update(new_target_initial_calls=0,new_source_fits=0,new_truth_function_calls=0,
            source_model_restorations=0,posterior_restorations=0,new_rng_draws=0,
            geometry_uniform_draws=0,download_bytes=0)
        for path,child in children:
            save(ROOT/path,child)
        save(MANIFEST,outer)
    return outer


if __name__=='__main__':
    print(register()['status'])
