"""Freeze the existing distinct-challenger terminal without running its predecessor."""
from datetime import datetime, timezone

from performance import run_mainline_terminal_equal as verifier
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.run_mainline_same_policy import transform
from performance.run_mainline_terminal_confirmation import input_record


PROTOCOL = 'same_policy_distinct_challenger'


def prepare(source_plan, terminal_plan):
    """Read fresh saved histories and freeze inputs for the unchanged verifier.

    ``terminal_plan`` uses the usual family cell schema, with only ``PROTOCOL``
    in ``protocols`` and an ``inherited_replay_check_file``.  No predecessor
    verification, paid raw, engine restoration or target call is needed.
    """
    source = ROOT / source_plan['output_dir']
    output = ROOT / terminal_plan['output_dir']
    assert terminal_plan['protocols'] == {PROTOCOL: [80, 128, 128]}
    inherited = terminal_plan['inherited_replay_check_file']
    assert read(ROOT / inherited)['status'] == 'passed'
    cells = {cell['name']: cell for cell in terminal_plan['arms']}
    active = verifier.cases(terminal_plan)
    with attempt(output, 'freeze-modified-only-terminal-inputs', source_plan['domain']) as cost:
        cost.update(new_source_fits=0, new_truth_function_calls=0, download_bytes=0)
        records = []
        for seed, arm, protocol in active:
            cell = cells[arm]
            directory = source / f'seed_{seed}' / cell['history_arm']
            search = read(directory / 'search.json')
            frozen = (read(directory / 'native_terminal.json')['terminal']
                      if cell['terminal_backend'] == 'native' else search['terminal'])
            item = transform(input_record(search, frozen, source_plan, cell, seed,
                terminal_plan['verification_seed_by_task'][str(seed)],
                str(directory.relative_to(ROOT)), tau=0.))
            assert item['protocol'] == protocol
            records.append((verifier.case_dir(output, seed, arm, protocol) / 'input.json', item))
        for path, item in records:
            if path.exists():
                assert read(path) == item, 'Frozen modified terminal input changed'
            else:
                save(path, item)
        save(output / 'prepared.json', dict(status='complete', case_count=len(records),
            protocol=PROTOCOL, truth_used=False, new_simulation_calls=0,
            frozen_utc=datetime.now(timezone.utc).isoformat()))
        save(output / 'replay_check.json', dict(status='passed', inherited=inherited,
            new_raw_replay=False,
            scope='Inherited unchanged verifier; new input transform validated separately with stubs'))

