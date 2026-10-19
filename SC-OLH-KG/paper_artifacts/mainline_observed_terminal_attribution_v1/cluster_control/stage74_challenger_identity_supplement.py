"""Select saved original-shortlist identities; do not rerun attribution."""
import argparse
import json
from pathlib import Path


def extract(directory):
    rows = []
    for path in sorted((directory/'cases').glob('seed_*/compact.json')):
        task = json.loads(path.read_text())['shortlists']
        for arm in task['arms']:
            saved = arm['objective_challenger']
            identity = saved.get('policy_id')
            roles = [r for r in arm['original_shortlist'] if r.get('role') in (
                'posterior_objective_primary', 'posterior_objective_verification_challenger')]
            resolved = roles[0] if identity is None and saved.get('status') == 'selected' and len(roles) == 1 else None
            rows.append(dict(seed=task['seed'], arm=arm['arm'], status=saved.get('status'),
                previous_policy_id=identity,
                policy_id=resolved['policy_id'] if resolved else identity,
                identity_source='saved_original_shortlist_role' if resolved else 'saved_challenger_identity',
                original_shortlist_role=None if resolved is None else resolved['role'],
                original_shortlist_position=None if resolved is None else resolved['position']))
    return dict(scope='Identity-only saved-field supplement; original analysis remains unchanged.',
        rows=rows, metrics_changed=False, final_score_values_changed=False,
        simulation_calls=0, truth_calls=0, posterior_restorations=0, random_draws=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(extract(args.directory), separators=(',', ':'))+'\n')
