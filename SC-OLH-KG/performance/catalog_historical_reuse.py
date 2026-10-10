"""Read the complete indexed history and catalog implementation families."""

import ast
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
INDEX = ROOT/'paper_artifacts/historical_component_reuse_20261007/history_index.json'
OUT = ROOT/'paper_artifacts/historical_complete_reuse_v1'
FAMILIES = {
    'state_and_trajectory': r'encoder|observable|exposure|profile|trajectory',
    'orthogonal_sparse_low_order': r'spectral|additive|sparsity|orthogonal|meta_prior',
    'source_hierarchy': r'source_(?:common|task|joint|population|risk|kernel|function)|exchangeable',
    'raw_posterior': r'posterior|replicat|raw_density|raw_filter|raw_source|variance_reference|joint_function_risk',
    'decision_and_verification': r'(?:^|/)(?:kg|.*lookahead|.*terminal|.*deployment)|single_olhkg|decision_backends',
    'numerical_efficiency': r'factorized|vectoriz|rqmc|sobol|cached|nested|torch',
    'matched_baselines': r'baselines/|transfer_external|matched_baseline|causal_prior',
    'theory_and_reproduction': r'^proof/|^.*tests/|theory|repro|artifact|claim|submission',
    'application_contracts': r'problems/|energy|battery|queue|inventory|factor_shock',
}


def summarize(path, content):
    encoding = 'utf-8'
    try:
        source = content.decode(encoding)
    except UnicodeDecodeError:
        # The initial Multiobj-KG MATLAB archive contains GBK comments.
        if not path.endswith('.m'): raise
        encoding = 'gb18030'; source = content.decode(encoding)
    imports, definitions, calls = [], [], set()
    if path.endswith('.py'):
        tree = ast.parse(source, filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imports.append(node.module)
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions.append({'name': node.name, 'line': node.lineno, 'end_line': node.end_lineno})
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name): calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute): calls.add(node.func.attr)
    searchable = path+' '+source
    return {'path': path, 'source_encoding': encoding, 'bytes_read': len(content), 'lines_read': len(source.splitlines()),
        'families': [name for name, pattern in FAMILIES.items() if re.search(pattern, searchable, re.I)],
        'imports': sorted(set(i for i in imports if i)), 'definitions': definitions,
        'numerical_calls': sorted(calls & {'quad_vec','minimize','cho_solve','cho_factor','cholesky',
            'elliptical_step','sample','local_joint_moments','local_raw_moments','posterior_mixture_weights',
            'likelihood_query_lookahead','random_base2','from_moments'}),
        'historical_resolution_markers': sorted(set(re.findall(
            r'(?i)\b(?:replication_only|source_mixture|factorized_rqmc|grouped_task_deconvolved|balanced4|'
            r'raw_likelihood|moment_projection|variance_reference|within_eta_covariance|'
            r'between_eta_mean_covariance|objective_incumbent|familywise_delta)\b',source)))}


def run():
    index = json.loads(INDEX.read_text()); OUT.mkdir(parents=True, exist_ok=True)
    requests = [(kind,row) for kind in ('code_versions','historical_registered_manifests','historical_documents')
        for row in index[kind]]
    queries = ''.join(f"{row['first_commit']}:{row['path']}\n" for _,row in requests).encode()
    batch = subprocess.run(['git','cat-file','--batch'],input=queries,capture_output=True,cwd=REPO,check=True).stdout
    position = 0; historical, contracts, documents = [], [], []
    for kind,row in requests:
        end = batch.index(b'\n',position); header = batch[position:end].split()
        assert header[1] == b'blob',(row['path'],header)
        size = int(header[2]); content = batch[end+1:end+1+size]; position = end+size+2
        if kind == 'code_versions':
            historical.append({'first_commit':row['first_commit'],'date':row['date'],**summarize(row['path'],content)})
        elif kind == 'historical_registered_manifests':
            data = json.loads(content)
            contracts.append({'path':row['path'],'first_commit':row['first_commit'],'bytes_read':size,
                'recorded_contract':{k:data[k] for k in ('status','law','scope','decision','execution','gate','cost') if k in data}})
        else:
            text = content.decode('utf-8')
            documents.append({'path':row['path'],'first_commit':row['first_commit'],'bytes_read':size,
                'outcome_passages':[line for line in text.splitlines()
                    if re.search(r'失败|缺口|修复|不收敛|HOLD|PASS|fail|improv|speedup|vacuous',line,re.I)]})
    assert position == len(batch)
    current = []
    for folder in ('representation','encoders','core','variance','acquisition','algorithms','baselines','problems','performance'):
        for path in sorted((ROOT/folder).rglob('*.py')):
            current.append(summarize(str(path.relative_to(REPO)),path.read_bytes()))
    counts = Counter(row['path'].split('/')[0] for row in historical)
    report = {'indexed_commits':len(index['commits']),'historical_code_versions_fully_read':len(historical),
        'historical_manifest_versions_fully_read':len(contracts),'historical_document_versions_fully_read':len(documents),
        'current_runtime_and_performance_files_fully_read':len(current),'historical_roots':dict(counts),
        'historical_bytes_read':sum(r['bytes_read'] for r in historical+contracts+documents),
        'family_coverage':{name:{'historical_versions':sum(name in r['families'] for r in historical),
            'current_files':sum(name in r['families'] for r in current)} for name in FAMILIES},
        'scope':'Complete content retrieval and algorithm/dependency catalog;manual semantic integration is organized by implementation family. This does not claim a line-by-line manual review of2251 code versions.'}
    for name,value in (('coverage',report),('historical_code',historical),('current_code',current),
                       ('historical_contracts',contracts),('historical_outcomes',documents)):
        (OUT/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    supplement_local_history(index)


def supplement_local_history(index=None):
    """Read text changed by real local reflog snapshots outside branch history."""
    if index is None:index=json.loads(INDEX.read_text())
    known={c['commit'] for c in index['commits']}
    local=set(subprocess.check_output(['git','rev-list','--all','--reflog'],cwd=REPO,text=True).split())
    snapshots=[]
    for commit in sorted(local-known):
        paths=subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r',commit],cwd=REPO,text=True).splitlines()
        rows=[]
        for path in paths:
            if Path(path).suffix not in ('.py','.m','.lean','.md','.tex','.json'):continue
            content=subprocess.check_output(['git','show',commit+':'+path],cwd=REPO)
            row=summarize(path,content) if Path(path).suffix in ('.py','.m','.lean') else {
                'path':path,'bytes_read':len(content),'lines_read':len(content.decode().splitlines())}
            rows.append(row)
        snapshots.append({'commit':commit,'changed_text_versions_fully_read':rows})
    (OUT/'additional_local_history.json').write_text(json.dumps({'origin':'local reflog outside branch graph','snapshots':snapshots},indent=2)+'\n')
    path=OUT/'coverage.json';report=json.loads(path.read_text())
    report.update(additional_reflog_commits_fully_read=len(snapshots),total_local_historical_commits_covered=len(known|local),
        additional_reflog_changed_text_versions_fully_read=sum(len(s['changed_text_versions_fully_read']) for s in snapshots))
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    run()
