import argparse,base64,concurrent.futures,json,re,shlex,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/home/erzhu419/mine_code/KG_op/SC-OLH-KG')
from performance.launch_historical_component_bank import CLI,DEPLOY,PYTHON,REMOTE,ROOT,QUEUE,refresh_jobs
sys.path.insert(0,str(Path(CLI).parent))
import scheduler

def invoke(command,node='node004',timeout=55):
 rc,out,err=scheduler.run_on(node,'cd '+shlex.quote(REMOTE)+' && '+command,timeout=timeout,check=False)
 if rc:
  print(out,end='');raise RuntimeError(err or out)
 return out

def execute_source(source):
 encoded=base64.b64encode(source.encode()).decode()
 return invoke(shlex.join(['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=.',PYTHON,'-c','import base64;exec(base64.b64decode('+repr(encoded)+'))']))

def load(path):return json.loads((ROOT/path).read_text())
def save(path,data):
 path=ROOT/path;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2)+'\n')

p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
if a.action=='sync':
 from performance.continue_historical_raw_axis_refinement import sync
 sync(a.args);print(json.dumps(dict(synced=a.args)))
elif a.action=='remote':
 print(invoke(shlex.join(['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=.',PYTHON,'-u','-m']+a.args)),end='')
elif a.action=='code':print(execute_source(Path(a.args[0]).read_text()),end='')
elif a.action=='fetch':
 source='import json,time\nfrom pathlib import Path\npaths='+repr(a.args)+'\nprint(json.dumps(dict(files={p:Path(p).read_text() for p in paths},reader_process_cpu_seconds=time.process_time())))'
 result=json.loads(execute_source(source))
 for path,content in result['files'].items():
  target=ROOT/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(content)
  print(json.dumps(dict(fetched=path,bytes=len(content.encode()))))
 print(json.dumps(dict(reader_process_cpu_seconds=result['reader_process_cpu_seconds'])))
elif a.action=='submit':
 plan=a.args[0];spec=load(plan);jobs_path=spec['output']+'/jobs.json';jobs=load(jobs_path) if (ROOT/jobs_path).exists() else []
 for j in spec['jobs']:
  command=shlex.join(['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=.',PYTHON,'-u','-m','performance.refine_historical_primary_incumbent_caps','run','--plan',plan,'--identity',str(j['identity']),'--shard',str(j['shard'])])
  description=f'KG primary{spec["primary_policy"]}/incumbent45 {j["identity"]} raw {j["records"][0]}'
  argv=[sys.executable,CLI,'submit','--description',description,'--cmd',command,'--cwd',DEPLOY,'--signature',f'KG_op/{Path(plan).stem}/{j["identity"]}/raw/{j["shard"]}','--project','KG_op','--vram','0','--ram-mb','2000','--cpu','1','--priority','low','--skip-launch-staging','--env-spec','none']
  for n in range(1,7):argv+=['--allowed-node',f'node{n:03d}']
  r=subprocess.run(argv,text=True,capture_output=True)
  if r.returncode:raise RuntimeError(r.stderr or r.stdout)
  item=dict(id=re.search(r'submitted (t\d+)',r.stdout).group(1),identity=j['identity'],shard=j['shard'],kind='raw',cmd=command,args=argv,declared_limit=j['maximum_worker_cpu_seconds'])
  jobs.append(item);save(jobs_path,jobs);print(json.dumps({k:item[k] for k in ('id','identity','shard','declared_limit')}),flush=True)
elif a.action=='dispatch':
 r=subprocess.run([sys.executable,CLI,'dispatch','--no-bulk-window','--lock-timeout','20']+sum([['--task-id',tid] for tid in a.args],[]),text=True,capture_output=True)
 print('\n'.join(line for line in r.stdout.splitlines() if line.startswith('=== dispatch') or any(tid in line for tid in a.args)))
 if r.stderr:print(r.stderr,file=sys.stderr)
 if r.returncode:raise RuntimeError(r.stdout)
elif a.action=='status':
 refresh_jobs(set(a.args));q=json.loads(QUEUE.read_text());print(json.dumps([{k:t.get(k) for k in ('id','status','node','log_path','parent_id','exit_code')} for t in q['tasks'] if t['id'] in a.args or t.get('parent_id') in a.args]))
elif a.action=='resources':
 plan,*ids=a.args;q=json.loads(QUEUE.read_text());tasks=[t for t in q['tasks'] if t['id'] in ids or t.get('parent_id') in ids]
 def get(t):
  code='from pathlib import Path;print("\\n".join(s for s in Path('+repr(t['log_path'])+').read_text().splitlines() if s.startswith(("SCHEDULEURM_RESOURCE ","SCHEDULEURM_DONE "))))'
  out=invoke(shlex.join([PYTHON,'-c',code]),node=t.get('node') or t['last_node'],timeout=25)
  return dict(id=t['id'],node=t.get('node'),status=t['status'],parent_id=t.get('parent_id'),resources=[json.loads(line.split(' ',1)[1]) for line in out.splitlines() if line.startswith('SCHEDULEURM_RESOURCE ')],done=[json.loads(line.split(' ',1)[1]) for line in out.splitlines() if line.startswith('SCHEDULEURM_DONE ')])
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:result=list(pool.map(get,tasks))
 save(load(plan)['output']+'/worker_resources.json',result);print(json.dumps(result))
