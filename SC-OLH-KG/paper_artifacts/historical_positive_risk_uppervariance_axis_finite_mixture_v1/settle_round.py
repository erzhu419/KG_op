import json
import shutil
from pathlib import Path

root = Path('/home/erzhu419/mine_code/KG_op/SC-OLH-KG')
name = 'historical_positive_risk_uppervariance_axis_finite_mixture_v1'
out = 'paper_artifacts/' + name
plan = 'performance/manifests/' + name + '.json'
parent_path = 'paper_artifacts/historical_positive_risk_uppervariance_axis_prior_f16_source_v1/completion.json'
folder = root / out
folder.mkdir(parents=True, exist_ok=True)

def read(path):
    return json.loads((root / path).read_text())

def save(path, value):
    (root / path).write_text(json.dumps(value, indent=2) + '\n')

parent = read(parent_path)
assert not (folder / 'completion.json').exists(), 'This settled phase must not be charged twice.'
attempt = json.loads(Path('/tmp/kg_finite_mixture_transfer_output.txt').read_text().splitlines()[-1])
assert not attempt['complete'] and attempt['error_type'] == 'TimeoutError'
fees = {}
for filename in ('kg_finite_mixture_first_read', 'kg_finite_mixture_code_read', 'kg_finite_mixture_writer_read'):
    evidence = json.loads(Path('/tmp/' + filename + '.json').read_text())
    fees[filename + '_process_cpu_seconds'] = evidence['process_cpu_seconds']
    save(out + '/' + filename + '.json', evidence)
fees['failed_matched_reader_process_cpu_seconds'] = attempt['reader_process_cpu_seconds']
fees['small_local_text_settlement_and_verification_conservative_cpu_seconds'] = 0.05
charge = sum(fees.values())
charged = parent['shared_reserve_charged_cpu_seconds'] + charge
remaining = 600 - charged

spec = dict(
    output=out, source_phase_completion=parent_path,
    source_bound_plan=parent['followup_bound_plan'], followup_bound_plan=parent['followup_bound_plan'],
    diagnostic_sources=['../markdown/GPT_diagnosis_v4.md', '../markdown/GPT6_action_competition_diagnosis_20261013.md'],
    shared_phase_reserve_cpu_seconds=600,
    inherited_shared_reserve_charged_cpu_seconds=parent['shared_reserve_charged_cpu_seconds'],
    new_phase_reserve_cpu_seconds=0, requested_reader_prof_timer_cpu_seconds=0.18,
    estimated_worker_cpu_seconds=0, estimated_phase_cpu_seconds=parent['estimated_phase_cpu_seconds'],
    source_phase_registered_estimated_cpu_seconds=parent['estimated_phase_cpu_seconds'],
    maximum_estimated_cpu_seconds=3600,
    scope='Read-only paid fixed355 F16/risk224 transfer diagnostic. No integration, bank replacement or new reference. The timed reader failed; no numerical transfer result is admitted.')
algebra = dict(
    code_sources=[dict(file='core/raw_terminal_head_grid.py', lines=[218, 229]),
                  dict(file='performance/prepare_historical_positive_risk_screen.py', lines=[101, 104])],
    origin_measure='Q_o uses objective and risk nodes conditional on the same origin o; external weight is the unrenormalized prior mass pi_o.',
    posterior='W_k(f,g)=pi_k*p_k(f)*p_k(g)/sum_r(pi_r*p_r(f)*p_r(g))',
    fixed_action='H_k(f,g)=F_k(f)*G_k(g); same fixed action355 throughout.',
    same_node_identity='A-B=sum_o pi_o Q_o[sum_{k!=o} W_k*(H_k-H_o)]',
    baseline_decomposition='A-C_prior=(A-B)+(B-C_prior), with B evaluated on the identical finite nodes.',
    source_weight_construction_correct=True, additional_density_ratio_required=False,
    prior_weights_renormalized=False, fixed_diagonal_can_replace_bank_maximum=False,
    full_continuous_identity_transfer_cancels=True,
    finite_origin_quadrature_transfer_need_not_cancel=True,
    numerical_diagnostic_complete=False, same_node_fixed_reproduction_checked=False,
    same_node_mix=None, same_node_diagonal=None, identity_transfer_values=None,
    limitation='Only source weighting and algebra are established. The failed reader did not reproduce the current fixed value or quantify either transfer. Tower residual attribution and absolute error certification remain open.')
failure = dict(attempt,
    requested_prof_timer_cpu_seconds=0.18,
    observed_prof_timer_overshoot_cpu_seconds=attempt['reader_process_cpu_seconds'] - 0.18,
    timing_explanation='The requested profiling timer did not interrupt the reader by0.18CPU seconds. Exact operation at delivery was not recorded.',
    retry_performed=False, recovery='Stopped scientific reads and retries; settled the measured failure in full, retaining a negative original reserve balance.')
save(plan, spec)
save(out + '/support_cpu.json', fees)
save(out + '/reader_failure.json', failure)
save(out + '/algebra.json', algebra)
shutil.copyfile('/tmp/kg_finite_mixture_transfer_read.py', folder / 'read_matched_transfer.py')
shutil.copyfile('/tmp/kg_settle_finite_mixture_round.py', folder / 'settle_round.py')

d = dict(parent)
d.update(
    complete=True, accounting_settled=True, diagnostic_complete=False,
    state='finite_mixture_diagnostic_failed_budget_exceeded',
    source_phase_completion=parent_path, source_bound_plan=spec['source_bound_plan'],
    worker_process_cpu_seconds=0, failed_worker_process_cpu_seconds=0,
    failed_support_reader_process_cpu_seconds=attempt['reader_process_cpu_seconds'],
    collection_and_support_cpu_seconds=charge, total_recorded_process_cpu_seconds=charge,
    inherited_shared_reserve_charged_cpu_seconds=parent['shared_reserve_charged_cpu_seconds'],
    additional_shared_reserve_charge_cpu_seconds=charge,
    shared_reserve_charged_cpu_seconds=charged, shared_reserve_remaining_cpu_seconds=remaining,
    shared_reserve_budget_exceeded=True, shared_reserve_overrun_cpu_seconds=-remaining,
    new_phase_reserve_cpu_seconds=0,
    previous_cap_refinement_recorded_cpu_seconds=parent['cumulative_cap_refinement_recorded_process_cpu_seconds'],
    cumulative_cap_refinement_recorded_process_cpu_seconds=parent['cumulative_cap_refinement_recorded_process_cpu_seconds'] + charge,
    arithmetic_cached_prefix_combinations=0, paid_scalar_head_values_reused=0,
    same_node_fixed_reproduction_checked=False, finite_mixture_transfer_residual=None,
    additional_density_ratio_required=False,
    new_reference_records=0, new_admitted_joint_action_references=0,
    precision_gate_retested=False, numerical_containment_retested=False,
    new_joint_integrations=0, new_pair_rule_evaluations_completed=0,
    computed_guard_statistic_rule_evaluations=0, reconstructed_guard_statistic_rule_evaluations=0,
    new_physical_calls=0, new_source_fits=0, new_verifier_calls=0,
    independent_confirmation=False, scientific_reads_stopped=True, retry_performed=False,
    remote_ledger_synced=False,
    costs_scope='Three timed local source readers plus the full failed remote reader, and0.05CPU conservative local text/settlement/verification support. All inherited failures and recoveries retained. Negative same600 reserve is recorded without truncation, reset or transfer from worker capacity.',
    scope=spec['scope'])
for filename in ('completion.json', 'status.json'):
    save(out + '/' + filename, d)

status_path = 'paper_artifacts/historical_positive_risk_uppervariance_axis_continuation_v1/status.json'
status = read(status_path)
status.update(
    state='bound_refinement_hold', bound_refinement_completion=out + '/completion.json',
    bound_refinement_recorded_process_cpu_seconds=d['cumulative_cap_refinement_recorded_process_cpu_seconds'],
    recorded_reference_campaign_plus_bound_refinement_cpu_seconds=status['recorded_process_cpu_seconds'] + d['cumulative_cap_refinement_recorded_process_cpu_seconds'],
    latest_bound_shared_reserve_charged_cpu_seconds=charged,
    latest_bound_shared_reserve_remaining_cpu_seconds=remaining,
    latest_bound_shared_reserve_budget_exceeded=True,
    latest_finite_mixture_diagnostic=out + '/algebra.json',
    latest_finite_mixture_reader_failure=out + '/reader_failure.json',
    hold_reason='Origin-conditional prior weighting is correct;no extra density ratio is required. Matched F16/risk224 reader timed out before a numerical result. Original600 support reserve is overrun;negative balance and full failure fee retained. Scientific reads/retries stopped;no new reserve or worker-capacity transfer. G and67/81groups,43accepted units,3pair HOLDs and1reward HOLD remain unchanged;no integration or absolute error certificate.')
status['bound_refinement_completed_phases'].append(plan)
save(status_path, status)

report = root / 'docs/gpt_v4_branch_coupling_review_20261013.md'
heading = '## 10月15日有限身份混合核对与超时结算'
assert heading not in report.read_text()
report.write_text(report.read_text() + f'''
{heading}

沿用前次v4/GPT6诊断。代码与代数确认：同身份条件节点的外权为原prior质量，不需追加mixture density ratio；同节点固定动作的混合差为`Σo πo Qo[Σk≠o Wk(Hk−Ho)]`。分身份固定动作乘积不能替换原bank的动作最大化。

同F16/risk224读取设置0.18 CPU秒timer，却在{attempt['reader_process_cpu_seconds']:.9f}秒才超时退出；未产出混合值、对角值或转移量，tower残差尚未数值定位。失败全额计费、不重试；raw/projected G仍为8.3496104e-5/2.7673058e-4，67/81组及原精度HOLD不变。

本轮费用{charge:.9f} CPU秒（含0.05秒本地结算支持保守费），累计{d['cumulative_cap_refinement_recorded_process_cpu_seconds']:.9f}秒；同600共享余量为{remaining:.9f}秒，即超支{-remaining:.9f}秒。保留全部旧失败/恢复费，无重置、借额或新增积分。当前停止科研读取；本地账链已结算，未追加服务器同步。后续需先解决预算约束，再完成同节点分解。

[代数与证据边界](../{out}/algebra.json)、[失败原始计时](../{out}/reader_failure.json)、[费用账链](../{out}/completion.json)。
''')

# These checks detect a dropped fee or an accidental change to inherited scientific gates.
# On failure, correct the local settlement; do not repeat the scientific reader.
assert charged > 600 and remaining < 0
assert abs(parent['shared_reserve_remaining_cpu_seconds'] - charge - remaining) < 1e-10
assert abs(sum(fees.values()) - d['additional_shared_reserve_charge_cpu_seconds']) < 1e-12
for key in ('comparison', 'inherited_accepted_units', 'inherited_precision_hold_units',
            'inherited_reference_reward_hold_records', 'numerical_containment_passed',
            'followup_bound_plan', 'settled_shared_phase_worker_rows'):
    assert read(out + '/completion.json')[key] == parent[key], key
assert status['latest_bound_diagnostic_plan'] == parent['followup_bound_plan']
assert not d['diagnostic_complete'] and not d['same_node_fixed_reproduction_checked']
print(json.dumps(dict(charge=charge, remaining=remaining,
    overrun=-remaining, cumulative=d['cumulative_cap_refinement_recorded_process_cpu_seconds'],
    settlement_checks_passed=True, new_integrations=0, diagnostic_complete=False)))
