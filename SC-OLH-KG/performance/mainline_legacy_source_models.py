"""Refit the current eight-identity architecture on frozen source observations.

The cross-family mean inputs are the old public ten coordinates; risk inputs
are learned from exactly the same source archive. Source integration is a fixed
1024-node positive Sobol rule, not the previous Inventory 1024-component fit or
its precision certificate. Target updates use the existing native raw filter.
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtri, polygamma
from scipy.stats import qmc

from performance.adapt_historical_source_covariance import coefficient_fits
from performance.connect_historical_matched_gp import fit_source_controls, controls_from_fit
from performance.connect_historical_prior_posterior import source_prior
from performance.prepare_legacy_f_variance_reference import variance_reference, VarianceReferencePosterior
from representation.additive_groups import TransferableAdditiveGroupBank
from representation.exchangeable_mean import centered_fit_covariance_parts
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_function_prior import legacy_public_coordinates, legacy_source_labels
from representation.legacy_observable_prior import LegacyObservableCoordinate, legacy_exposure_descriptors
from representation.meta_prior import LearnedMetaPrior
from representation.transferable_spectral import SourceDomainBatch, TransferableSpectralBasis


HEADS = ('f', 'g', 'eta_g', 'eta_f')
DEFAULTS = {
    'quadrature_power': 10, 'seed': 1622, 'minimum_source_ess': 8.,
    'spectral': dict(active_dim=6, max_library_size=64, low_frequency_components=8,
                     n_neighbors=10, pilot_cv_size=10, pilot_cv_repeats=3,
                     orthogonalization='symmetric'),
    'additive': dict(max_groups=8, max_library_size=64, low_frequency_components=8,
                     n_neighbors=10, strong_heredity=True, max_interactions=2),
    'optimizer': dict(method='L-BFGS-B', max_iterations=512, ftol=1e-9, gtol=1e-5),
    'risk_max_iterations': 80,
}


class SourceFitError(RuntimeError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(diagnostics['failure'])


def _standardize(values):
    center, scale = values.mean(axis=0), values.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    return (values-center)/scale, center, scale


def _normal_log_density(error, covariance):
    factor = cho_factor(covariance, lower=True)
    return float(-.5*(len(error)*np.log(2*np.pi)
        + 2*np.log(np.diag(factor[0])).sum() + error @ cho_solve(factor, error)))


def _law(prior, design, task_covariance, sizes):
    common = np.asarray(prior.covariance)
    blocks = np.split(design, np.cumsum(sizes)[:-1])
    source_covariance = design@common@design.T + block_diag(*[
        d@task_covariance@d.T for d in blocks]) + prior.discrepancy_variance*np.eye(len(design))
    return dict(mean=np.asarray(prior.mean), common=common, task=task_covariance,
                white=prior.discrepancy_variance, source_design=design,
                source_mean=design@prior.mean, source_covariance=source_covariance)


def _eta_rule(law, sample_variances, settings, seed):
    mean, covariance = variance_reference(law['source_mean'], law['source_covariance'], sample_variances)
    uniforms = qmc.Sobol(len(mean), scramble=True, seed=seed).random_base2(settings['quadrature_power'])
    states = mean + ndtri(uniforms) @ np.linalg.cholesky(covariance).T
    # The reference normalizer is needed for comparable outer identity masses.
    from scipy.special import digamma
    labels = np.log(sample_variances)-digamma(1.)
    logz = _normal_log_density(labels-law['source_mean'],
        law['source_covariance'] + float(polygamma(1, 1.))*np.eye(len(mean)))
    return states, logz


def _joint_risk_fit(law, public_mean, sizes, eta, labels, sample_variances, settings):
    """Fit old task scale and new common/task directions in one raw likelihood.

    Fixed source eta nodes retain their original variance-reference proposal.
    A low-rank determinant identity evaluates the same source Gaussian density;
    no target field receives a post-hoc residual or covariance adjustment.
    """
    D = law['source_design']
    projection = np.linalg.lstsq(D, public_mean, rcond=None)[0]
    H = public_mean-D@projection
    blocks = np.cumsum(sizes)[:-1]
    eigen, vectors = np.linalg.eigh(law['task'])
    root = vectors[:, eigen > 1e-12] * np.sqrt(eigen[eigen > 1e-12])
    original = block_diag(*[d@root for d in np.split(D, blocks)])
    scale = float(np.var(labels))
    common = np.sqrt(scale)*H
    task = block_diag(*np.split(common, blocks))
    loading = np.column_stack([original, common, task])
    groups = np.repeat(np.arange(3), [original.shape[1], common.shape[1], task.shape[1]])
    base = D@law['common']@D.T + law['white']*np.eye(len(D))
    model = VarianceReferencePosterior(law['source_mean'], base, labels, sample_variances)
    records = []
    for state in eta:
        logdensity, factor, alpha = model.evaluate(state)
        records.append((logdensity, loading.T@cho_solve(factor, loading), loading.T@alpha))

    def objective(parameters):
        root_scale = np.sqrt(parameters[groups])
        logs, gradients = [], []
        for logdensity, gram, score in records:
            factor = cho_factor(np.eye(len(groups)) + root_scale[:, None]*gram*root_scale, lower=True)
            weighted = root_scale*score
            solved = cho_solve(factor, weighted)
            logs.append(logdensity + .5*(weighted@solved-2*np.log(np.diag(factor[0])).sum()))
            alpha = score-gram@(root_scale*solved)
            inverse = gram-(gram*root_scale)@cho_solve(factor, root_scale[:, None]*gram)
            gradient = .5*(alpha**2-np.diag(inverse))
            gradients.append(np.bincount(groups, weights=gradient, minlength=3))
        weights = np.exp(logs-logsumexp(logs))
        return float(-(logsumexp(logs)-np.log(len(logs)))), -weights@np.asarray(gradients)

    start = np.array([1., 0., 0.])
    fitted = minimize(objective, start, jac=True, method='L-BFGS-B', bounds=[(0., None)]*3,
        options={'maxiter': settings['risk_max_iterations'], 'ftol': 1e-9, 'gtol': 1e-5})
    task_scale, common_extra, task_extra = fitted.x
    augmented = np.column_stack([D, H]); extra_count = H.shape[1]
    common_covariance = block_diag(law['common'], scale*common_extra*np.eye(extra_count))
    task_covariance = block_diag(task_scale*law['task'], scale*task_extra*np.eye(extra_count))
    result = dict(law, mean=np.r_[law['mean'], np.zeros(extra_count)],
        common=common_covariance, task=task_covariance, source_design=augmented,
        public_projection=projection,
        source_covariance=augmented@common_covariance@augmented.T
            + block_diag(*[d@task_covariance@d.T for d in np.split(augmented, blocks)])
            + law['white']*np.eye(len(D)))
    diagnostic = {'original_task_covariance_multiplier': float(task_scale),
        'public_common_variance': float(scale*common_extra), 'public_task_variance': float(scale*task_extra),
        'optimizer_converged': bool(fitted.success), 'optimizer_message': str(fitted.message),
        'optimizer_iterations': int(fitted.nit), 'initial_negative_log_integral': objective(start)[0],
        'final_negative_log_integral': float(fitted.fun), 'source_only': True,
        'old_and_public_directions_jointly_conditioned': True}
    return result, diagnostic


def _condition_pair(law, risk, labels, sample_variances, eta, reference_logz):
    """Cache common coefficient posteriors; future bank projection is inexpensive."""
    model = VarianceReferencePosterior(law['source_mean'], law['source_covariance'], labels, sample_variances)
    cross = law['common']@law['source_design'].T
    eta_cross = risk['common']@risk['source_design'].T
    eta_gain = cho_solve(cho_factor(risk['source_covariance'], lower=True), eta_cross.T).T
    eta_means = risk['mean'] + (eta-risk['source_mean'])@eta_gain.T
    eta_covariance = risk['common']-eta_gain@eta_cross.T
    means, covariances, logs = [], [], []
    for state in eta:
        logdensity, factor, alpha = model.evaluate(state)
        means.append(law['mean']+cross@alpha)
        covariance = law['common']-cross@cho_solve(factor, cross.T)
        covariances.append((covariance+covariance.T)/2)
        logs.append(logdensity)
    logs = np.asarray(logs); weights = np.exp(logs-logsumexp(logs))
    halves = np.split(logs, 2)
    diagnostic = {'positive_components': len(logs), 'weight_ess': float(1/(weights@weights)),
        'maximum_component_mass': float(weights.max()),
        'split_log_normalizer_gap': float(abs(logsumexp(halves[0])-logsumexp(halves[1]))),
        'log_source_evidence': float(reference_logz+logsumexp(logs)-np.log(len(logs))),
        'precision_certified': False}
    return dict(law=law, risk=risk, mean=np.asarray(means), covariance=np.asarray(covariances),
        eta_mean=eta_means, eta_covariance=(eta_covariance+eta_covariance.T)/2,
        log_weights=logs-logsumexp(logs)), diagnostic


def fit_source_models(archive, source_problems, *, settings=None):
    """Fit the full archive or an equal-size, whole-policy source subset.

    Providers are keyed by archived family name. All response-dependent maps
    and laws are fitted anew from the supplied two-family, three-repeat rows.
    """
    settings = {**DEFAULTS, **(settings or {})}; started = time.process_time()
    tasks = legacy_source_labels(archive); sizes = [len(t['response']) for t in tasks]
    if (len(sizes) != 2 or sizes[0] != sizes[1]
            or any(len(values) != 3 for task in archive.tasks for values in task.Y_replicates)):
        raise ValueError('Source fitting requires two equal-size families with three raw repeats per policy')
    descriptors = np.vstack([legacy_exposure_descriptors(source_problems[t.name], t.X_integer, 50)
                              for t in archive.tasks])
    encoder = LegacyObservableCoordinate().fit(descriptors, tasks)
    mean, mean_center, mean_scale = _standardize(np.vstack([t['psi'] for t in tasks]))
    risk, risk_center, risk_scale = _standardize(encoder.transform(descriptors))
    coordinates = [np.split(mean, np.cumsum(sizes)[:-1]), np.split(risk, np.cumsum(sizes)[:-1])]
    maps, laws, diagnostics = {}, {}, []
    meta = LearnedMetaPrior()
    for kind, cls in [('spectral', TransferableSpectralBasis), ('additive', TransferableAdditiveGroupBank)]:
        for support in ('head_specific', 'joint'):
            joint = [np.column_stack(pair) for pair in zip(*coordinates)]
            for head in range(4):
                key = (kind, support, head if support == 'head_specific' else -1)
                inputs = joint if support == 'joint' else coordinates[int(head >= 2)]
                if key not in maps:
                    batches = [SourceDomainBatch(t['name'], x,
                        t['response'] if support == 'joint' else t['response'][:, head, None])
                        for t, x in zip(tasks, inputs)]
                    maps[key] = cls(**settings[kind]).fit(batches)
                basis = maps[key]
                prior, _, units = source_prior(basis, inputs, tasks, head, meta)
                features = basis.transform(np.vstack(inputs))
                coefficients, error = coefficient_fits(features, tasks, head, units, .001)
                parts = centered_fit_covariance_parts(coefficients, error)
                design = np.column_stack([np.ones(sum(sizes)), features])
                for treatment in ('observed_task', 'deconvolved_task'):
                    task_covariance = parts['observed' if treatment == 'observed_task' else 'deconvolved']
                    laws[kind, support, treatment, head] = _law(prior, design, task_covariance, sizes)
    rows, identities, evidence = [], [], []
    responses = np.vstack([t['response'] for t in tasks])
    sample_variances = np.vstack([t.replicate_variance for t in archive.tasks])
    for kind in ('spectral', 'additive'):
        for support in ('head_specific', 'joint'):
            for treatment in ('observed_task', 'deconvolved_task'):
                identity = (kind, support, treatment); pair = []; logz = 0.
                for head, eta_head in [(0, 3), (1, 2)]:
                    law, eta_law = laws[(*identity, head)], laws[(*identity, eta_head)]
                    states, reference_logz = _eta_rule(eta_law, sample_variances[:, head], settings, settings['seed']+head)
                    joint_fit = None
                    if kind == 'spectral' and support == 'head_specific' and head == 1:
                        law, joint_fit = _joint_risk_fit(law, mean, sizes, states,
                            responses[:, head], sample_variances[:, head], settings)
                    conditioned, diagnostic = _condition_pair(law, eta_law, responses[:, head],
                        sample_variances[:, head], states, reference_logz)
                    diagnostic.update(identity=list(identity), head=HEADS[head], joint_risk_fit=joint_fit)
                    diagnostics.append(diagnostic)
                    print('source', '/'.join(identity), HEADS[head], 'ESS', diagnostic['weight_ess'], flush=True)
                    diagnostic['low_importance_ess'] = diagnostic['weight_ess'] < settings['minimum_source_ess']
                    pair.append(conditioned); logz += diagnostic['log_source_evidence']
                identities.append('/'.join(identity)); rows.append(pair); evidence.append(logz)
    groups = [[dict(t, features=x) for t, x in zip(tasks, block)] for block in coordinates]
    gp = fit_source_controls(groups, settings['optimizer'])
    if any(not stage['optimizer_converged'] for head in gp['heads'] for stage in head.values()):
        raise SourceFitError({'failure': 'A same-source GP optimization did not converge',
            'gp': gp, 'source_identity_heads': diagnostics, 'cpu_seconds': time.process_time()-started})
    return dict(encoder=encoder, mean_center=mean_center, mean_scale=mean_scale,
        risk_center=risk_center, risk_scale=risk_scale, maps=maps, components=rows,
        identities=identities, log_weights=np.asarray(evidence)-logsumexp(evidence), gp=gp,
        settings=settings, diagnostics={'source_task_names': [t['name'] for t in tasks],
            'source_sites': sum(sizes), 'reused_source_calls': archive.simulator_calls, 'new_source_simulator_calls': 0,
            'source_identity_heads': diagnostics, 'cpu_seconds': time.process_time()-started,
            'source_identities': identities,
            'source_identity_weights': np.exp(np.asarray(evidence)-logsumexp(evidence)).tolist(),
            'source_quadrature_nodes': 2**settings['quadrature_power'],
            'source_likelihood': 'collapsed raw Normal/chi-square, fixed positive source quadrature',
            'source_hyperparameter_estimation': 'source coefficient moments; joint raw-risk amplitudes',
            'mean_coordinates': 'source-standardized old public ordered ten',
            'risk_coordinates': 'same-source observable ten', 'precision_certified': False})


def _query_design(bundle, identity, head, coordinates, law):
    kind, support, _ = identity.split('/')
    key = (kind, support, head if support == 'head_specific' else -1)
    inputs = np.column_stack(coordinates) if support == 'joint' else coordinates[int(head >= 2)]
    design = np.column_stack([np.ones(len(inputs)), bundle['maps'][key].transform(inputs)])
    if 'public_projection' in law:
        design = np.column_stack([design, coordinates[0]-design@law['public_projection']])
    return design


def _project_pair(bundle, identity, head, record, coordinates):
    law, risk = record['law'], record['risk']; count = len(record['mean'])
    D = _query_design(bundle, identity, head, coordinates, law)
    E = _query_design(bundle, identity, 3 if head == 0 else 2, coordinates, risk)
    n = len(D); means = np.stack([record['mean']@D.T, record['eta_mean']@E.T], axis=2)
    covariance = np.zeros((count, n, 2, n, 2))
    covariance[:, :, 0, :, 0] = np.einsum('np,kpq,mq->knm', D, record['covariance']+law['task'], D)
    covariance[:, :, 0, :, 0] += law['white']*np.eye(n)
    covariance[:, :, 1, :, 1] = E@(record['eta_covariance']+risk['task'])@E.T + risk['white']*np.eye(n)
    field = JointFunctionRiskPosterior.from_moments(means[0], covariance[0])
    field.means = means.reshape(count, 2*n).copy()
    field.covariances = covariance.reshape(count, 2*n, 2*n)
    field.log_weights = record['log_weights'].copy(); field.component_pairs = [(0, 0)]*count
    return field


def restore_model(bundle, arm, target_problem, bank, paid_history=()):
    """Reproject a zero-paid model on an expanded bank, then replay its own raws.

    Each history row is {'point': physical_integer_vector, 'raw': [f, raw_g]}.
    Replaying is computation only; the returned paid count is the historical
    number of physical episodes, not additional simulator expenditure.
    """
    mean = (legacy_public_coordinates(np.asarray(bank)/target_problem.L)-bundle['mean_center'])/bundle['mean_scale']
    risk = (bundle['encoder'].transform(legacy_exposure_descriptors(target_problem, bank, 50))
            -bundle['risk_center'])/bundle['risk_scale']
    if arm == 'source_common_task_rbf':
        model = controls_from_fit(bundle['gp'], mean, risk, target_problem.sigma_level)[arm]
    elif arm == 'source_structural_mixture':
        pairs = [tuple(_project_pair(bundle, name, h, record, (mean, risk)) for h, record in enumerate(pair))
                 for name, pair in zip(bundle['identities'], bundle['components'])]
        model = JointResponseMixture(bundle['identities'], pairs, np.exp(bundle['log_weights']))
    else:
        raise ValueError(arm)
    index = {tuple(point): i for i, point in enumerate(bank)}
    if len(index) != len(bank):
        raise ValueError('Dynamic bank must contain unique physical policies')
    for row in paid_history:
        raw = row['raw']; model.update_raw(index[tuple(row['point'])], [raw[0], raw[1]-target_problem.tau])
    assert model.paid_episodes == len(paid_history)
    return model


def _merge_moments(previous, log_mass, mean, covariance, noise):
    if previous is None:
        return log_mass, mean, covariance, noise
    old_mass, old_mean, old_covariance, old_noise = previous
    total = np.logaddexp(old_mass, log_mass)
    a, b = np.exp(old_mass-total), np.exp(log_mass-total)
    delta = old_mean-mean
    return (total, a*old_mean+b*mean,
            a*old_covariance+b*covariance+a*b*np.outer(delta, delta), a*old_noise+b*noise)


def _stream_head(record, D, E, history, head, component_batch_size):
    """Native updates on coefficients plus observed discrepancies, then project.

    Virtual coefficient slots are never observed. The observed function slots
    include their independent point discrepancies, so repeats and observed/new
    covariance have exactly the same law as the full candidate-bank filter.
    """
    law, risk = record['law'], record['risk']; n = len(D)
    observed = list(dict.fromkeys(row[0] for row in history)); local = {i: j for j, i in enumerate(observed)}
    no = len(observed); p = max(D.shape[1], E.shape[1]); compact_n = no+p
    DF = np.vstack([D[observed], np.eye(p, D.shape[1])])
    DE = np.vstack([E[observed], np.eye(p, E.shape[1])])
    LF, LE = np.zeros((n, compact_n)), np.zeros((n, compact_n))
    LF[:, no:no+D.shape[1]] = D; LE[:, no:no+E.shape[1]] = E
    unobserved = np.ones(n, dtype=bool)
    for i, j in local.items():
        LF[i] = 0.; LE[i] = 0.; LF[i, j] = LE[i, j] = 1.; unobserved[i] = False
    total = None; roundoff_events = []
    for start in range(0, len(record['mean']), component_batch_size):
        stop = start+component_batch_size
        cm, em = record['mean'][start:stop], record['eta_mean'][start:stop]
        count = len(cm)
        means = np.stack([cm@DF.T, em@DE.T], axis=2)
        covariance = np.zeros((count, compact_n, 2, compact_n, 2))
        covariance[:, :, 0, :, 0] = np.einsum('np,kpq,mq->knm', DF,
            record['covariance'][start:stop]+law['task'], DF)
        covariance[:, :, 1, :, 1] = DE@(record['eta_covariance']+risk['task'])@DE.T
        covariance[:, np.arange(no), 0, np.arange(no), 0] += law['white']
        covariance[:, np.arange(no), 1, np.arange(no), 1] += risk['white']
        field = JointFunctionRiskPosterior.from_moments(means[0], covariance[0])
        field.means = means.reshape(count, 2*compact_n).copy()
        field.covariances = covariance.reshape(count, 2*compact_n, 2*compact_n)
        field.component_pairs = [(0, 0)]*count
        prior_logs = record['log_weights'][start:stop]
        mass = float(logsumexp(prior_logs)); field.log_weights = prior_logs-mass
        for index, raw in history:
            mass += field.update_batch(local[index], raw[head], None, 1)
        roundoff_events.extend([{**event, 'component_start': start,
            'component_stop': min(stop, len(record['mean'])), 'head': HEADS[head]}
            for event in getattr(field, 'roundoff_events', [])])
        mean, cov = field.predict(full_cov=True)
        eta_mean = field.means[:, 1::2]@LE.T
        eta_var = np.einsum('ni,kij,nj->kn', LE, field.covariances[:, 1::2, 1::2], LE)
        eta_var += risk['white']*unobserved
        noise = np.exp(field.log_weights)@np.exp(eta_mean+eta_var/2)
        total = _merge_moments(total, mass, mean.reshape(-1), cov.reshape(2*compact_n, 2*compact_n), noise)
    mass, mean, covariance, noise = total
    transform = np.zeros((2*n, 2*compact_n)); transform[0::2, 0::2] = LF; transform[1::2, 1::2] = LE
    result_mean = transform@mean
    result_covariance = transform@covariance@transform.T
    result_covariance += np.diag(np.column_stack([law['white']*unobserved, risk['white']*unobserved]).reshape(-1))
    return mass, result_mean.reshape(n, 2), result_covariance.reshape(n, 2, n, 2), noise, roundoff_events


def restore_moment_state(bundle, arm, target_problem, bank, paid_history=(), *, component_batch_size=64):
    """Full-bank native posterior moments without storing all component fields."""
    if arm == 'source_common_task_rbf':
        from performance.mainline_moment_policy import moment_state
        model = restore_model(bundle, arm, target_problem, bank, paid_history)
        return {**moment_state(model), 'weights': model.weights, 'paid_episodes': model.paid_episodes}
    if arm != 'source_structural_mixture':
        raise ValueError(arm)
    index = {tuple(point): i for i, point in enumerate(bank)}
    if len(index) != len(bank):
        raise ValueError('Dynamic bank must contain unique physical policies')
    history = [(index[tuple(row['point'])], [row['raw'][0], row['raw'][1]-target_problem.tau]) for row in paid_history]
    mean = (legacy_public_coordinates(np.asarray(bank)/target_problem.L)-bundle['mean_center'])/bundle['mean_scale']
    risk = (bundle['encoder'].transform(legacy_exposure_descriptors(target_problem, bank, 50))
            -bundle['risk_center'])/bundle['risk_scale']
    coordinates = (mean, risk); n = len(bank); total = None; identity_masses = []; roundoff_events = []
    for identity, pair, prior_log_mass in zip(bundle['identities'], bundle['components'], bundle['log_weights']):
        results = []
        for head, record in enumerate(pair):
            D = _query_design(bundle, identity, head, coordinates, record['law'])
            E = _query_design(bundle, identity, 3 if head == 0 else 2, coordinates, record['risk'])
            results.append(_stream_head(record, D, E, history, head, component_batch_size))
        mass = float(prior_log_mass+sum(row[0] for row in results)); identity_masses.append(mass)
        roundoff_events.extend([{**event, 'identity': identity} for row in results for event in row[4]])
        mean = np.column_stack([row[1] for row in results]).reshape(-1)
        covariance = np.zeros((n, 4, n, 4))
        covariance[:, :2, :, :2], covariance[:, 2:, :, 2:] = results[0][2], results[1][2]
        noise = np.column_stack([row[3] for row in results])
        total = _merge_moments(total, mass, mean, covariance.reshape(4*n, 4*n), noise)
    mass, mean, covariance, noise = total
    return {'mean': mean.reshape(n, 4), 'cov': covariance.reshape(n, 4, n, 4), 'noise': noise,
            'weights': np.exp(np.asarray(identity_masses)-mass), 'paid_episodes': len(history),
            'roundoff_events': roundoff_events}


def main():
    from baselines.transfer_archive import FrozenTransferArchive
    from performance.benchmark_lodo_meta_prior import build_scalarized_problem
    from performance.diagnose_legacy_risk_transfer import no_outcomes
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', default='InventorySupplyChain')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    archive_path = Path('paper_artifacts/legacy_joint_bridge_v1/inputs')/args.family/'archive.json'
    started = time.process_time(); ledger_path = output/'source_fit_attempts.json'
    attempts = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    attempt = {'family': args.family, 'archive': str(archive_path), 'status': 'failed'}
    try:
        archive = FrozenTransferArchive.load(archive_path)
        providers = {t.name: no_outcomes(build_scalarized_problem(t.name, t.X.shape[1], 100, .04, .05, [.5, .5]))
                     for t in archive.tasks}
        bundle = fit_source_models(archive, providers)
        with (output/'source_models.pkl').open('wb') as stream:
            pickle.dump(bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)
        (output/'source_diagnostics.json').write_text(json.dumps(bundle['diagnostics'], indent=2, allow_nan=False)+'\n')
        (output/f'source_diagnostics_attempt{len(attempts)+1:02d}.json').write_text(
            json.dumps(bundle['diagnostics'], indent=2, allow_nan=False)+'\n')
        attempt['status'] = 'complete'
    except Exception as error:
        attempt['error'] = repr(error)
        if isinstance(error, SourceFitError):
            (output/'source_diagnostics.json').write_text(json.dumps(error.diagnostics, indent=2, allow_nan=False)+'\n')
            (output/f'source_diagnostics_attempt{len(attempts)+1:02d}.json').write_text(
                json.dumps(error.diagnostics, indent=2, allow_nan=False)+'\n')
        raise
    finally:
        attempt.update(cpu_seconds=time.process_time()-started, new_source_simulator_calls=0,
                       new_target_simulator_calls=0, new_verification_calls=0)
        attempts.append(attempt); ledger_path.write_text(json.dumps(attempts, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
