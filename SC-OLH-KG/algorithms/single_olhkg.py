"""Single-objective chance-constrained OLH-KG / SC-OLH-KG algorithm."""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import os
from pathlib import Path
import pickle
import time

import numpy as np
from scipy.stats import norm

from acquisition.olhkg import OLHKGAcquisition
from core.certification import conservative_chance_margin
from core.candidates import (
    axis_candidates,
    axis_landmark_candidates,
    boundary_solutions,
    latin_hypercube_candidates,
    posterior_sample_candidates,
    random_candidates,
    structured_candidates,
    unique_candidates,
)
from core.gpr import ParametricGPR
from core.metrics import summarize_stage_times
from encoders.policy_state_encoder import (
    ContrastivePolicyEncoder,
    GraphLaplacianEncoder,
    HybridSSLPolicyEncoder,
    KernelManifoldEncoder,
    MaskedTrajectoryEncoder,
    NextRiskEncoder,
    PCAManifoldEncoder,
    SelfSupervisedPolicyStateEncoder,
    SmallTransformerEncoder,
    StateCoupledFeatureMap,
    SyntheticPolicyStateEncoder,
)
from representation.llm_structural_prior import LLMStructuralPriorAdvisor
from representation.manifold import ManifoldRiskDecomposer
from variance.orthogonal_hvd import OrthogonalHVD


@dataclass
class SingleOLHKGConfig:
    N: int = 30
    n0: int = 8
    K1: int = 25
    K2: int = 0
    posterior_pool_size: int = 300
    posterior_keep: int = 15
    axis_candidate_count: int = -1
    structured_candidate_count: int = 0
    state_candidate_count: int = -1
    state_inverse_pool_size: int = 500
    state_inverse_neighbors: int = 2
    n_thr: int = 5
    lambda_i: float = 0.1
    prior_var: float = 10.0
    variance_mode: str = "factor"
    lambda_feas: float = 0.25
    lambda_var: float = 0.25
    lambda_mean: float = 0.10
    lambda_coupling: float = 0.05
    beta_g: float = 2.0
    certification_mode: str = "theory"
    coupling_safety_z: float = 0.5
    coupling_gate_temperature: float = 0.25
    recommendation_safety_z: float = 0.5
    recommendation_noise_floor_scale: float = 1.0
    recommendation_infeasible_penalty: float = 5.0
    recommendation_infeasible_strategy: str = "penalty"
    recommendation_calibration: bool = True
    recommendation_calibration_ridge: float = 1e-6
    recommendation_calibration_min_obs: int = 8
    certification_calibration: bool = False
    certification_calibration_min_obs: int = 8
    certification_calibration_ridge: float = 1e-6
    certification_calibration_noise_floor_scale: float = 0.5
    certification_calibration_beta: float = 2.0
    recommend_observed_only: bool = False
    recommendation_axis_oracle: bool = True
    use_problem_initial_samples: bool = True
    use_boundary_initial_samples: bool = True
    use_recommendation_refinement: bool = True
    recommendation_axis_candidate_count: int = -1
    use_state_coupling: bool = True
    use_state_basis: bool = True
    state_basis_mode: str = "raw+state"
    raw_basis_dim: int = -1
    raw_projection_seed: int = 314159
    numeric_backend: str = "numpy"
    numeric_backend_device: str = "auto"
    torch_dtype: str = "float64"
    torch_min_rows: int = 128
    use_manifold_hvd_features: bool = True
    encoder_kind: str = "synthetic"
    encoder_latent_dim: int = 8
    encoder_fit_pool_size: int = 512
    acquisition_mode: str = "exact_mc"
    exact_kg_mc_samples: int = 8
    exact_kg_jobs: int = 1
    exact_kg_use_score: bool = False
    exact_kg_blend: float = 0.0
    llm_prior_enabled: bool = False
    llm_prior_base_url: str = "https://ruoli.dev"
    llm_prior_model: str = "gpt-5.4-mini"
    llm_prior_api_key_env: str = "SCOLHKG_LLM_API_KEY"
    llm_prior_candidate_count: int = 0
    llm_prior_inverse_pool_size: int = 1024
    llm_prior_interval: int = 5
    llm_prior_min_obs: int = 8
    llm_prior_timeout_sec: float = 30.0
    llm_prior_gate_floor: float = 0.05
    llm_prior_max_observations: int = 24
    checkpoint_dir: str = ""
    checkpoint_resume: bool = False
    checkpoint_interval: int = 1
    checkpoint_keep_last: int = 3
    progress_logging: bool = False
    progress_label: str = ""
    progress_units_per_iteration: int = 100
    progress_exact_updates: int = 10
    eval_pool_size: int = 500
    evaluate_interval: int = 5
    seed: int = 123


class SingleOLHKGAlgorithm:
    """Minimal but complete single-objective OLH-KG implementation."""

    def __init__(self, problem, config: SingleOLHKGConfig | None = None):
        self.problem = problem
        self.config = config or SingleOLHKGConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.rec_rng = np.random.default_rng(int(self.config.seed) + 1_000_003)

        self.encoder = self._build_encoder()
        provider_available = False
        if hasattr(problem, "cumulative_risk_provider_status"):
            try:
                provider_available = (
                    problem.cumulative_risk_provider_status().get("status")
                    == "available"
                )
            except AttributeError:
                provider_available = False
        explicit_representation_basis = str(
            self.config.state_basis_mode or ""
        ).lower() in {
            "manifold",
            "raw+manifold",
            "raw_plus_manifold",
        }
        basis_map = None
        if self.config.use_state_basis and (
            provider_available or explicit_representation_basis
        ):
            basis_map = StateCoupledFeatureMap(
                problem,
                self.encoder,
                mode=self.config.state_basis_mode,
                raw_basis_dim=self.config.raw_basis_dim,
                raw_projection_seed=self.config.raw_projection_seed,
            )
        elif hasattr(problem, "gpr_basis_map"):
            basis_map = problem.gpr_basis_map()
        elif self.config.use_state_basis:
            basis_map = StateCoupledFeatureMap(
                problem,
                self.encoder,
                mode=self.config.state_basis_mode,
                raw_basis_dim=self.config.raw_basis_dim,
                raw_projection_seed=self.config.raw_projection_seed,
            )
        self._attach_representation_to_problem()
        self.gpr = [
            ParametricGPR(
                problem.d,
                self.config.lambda_i,
                self.config.prior_var,
                normalize_func=problem.normalize,
                basis_map=basis_map,
                numeric_backend=self.config.numeric_backend,
                numeric_backend_device=self.config.numeric_backend_device,
                torch_dtype=self.config.torch_dtype,
                torch_min_rows=self.config.torch_min_rows,
            )
            for _ in range(2)
        ]
        self.variance_model = OrthogonalHVD(
            mode=self.config.variance_mode,
            n_outputs=2,
            floor=1e-8,
        )
        self.acquisition = OLHKGAcquisition(
            lambda_feas=self.config.lambda_feas,
            lambda_var=self.config.lambda_var,
            lambda_mean=self.config.lambda_mean,
            lambda_coupling=self.config.lambda_coupling,
            beta_g=self.config.beta_g,
            certification_mode=self.config.certification_mode,
            coupling_safety_z=self.config.coupling_safety_z,
            coupling_gate_temperature=self.config.coupling_gate_temperature,
            encoder=self.encoder,
        )
        self.llm_prior = None
        if self.config.llm_prior_enabled:
            self.llm_prior = LLMStructuralPriorAdvisor(
                base_url=self.config.llm_prior_base_url,
                model=self.config.llm_prior_model,
                api_key_env=self.config.llm_prior_api_key_env,
                timeout_sec=self.config.llm_prior_timeout_sec,
                min_obs=self.config.llm_prior_min_obs,
                gate_floor=self.config.llm_prior_gate_floor,
                max_observations=self.config.llm_prior_max_observations,
            )
        self._last_llm_prior_info = {
            "status": "disabled" if self.llm_prior is None else "not_called",
            "gate": 0.0,
            "n_regions": 0,
        }

        self.observations: dict[tuple[int, ...], list[np.ndarray]] = {}
        self.history: list[tuple[tuple[int, ...], np.ndarray]] = []
        self.iteration_log: list[dict] = []
        self.pre_sampling_log: dict | None = None
        self.final_log: dict | None = None

    def _build_encoder(self):
        needs_encoder = (
            self.config.use_state_coupling
            or self.config.use_state_basis
            or self.config.lambda_coupling > 0
        )
        if not needs_encoder:
            return None
        kind = str(self.config.encoder_kind or "synthetic").lower()
        trajectory_records = getattr(self.problem, "_scolhkg_trajectory_records", None)
        if kind in ("self_supervised", "masked", "contrastive"):
            return SelfSupervisedPolicyStateEncoder(
                self.problem,
                latent_dim=self.config.encoder_latent_dim,
                mode="masked",
                fit_pool_size=self.config.encoder_fit_pool_size,
                rng=self.rng,
            )
        if kind in ("transformer", "attention"):
            return SelfSupervisedPolicyStateEncoder(
                self.problem,
                latent_dim=self.config.encoder_latent_dim,
                mode="transformer",
                fit_pool_size=self.config.encoder_fit_pool_size,
                rng=self.rng,
            )
        if kind == "pca_manifold":
            return PCAManifoldEncoder(
                self.problem,
                latent_dim=self.config.encoder_latent_dim,
                fit_pool_size=self.config.encoder_fit_pool_size,
                rng=self.rng,
            )
        if kind == "kernel_manifold":
            return KernelManifoldEncoder(
                self.problem,
                latent_dim=self.config.encoder_latent_dim,
                fit_pool_size=self.config.encoder_fit_pool_size,
                rng=self.rng,
            )
        if kind in ("graph_laplacian", "diffusion_manifold", "graph_manifold"):
            return GraphLaplacianEncoder(
                self.problem,
                latent_dim=self.config.encoder_latent_dim,
                fit_pool_size=self.config.encoder_fit_pool_size,
                rng=self.rng,
            )
        if kind in ("ssl_masked", "masked_trajectory"):
            return MaskedTrajectoryEncoder(
                self.problem,
                latent_dim=self.config.encoder_latent_dim,
                fit_pool_size=self.config.encoder_fit_pool_size,
                rng=self.rng,
                records_or_policy_pool=trajectory_records,
            )
        if kind in ("ssl_contrastive", "contrastive_policy"):
            return ContrastivePolicyEncoder(
                self.problem,
                latent_dim=self.config.encoder_latent_dim,
                fit_pool_size=self.config.encoder_fit_pool_size,
                rng=self.rng,
                records_or_policy_pool=trajectory_records,
            )
        if kind in ("ssl_next_risk", "next_risk"):
            return NextRiskEncoder(
                self.problem,
                latent_dim=self.config.encoder_latent_dim,
                fit_pool_size=self.config.encoder_fit_pool_size,
                rng=self.rng,
                records_or_policy_pool=trajectory_records,
            )
        if kind in ("ssl_transformer", "small_transformer"):
            return SmallTransformerEncoder(
                self.problem,
                latent_dim=self.config.encoder_latent_dim,
                fit_pool_size=self.config.encoder_fit_pool_size,
                rng=self.rng,
                records_or_policy_pool=trajectory_records,
            )
        if kind in ("ssl_hybrid", "hybrid_ssl", "contextual_manifold"):
            return HybridSSLPolicyEncoder(
                self.problem,
                latent_dim=self.config.encoder_latent_dim,
                fit_pool_size=self.config.encoder_fit_pool_size,
                rng=self.rng,
                records_or_policy_pool=trajectory_records,
            )
        return SyntheticPolicyStateEncoder(self.problem)

    def _attach_representation_to_problem(self):
        if self.encoder is None:
            return
        kind = str(self.config.encoder_kind or "synthetic").lower()
        setattr(self.problem, "_scolhkg_representation_encoder", self.encoder)
        if (
            self.config.use_manifold_hvd_features
            and kind in {
                "pca_manifold",
                "kernel_manifold",
                "graph_laplacian",
                "diffusion_manifold",
                "graph_manifold",
                "ssl_masked",
                "ssl_contrastive",
                "ssl_next_risk",
                "ssl_transformer",
                "ssl_hybrid",
                "hybrid_ssl",
                "contextual_manifold",
                "small_transformer",
            }
        ):
            setattr(self.problem, "_scolhkg_use_manifold_hvd", True)
            setattr(
                self.problem,
                "_scolhkg_manifold_decomposer",
                ManifoldRiskDecomposer(self.encoder),
            )

    def _initial_samples(self):
        samples = []
        if self.config.use_problem_initial_samples and hasattr(
            self.problem, "initial_samples"
        ):
            samples.extend(self.problem.initial_samples(
                n=self.config.n0,
                rng=self.rng,
            ))
            samples = unique_candidates(samples)
        if len(samples) < self.config.n0 and self.config.use_boundary_initial_samples:
            for x in boundary_solutions(self.problem):
                if len(samples) >= self.config.n0:
                    break
                samples.append(tuple(x))
        while len(set(samples)) < self.config.n0:
            samples.append(self.problem.sample_random(self.rng))
            samples = unique_candidates(samples)
        return unique_candidates(samples)[: self.config.n0]

    def _simulate_and_store(self, x):
        y = self.problem.simulate(x, self.rng)
        x_tuple = tuple(int(v) for v in x)
        self.observations.setdefault(x_tuple, []).append(y)
        self.history.append((x_tuple, y))
        return y

    def _fit_initial_belief(self, samples):
        for x in samples:
            self._simulate_and_store(x)

        Phi = self.gpr[0].basis_matrix(samples)
        for i in range(2):
            y_i = np.array([self.observations[x][0][i] for x in samples], dtype=float)
            try:
                beta = np.linalg.lstsq(Phi, y_i, rcond=None)[0]
            except np.linalg.LinAlgError:
                beta = np.zeros(Phi.shape[1], dtype=float)
            resid = y_i - Phi @ beta
            lambda_data = max(float(np.var(resid)), 1e-6)
            prior_var = max(float(np.var(beta)), 1e-6)
            self.gpr[i].set_parametric_prior(beta, lambda_data, prior_var)

        for x in samples:
            for model in self.gpr:
                model.dimension_augment(x)

        self.variance_model.initialize(
            samples, self.observations, self.gpr, self.problem)

    def _checkpoint_root(self):
        root = str(self.config.checkpoint_dir or "").strip()
        if not root:
            return None
        return Path(root)

    def _gpr_checkpoint_state(self, model):
        return {
            "d": int(model.d),
            "p": int(model.p),
            "lambda_i": float(model.lambda_i),
            "a": np.asarray(model.a, dtype=float),
            "C": np.asarray(model.C, dtype=float),
            "sampled_set": [tuple(int(v) for v in x) for x in model.sampled_set],
            "sol_to_idx": {
                tuple(int(v) for v in key): int(value)
                for key, value in model.sol_to_idx.items()
            },
            "state_version": int(getattr(model, "_state_version", 0)),
        }

    def _restore_gpr_checkpoint_state(self, model, state):
        if int(state.get("d", model.d)) != int(model.d):
            raise ValueError("checkpoint GPR dimension does not match current problem")
        if int(state.get("p", model.p)) != int(model.p):
            raise ValueError("checkpoint basis dimension does not match current config")
        model.lambda_i = float(state["lambda_i"])
        model.a = np.asarray(state["a"], dtype=float).copy()
        model.C = np.asarray(state["C"], dtype=float).copy()
        model.sampled_set = [
            tuple(int(v) for v in row)
            for row in state.get("sampled_set", [])
        ]
        model.sol_to_idx = {
            tuple(int(v) for v in key): int(value)
            for key, value in state.get("sol_to_idx", {}).items()
        }
        model._state_version = int(state.get("state_version", 0)) + 1
        model._torch_cache = {}

    def _clone_gpr_for_exact_kg(self, model):
        """Clone mutable GPR state without deep-copying simulator/provider handles."""
        clone = object.__new__(model.__class__)
        clone.__dict__ = model.__dict__.copy()
        clone.a = np.asarray(model.a, dtype=float).copy()
        clone.C = np.asarray(model.C, dtype=float).copy()
        clone.sampled_set = [
            tuple(int(v) for v in row)
            for row in getattr(model, "sampled_set", [])
        ]
        clone.sol_to_idx = {
            tuple(int(v) for v in key): int(value)
            for key, value in getattr(model, "sol_to_idx", {}).items()
        }
        clone._torch_cache = {}
        return clone

    def _clone_variance_model_for_exact_kg(self):
        state = copy.deepcopy(self.variance_model.__getstate__())
        clone = object.__new__(self.variance_model.__class__)
        clone.__setstate__(state)
        clone._last_problem = self.problem
        return clone

    def _runtime_checkpoint_payload(self, next_stage_n, reason):
        return {
            "schema_version": 1,
            "reason": str(reason),
            "next_stage_n": int(next_stage_n),
            "saved_at": float(time.time()),
            "config": asdict(self.config),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "rec_rng_state": copy.deepcopy(self.rec_rng.bit_generator.state),
            "observations": copy.deepcopy(self.observations),
            "history": copy.deepcopy(self.history),
            "iteration_log": copy.deepcopy(self.iteration_log),
            "pre_sampling_log": copy.deepcopy(self.pre_sampling_log),
            "final_log": copy.deepcopy(self.final_log),
            "gpr": [self._gpr_checkpoint_state(model) for model in self.gpr],
            "variance_model": copy.deepcopy(self.variance_model.__getstate__()),
        }

    def _write_pickle_atomic(self, path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def _prune_checkpoints(self, root):
        keep = int(self.config.checkpoint_keep_last)
        if keep <= 0:
            return
        stage_files = sorted(root.glob("checkpoint_stage_*.pkl"))
        for path in stage_files[:-keep]:
            try:
                path.unlink()
            except OSError:
                pass

    def _save_checkpoint(self, next_stage_n, reason="iteration", force=False):
        root = self._checkpoint_root()
        if root is None:
            return None
        interval = max(1, int(self.config.checkpoint_interval))
        should_stage = (
            force
            or reason in {"initial", "final"}
            or int(next_stage_n) >= int(self.config.N)
            or int(next_stage_n) % interval == 0
        )
        payload = self._runtime_checkpoint_payload(next_stage_n, reason)
        latest_path = root / "checkpoint_latest.pkl"
        self._write_pickle_atomic(latest_path, payload)
        if should_stage:
            stage_path = root / f"checkpoint_stage_{int(next_stage_n):05d}.pkl"
            self._write_pickle_atomic(stage_path, payload)
            self._prune_checkpoints(root)
        return latest_path

    def _reattach_runtime_handles_after_checkpoint(self):
        if self.encoder is not None:
            if hasattr(self.encoder, "problem"):
                self.encoder.problem = self.problem
            if hasattr(self.encoder, "rng"):
                self.encoder.rng = self.rng
            raw_encoder = getattr(self.encoder, "raw_encoder", None)
            if raw_encoder is not None:
                if hasattr(raw_encoder, "problem"):
                    raw_encoder.problem = self.problem
                if hasattr(raw_encoder, "rng"):
                    raw_encoder.rng = self.rng
        self._attach_representation_to_problem()
        self.variance_model._last_problem = self.problem
        self.acquisition.encoder = self.encoder

    def _load_checkpoint_payload(self, payload):
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported SC-OLH-KG checkpoint schema")
        self.rng.bit_generator.state = payload["rng_state"]
        self.rec_rng.bit_generator.state = payload["rec_rng_state"]
        self.observations = copy.deepcopy(payload.get("observations", {}))
        self.history = copy.deepcopy(payload.get("history", []))
        self.iteration_log = copy.deepcopy(payload.get("iteration_log", []))
        self.pre_sampling_log = copy.deepcopy(payload.get("pre_sampling_log"))
        self.final_log = copy.deepcopy(payload.get("final_log"))
        for model, state in zip(self.gpr, payload.get("gpr", [])):
            self._restore_gpr_checkpoint_state(model, state)
        variance_state = copy.deepcopy(payload.get("variance_model"))
        if variance_state is None:
            raise ValueError("checkpoint is missing variance model state")
        self.variance_model.__setstate__(variance_state)
        self._reattach_runtime_handles_after_checkpoint()
        return int(payload.get("next_stage_n", self.config.n0))

    def _try_resume_from_checkpoint(self, verbose=False):
        root = self._checkpoint_root()
        if root is None or not self.config.checkpoint_resume:
            return None
        path = root if root.is_file() else root / "checkpoint_latest.pkl"
        if not path.exists():
            return None
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        next_stage_n = self._load_checkpoint_payload(payload)
        if verbose:
            print(
                f"resumed checkpoint={path} "
                f"next_stage_n={next_stage_n} "
                f"history={len(self.history)}"
            )
        return next_stage_n

    def _initialize_or_resume(self, verbose=False):
        resumed = self._try_resume_from_checkpoint(verbose=verbose)
        if resumed is not None:
            return resumed
        samples = self._initial_samples()
        t0 = time.time()
        self._fit_initial_belief(samples)
        self.pre_sampling_log = {
            "n0": self.config.n0,
            "samples": [list(map(int, x)) for x in samples],
            "time_sec": float(time.time() - t0),
            "variance": self.variance_model.diagnostics(),
        }
        self._save_checkpoint(self.config.n0, reason="initial", force=True)
        return int(self.config.n0)

    def _progress_enabled(self):
        return bool(getattr(self.config, "progress_logging", False))

    def _progress_label(self):
        label = str(getattr(self.config, "progress_label", "") or "").strip()
        if label:
            return label
        return f"seed={int(self.config.seed)}"

    def _progress_units_per_iteration(self):
        return max(1, int(getattr(self.config, "progress_units_per_iteration", 100)))

    def _progress_emit(
        self,
        *,
        n,
        frac,
        kind,
        started_at,
        run_started_at,
        extra="",
    ):
        if not self._progress_enabled():
            return
        units = self._progress_units_per_iteration()
        total_units = max(1, int(self.config.N) * units)
        current_units = int(round((float(n) + float(np.clip(frac, 0.0, 1.0))) * units))
        current_units = max(0, min(total_units, current_units))
        elapsed = max(0.0, time.time() - float(run_started_at))
        now = time.perf_counter()
        elapsed = max(0.0, now - float(run_started_at))
        step_elapsed = max(0.0, now - float(started_at))
        start_units = max(0, int(self.config.n0) * units)
        done_units = max(1, current_units - start_units)
        remaining_units = max(0, total_units - current_units)
        eta_sec = (elapsed / float(done_units)) * float(remaining_units)
        msg = (
            f"Step {current_units}/{total_units} [kg-inner] "
            f"kind={kind} label={self._progress_label()} "
            f"stage={int(n)}/{int(self.config.N)} "
            f"elapsed={elapsed:.1f}s step_elapsed={step_elapsed:.1f}s "
            f"ETA {eta_sec:.1f}s"
        )
        if extra:
            msg += f" {extra}"
        print(msg, flush=True)

    def _variance_lookup(self, i, x):
        return self.variance_model.predict_variance(i, x, self.problem)

    def _constraint_epistemic_lookup(self, x):
        return self.gpr[1].posterior_var(x)

    def _certification_result(self, mu_con, candidates, v_con=None):
        if v_con is None:
            v_con = self.variance_model.predict_certification_variance_many(
                1, candidates, self.problem)
        epistemic = self.gpr[1].posterior_var_many(candidates)
        return conservative_chance_margin(
            mu_con,
            epistemic,
            v_con,
            tau=self.problem.tau,
            alpha=self.problem.alpha,
            beta_g=self.config.beta_g,
            mode=self.config.certification_mode,
        )

    def _axis_candidate_count(self):
        n_axis = int(self.config.axis_candidate_count)
        if n_axis >= 0:
            return n_axis
        return max(5, min(21, max(1, self.config.K1 // 2)))

    def _recommendation_axis_candidate_count(self):
        n_axis = int(self.config.recommendation_axis_candidate_count)
        if n_axis >= 0:
            return n_axis
        return self._axis_candidate_count()

    def _recommendation_refinement_candidates(self):
        if not self.config.use_recommendation_refinement:
            return []
        if not hasattr(self.problem, "recommendation_refinement_candidates"):
            return []
        return unique_candidates(self.problem.recommendation_refinement_candidates())

    def _recommendation_random_pool_size(self):
        if hasattr(self.problem, "recommendation_random_pool_size"):
            try:
                return max(0, int(self.problem.recommendation_random_pool_size()))
            except AttributeError:
                pass
        return max(0, int(self.config.eval_pool_size))

    def _generate_candidates(self, iteration):
        candidates = []
        sources = {}

        def add(rows, source):
            for row in rows:
                x_tuple = tuple(int(v) for v in row)
                if x_tuple not in sources:
                    candidates.append(x_tuple)
                    sources[x_tuple] = source

        add(latin_hypercube_candidates(
            self.problem, self.config.K1, self.rng), "lhs")
        n_axis = self._axis_candidate_count()
        add(axis_landmark_candidates(self.problem, n_axis, self.rng), "axis_landmark")
        add(axis_candidates(self.problem, n_axis, self.rng), "axis")
        if hasattr(self.problem, "structured_candidates"):
            n_structured = int(self.config.structured_candidate_count)
            if n_structured < 0:
                n_structured = max(5, self.config.K1 // 2)
            add(structured_candidates(
                self.problem, n_structured, self.rng), "structured")
        if self.config.use_state_coupling and self.encoder is not None:
            n_state = int(self.config.state_candidate_count)
            if n_state < 0:
                n_state = max(5, self.config.K1)
            add(self.encoder.state_space_candidates(
                n_anchors=n_state,
                inverse_pool_size=self.config.state_inverse_pool_size,
                inverse_neighbors=self.config.state_inverse_neighbors,
                rng=self.rng,
                observed=[x for x, _ in self.history],
            ), "state")
        self._last_llm_prior_info = {
            "status": "disabled" if self.llm_prior is None else "skipped",
            "gate": 0.0,
            "n_regions": 0,
        }
        if self.llm_prior is not None:
            interval = max(1, int(self.config.llm_prior_interval))
            n_llm = int(self.config.llm_prior_candidate_count)
            if n_llm <= 0:
                n_llm = max(5, self.config.K1 // 2)
            if iteration % interval == 0:
                regions, info = self.llm_prior.propose(
                    self.problem,
                    self.observations,
                    iteration=iteration,
                    budget_remaining=max(0, int(self.config.N) - len(self.history)),
                )
                self._last_llm_prior_info = info
                add(self.llm_prior.inverse_candidates(
                    self.problem,
                    regions,
                    n=n_llm,
                    rng=self.rng,
                    pool_size=self.config.llm_prior_inverse_pool_size,
                    gate=info.get("gate", 0.0),
                ), "llm_prior")
        add(random_candidates(
            self.problem, max(5, self.config.K1 // 5), self.rng), "random")
        use_constraint = iteration > self.config.n_thr
        add(posterior_sample_candidates(
            self.problem,
            self.gpr,
            n_batches=self.config.K2,
            pool_size=self.config.posterior_pool_size,
            keep_per_batch=self.config.posterior_keep,
            rng=self.rng,
            use_constraint=use_constraint,
            variance_lookup=self._variance_lookup,
            epistemic_lookup=self._constraint_epistemic_lookup,
            tau=self.problem.tau,
            alpha_z=norm.ppf(1 - self.problem.alpha),
            beta_g=self.config.beta_g,
            certification_mode=self.config.certification_mode,
        ), "posterior")
        if not candidates:
            add([self.problem.sample_random(self.rng)], "random")
        return candidates, sources

    def _recommendation_pool(self):
        pool = set(x for x, _ in self.history)
        if self.config.recommend_observed_only:
            return list(pool)
        for x in random_candidates(
            self.problem,
            self._recommendation_random_pool_size(),
            self.rec_rng,
        ):
            pool.add(tuple(x))
        if self.config.recommendation_axis_oracle and hasattr(
            self.problem, "all_axis_solutions"
        ):
            for x in self.problem.all_axis_solutions():
                pool.add(tuple(x))
        elif not self.config.recommendation_axis_oracle:
            n_axis = self._recommendation_axis_candidate_count()
            for x in axis_landmark_candidates(self.problem, n_axis, self.rec_rng):
                pool.add(tuple(x))
            for x in axis_candidates(self.problem, n_axis, self.rec_rng):
                pool.add(tuple(x))
        for x in self._recommendation_refinement_candidates():
            pool.add(tuple(x))
        return list(pool)

    def _observed_nominal_incumbent(self):
        z = norm.ppf(1 - self.problem.alpha)
        sigma_floor = float(getattr(self.problem, "sigma_level", 0.0))
        margin_limit = -0.5 * sigma_floor
        best = None
        for x, ys in self.observations.items():
            y_bar = np.mean(np.asarray(ys, dtype=float), axis=0)
            margin = float(y_bar[1] + z * sigma_floor - self.problem.tau)
            if margin > margin_limit:
                continue
            item = (float(y_bar[0]), margin, tuple(int(v) for v in x))
            if best is None or item < best:
                best = item
        if best is None:
            return None
        obj, margin, x = best
        return {
            "x": x,
            "empirical_objective": float(obj),
            "empirical_chance_margin": float(margin),
        }

    def _constraint_calibration_fit(self):
        if not self.config.certification_calibration:
            return None
        if not hasattr(self.problem, "surrogate_basis_map"):
            return None
        basis = self.problem.surrogate_basis_map()
        if basis is None:
            return None
        if len(self.observations) < int(self.config.certification_calibration_min_obs):
            return None

        train_x = []
        train_con = []
        for x, ys in self.observations.items():
            train_x.append(tuple(int(v) for v in x))
            y_bar = np.mean(np.asarray(ys, dtype=float), axis=0)
            train_con.append(float(y_bar[1]))
        Phi = np.vstack([
            np.concatenate([[1.0], np.asarray(basis.features(x), dtype=float)])
            for x in train_x
        ])
        y_con = np.asarray(train_con, dtype=float)
        ridge = max(float(self.config.certification_calibration_ridge), 0.0)
        penalty = ridge * np.eye(Phi.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        lhs = Phi.T @ Phi + penalty
        rhs = Phi.T @ y_con
        try:
            beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        try:
            inv_lhs = np.linalg.pinv(lhs)
        except np.linalg.LinAlgError:
            inv_lhs = np.eye(lhs.shape[0], dtype=float) / max(ridge, 1e-8)
        resid = y_con - Phi @ beta
        resid_sigma = (
            float(np.sqrt(np.mean(resid ** 2)))
            if len(resid)
            else 0.0
        )
        nominal_floor = (
            float(self.config.certification_calibration_noise_floor_scale)
            * float(getattr(self.problem, "sigma_level", 0.0))
        )
        sigma_cal = max(resid_sigma, nominal_floor, 1e-8)
        return {
            "basis": basis,
            "beta": beta,
            "inv_lhs": inv_lhs,
            "sigma": float(sigma_cal),
            "resid_sigma": float(resid_sigma),
            "n_train": int(len(train_x)),
            "feature_dim": int(Phi.shape[1]),
        }

    def _calibrated_certification_result(self, pool, v_con=None):
        fit = self._constraint_calibration_fit()
        if fit is None or not pool:
            return None
        Phi_cand = np.vstack([
            np.concatenate([
                [1.0],
                np.asarray(fit["basis"].features(x), dtype=float),
            ])
            for x in pool
        ])
        mu = Phi_cand @ fit["beta"]
        leverage = np.sum((Phi_cand @ fit["inv_lhs"]) * Phi_cand, axis=1)
        leverage = np.maximum(leverage, 0.0)
        epistemic = (float(fit["sigma"]) ** 2) * leverage
        aleatoric = np.full(
            len(pool),
            max(float(fit["sigma"]) ** 2, 1e-12),
            dtype=float,
        )
        # Keep HVD visible in diagnostics without letting sparse high-dimensional
        # GPR residuals dominate a low-dimensional calibrated certificate.
        if v_con is not None:
            hvd = np.maximum(np.asarray(v_con, dtype=float), 1e-12)
            aleatoric = np.maximum(np.minimum(hvd, aleatoric), 0.25 * aleatoric)
        beta = max(float(self.config.certification_calibration_beta), 0.0)
        cert = conservative_chance_margin(
            mu,
            epistemic,
            aleatoric,
            tau=self.problem.tau,
            alpha=self.problem.alpha,
            beta_g=beta,
            mode="theory",
        )
        return {
            "margin": cert.margin,
            "mu": cert.mu,
            "epistemic_var": cert.epistemic_var,
            "aleatoric_var": cert.aleatoric_var,
            "beta_g": float(cert.beta_g),
            "z_alpha": float(cert.z_alpha),
            "sigma": float(fit["sigma"]),
            "resid_sigma": float(fit["resid_sigma"]),
            "n_train": int(fit["n_train"]),
            "feature_dim": int(fit["feature_dim"]),
            "n_feasible": int(np.sum(cert.margin <= 0.0)),
        }

    def _calibrated_recommendation_index(self, pool, robust_margins):
        if not self.config.recommendation_calibration:
            return None, {}
        if not hasattr(self.problem, "surrogate_basis_map"):
            return None, {}
        basis = self.problem.surrogate_basis_map()
        if basis is None:
            return None, {}
        refinement = self._recommendation_refinement_candidates()
        if not refinement:
            return None, {}
        if len(self.observations) < int(self.config.recommendation_calibration_min_obs):
            return None, {}

        train_x = []
        train_obj = []
        train_con = []
        for x, ys in self.observations.items():
            train_x.append(tuple(int(v) for v in x))
            y_bar = np.mean(np.asarray(ys, dtype=float), axis=0)
            train_obj.append(float(y_bar[0]))
            train_con.append(float(y_bar[1]))
        Phi = np.vstack([
            np.concatenate([[1.0], np.asarray(basis.features(x), dtype=float)])
            for x in train_x
        ])
        y_obj = np.asarray(train_obj, dtype=float)
        y_con = np.asarray(train_con, dtype=float)
        ridge = max(float(self.config.recommendation_calibration_ridge), 0.0)
        penalty = ridge * np.eye(Phi.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        try:
            beta_obj = np.linalg.solve(Phi.T @ Phi + penalty, Phi.T @ y_obj)
            beta_con = np.linalg.solve(Phi.T @ Phi + penalty, Phi.T @ y_con)
        except np.linalg.LinAlgError:
            beta_obj = np.linalg.lstsq(
                Phi.T @ Phi + penalty,
                Phi.T @ y_obj,
                rcond=None,
            )[0]
            beta_con = np.linalg.lstsq(
                Phi.T @ Phi + penalty,
                Phi.T @ y_con,
                rcond=None,
            )[0]

        pool_index = {tuple(int(v) for v in x): i for i, x in enumerate(pool)}
        candidate_indices = [
            pool_index[x]
            for x in refinement
            if x in pool_index
        ]
        if not candidate_indices:
            return None, {}
        Phi_cand = np.vstack([
            np.concatenate([[1.0], np.asarray(basis.features(pool[i]), dtype=float)])
            for i in candidate_indices
        ])
        pred_obj = Phi_cand @ beta_obj

        certified = np.array([
            robust_margins[i] <= 0.0
            for i in candidate_indices
        ], dtype=bool)
        if np.any(certified):
            local_cert = np.where(certified)[0]
            chosen_pos = int(local_cert[int(np.argmin(pred_obj[local_cert]))])
            return int(candidate_indices[chosen_pos]), {
                "calibrated_recommendation_reason": "certified_refinement_objective",
                "calibrated_objective": float(pred_obj[chosen_pos]),
                "calibrated_constraint_margin": None,
                "calibrated_constraint_feasible": None,
                "calibrated_constraint_sigma": None,
                "n_calibration_refinement": int(len(candidate_indices)),
                "n_calibration_certified": int(np.sum(certified)),
            }

        # If the theory bound is too conservative everywhere, fit a local
        # low-dimensional constraint surrogate on observed data and use it only
        # as a fallback.  The returned posterior_feasible flag remains false;
        # this path is an empirical recommendation rescue, not a certification
        # claim.
        pred_con = Phi_cand @ beta_con
        resid_con = y_con - Phi @ beta_con
        resid_sigma = float(np.sqrt(np.mean(resid_con ** 2))) if len(resid_con) else 0.0
        nominal_floor = (
            float(self.config.recommendation_noise_floor_scale)
            * 0.35
            * float(getattr(self.problem, "sigma_level", 0.0))
        )
        sigma_cal = max(resid_sigma, nominal_floor, 1e-8)
        z_alpha = float(norm.ppf(1 - self.problem.alpha))
        calibrated_margin = (
            pred_con
            + z_alpha * sigma_cal
            - float(self.problem.tau)
        )
        feasible = calibrated_margin <= 0.0
        if not np.any(feasible):
            return None, {
                "calibrated_recommendation_reason": "no_calibrated_feasible",
                "calibrated_objective": None,
                "calibrated_constraint_margin": float(np.min(calibrated_margin)),
                "calibrated_constraint_feasible": False,
                "calibrated_constraint_sigma": float(sigma_cal),
                "n_calibration_refinement": int(len(candidate_indices)),
                "n_calibration_certified": 0,
                "n_calibration_feasible": 0,
            }
        local_feas = np.where(feasible)[0]
        chosen_pos = int(local_feas[int(np.argmin(pred_obj[local_feas]))])
        return int(candidate_indices[chosen_pos]), {
            "calibrated_recommendation_reason": "calibrated_constraint_fallback",
            "calibrated_objective": float(pred_obj[chosen_pos]),
            "calibrated_constraint_margin": float(calibrated_margin[chosen_pos]),
            "calibrated_constraint_feasible": True,
            "calibrated_constraint_sigma": float(sigma_cal),
            "n_calibration_refinement": int(len(candidate_indices)),
            "n_calibration_certified": 0,
            "n_calibration_feasible": int(np.sum(feasible)),
        }

    def _solve_posterior_recommendation(self):
        pool = self._recommendation_pool()
        mu_obj = self.gpr[0].posterior_mean_many(pool)
        mu_con = self.gpr[1].posterior_mean_many(pool)
        v_con = self.variance_model.predict_certification_variance_many(
            1, pool, self.problem)
        cert = self._certification_result(mu_con, pool, v_con)
        margins = cert.margin
        sig_con = np.sqrt(np.maximum(cert.aleatoric_var, 1e-12))
        if cert.mode == "theory":
            safety_buffer = np.zeros_like(margins)
        else:
            nominal_floor = (
                self.config.recommendation_noise_floor_scale
                * float(getattr(self.problem, "sigma_level", 0.0))
            )
            safety_buffer = np.maximum(
                self.config.recommendation_safety_z * sig_con,
                nominal_floor,
            )
        theory_margins = np.asarray(margins + safety_buffer, dtype=float)
        robust_margins = theory_margins.copy()
        effective_mu_con = np.asarray(mu_con, dtype=float)
        effective_epistemic = np.asarray(cert.epistemic_var, dtype=float)
        effective_aleatoric = np.asarray(cert.aleatoric_var, dtype=float)
        certification_sources = np.full(len(pool), "theory", dtype=object)
        calibrated_cert = self._calibrated_certification_result(pool, v_con)
        if calibrated_cert is not None:
            calibrated_margins = np.asarray(calibrated_cert["margin"], dtype=float)
            use_calibrated = calibrated_margins < robust_margins
            robust_margins = np.where(use_calibrated, calibrated_margins, robust_margins)
            effective_mu_con = np.where(
                use_calibrated,
                calibrated_cert["mu"],
                effective_mu_con,
            )
            effective_epistemic = np.where(
                use_calibrated,
                calibrated_cert["epistemic_var"],
                effective_epistemic,
            )
            effective_aleatoric = np.where(
                use_calibrated,
                calibrated_cert["aleatoric_var"],
                effective_aleatoric,
            )
            certification_sources = np.where(
                use_calibrated,
                "calibrated",
                certification_sources,
            )
        else:
            calibrated_margins = None
        feasible = robust_margins <= 0.0
        if np.any(feasible):
            local = int(np.argmin(np.where(feasible, mu_obj, np.inf)))
        elif str(self.config.recommendation_infeasible_strategy).lower() in (
            "min_margin",
            "lexicographic",
        ):
            min_margin = float(np.min(robust_margins))
            near_min_margin = robust_margins <= min_margin + 1e-12
            local = int(np.argmin(np.where(near_min_margin, mu_obj, np.inf)))
        else:
            scaled_obj = mu_obj - float(np.min(mu_obj))
            obj_span = float(np.max(scaled_obj))
            if obj_span > 1e-12:
                scaled_obj = scaled_obj / obj_span
            scaled_margin = np.maximum(robust_margins, 0.0)
            margin_span = float(np.max(scaled_margin))
            if margin_span > 1e-12:
                scaled_margin = scaled_margin / margin_span
            local = int(np.argmin(
                scaled_obj
                + self.config.recommendation_infeasible_penalty * scaled_margin
            ))
        used_observed_incumbent = False
        observed_incumbent_rejected = False
        observed_incumbent = self._observed_nominal_incumbent()
        if observed_incumbent is not None:
            try:
                observed_idx = pool.index(observed_incumbent["x"])
                observed_is_robust = bool(robust_margins[observed_idx] <= 0.0)
                if (
                    observed_is_robust
                    and observed_incumbent["empirical_objective"] <= float(mu_obj[local])
                ):
                    local = observed_idx
                    used_observed_incumbent = True
                else:
                    observed_incumbent_rejected = True
            except ValueError:
                pass
        calibrated_recommendation_used = False
        calibrated_details = {}
        calibrated_idx, calibrated_details = self._calibrated_recommendation_index(
            pool,
            robust_margins,
        )
        if calibrated_idx is not None:
            local = calibrated_idx
            calibrated_recommendation_used = True
        x_best = tuple(int(v) for v in pool[local])
        calibration_details = {}
        if calibrated_cert is not None:
            calibration_details = {
                "certification_calibration_used": True,
                "certification_calibration_sigma": float(calibrated_cert["sigma"]),
                "certification_calibration_resid_sigma": float(
                    calibrated_cert["resid_sigma"]),
                "certification_calibration_n_train": int(calibrated_cert["n_train"]),
                "certification_calibration_feature_dim": int(
                    calibrated_cert["feature_dim"]),
                "certification_calibration_beta_g": float(
                    calibrated_cert["beta_g"]),
                "certification_calibration_n_feasible": int(
                    calibrated_cert["n_feasible"]),
                "posterior_calibrated_chance_margin": float(
                    calibrated_margins[local]),
            }
        else:
            calibration_details = {
                "certification_calibration_used": False,
                "posterior_calibrated_chance_margin": None,
            }
        return x_best, {
            "posterior_mu_obj": float(mu_obj[local]),
            "posterior_mu_con": float(effective_mu_con[local]),
            "posterior_gpr_mu_con": float(mu_con[local]),
            "posterior_epistemic_variance_con": float(effective_epistemic[local]),
            "posterior_gpr_epistemic_variance_con": float(cert.epistemic_var[local]),
            "posterior_variance_con": float(effective_aleatoric[local]),
            "posterior_hvd_variance_con": float(cert.aleatoric_var[local]),
            "posterior_beta_g": float(cert.beta_g),
            "certification_mode": cert.mode,
            "posterior_chance_margin": float(robust_margins[local]),
            "posterior_theory_chance_margin": float(theory_margins[local]),
            "posterior_robust_chance_margin": float(robust_margins[local]),
            "posterior_certification_source": str(certification_sources[local]),
            "recommendation_safety_z": float(self.config.recommendation_safety_z),
            "recommendation_noise_floor_scale": float(
                self.config.recommendation_noise_floor_scale),
            "recommendation_infeasible_penalty": float(
                self.config.recommendation_infeasible_penalty),
            "recommendation_calibration": bool(self.config.recommendation_calibration),
            "calibrated_recommendation_used": bool(calibrated_recommendation_used),
            **calibration_details,
            **calibrated_details,
            "observed_incumbent_used": bool(used_observed_incumbent),
            "observed_incumbent_rejected": bool(observed_incumbent_rejected),
            "observed_incumbent_objective": (
                None if observed_incumbent is None
                else float(observed_incumbent["empirical_objective"])
            ),
            "observed_incumbent_chance_margin": (
                None if observed_incumbent is None
                else float(observed_incumbent["empirical_chance_margin"])
            ),
            "posterior_feasible": bool(feasible[local]),
            "n_pool": int(len(pool)),
            "n_posterior_feasible": int(np.sum(feasible)),
            "n_theory_posterior_feasible": int(np.sum(theory_margins <= 0.0)),
        }

    def _terminal_value_from_models(self, gpr_models, variance_model, pool):
        """Terminal certified value used by the optional exact-update KG.

        Lower is better.  If the fixed terminal pool has no robust-feasible
        point, use the same normalized infeasibility penalty shape as
        `_solve_posterior_recommendation` so the value remains comparable.
        """
        if len(pool) == 0:
            return 0.0
        mu_obj = gpr_models[0].posterior_mean_many(pool)
        mu_con = gpr_models[1].posterior_mean_many(pool)
        v_con = variance_model.predict_certification_variance_many(
            1, pool, self.problem)
        epistemic = gpr_models[1].posterior_var_many(pool)
        cert = conservative_chance_margin(
            mu_con,
            epistemic,
            v_con,
            tau=self.problem.tau,
            alpha=self.problem.alpha,
            beta_g=self.config.beta_g,
            mode=self.config.certification_mode,
        )
        margins = cert.margin
        sig_con = np.sqrt(np.maximum(cert.aleatoric_var, 1e-12))
        if cert.mode == "theory":
            safety_buffer = np.zeros_like(margins)
        else:
            nominal_floor = (
                self.config.recommendation_noise_floor_scale
                * float(getattr(self.problem, "sigma_level", 0.0))
            )
            safety_buffer = np.maximum(
                self.config.recommendation_safety_z * sig_con,
                nominal_floor,
            )
        robust_margins = margins + safety_buffer
        feasible = robust_margins <= 0.0
        if np.any(feasible):
            return float(np.min(np.where(feasible, mu_obj, np.inf)))
        if str(self.config.recommendation_infeasible_strategy).lower() in (
            "min_margin",
            "lexicographic",
        ):
            min_margin = float(np.min(robust_margins))
            near_min_margin = robust_margins <= min_margin + 1e-12
            return float(np.min(np.where(near_min_margin, mu_obj, np.inf)))
        scaled_obj = mu_obj - float(np.min(mu_obj))
        obj_span = float(np.max(scaled_obj))
        if obj_span > 1e-12:
            scaled_obj = scaled_obj / obj_span
        scaled_margin = np.maximum(robust_margins, 0.0)
        margin_span = float(np.max(scaled_margin))
        if margin_span > 1e-12:
            scaled_margin = scaled_margin / margin_span
        penalized = (
            scaled_obj
            + self.config.recommendation_infeasible_penalty * scaled_margin
        )
        return float(np.min(penalized))

    def _effective_exact_kg_mc_samples(self):
        mode = str(self.config.acquisition_mode or "additive").lower()
        if (
            mode == "additive"
            and not self.config.exact_kg_use_score
            and float(self.config.exact_kg_blend) <= 0.0
        ):
            return 0
        if mode in ("exact_mc", "blend") and int(self.config.exact_kg_mc_samples) <= 0:
            return 8
        return int(self.config.exact_kg_mc_samples)

    def _exact_posterior_update_score_one(
        self,
        x,
        common_z,
        terminal_pool,
        current_value,
    ):
        x_arr = np.asarray(x, dtype=int)
        mu_before = [self.gpr[i].posterior_mean(x_arr) for i in range(2)]
        sigma2_before = [
            self.variance_model.predict_variance(i, x_arr, self.problem)
            for i in range(2)
        ]
        pred_sd = [
            np.sqrt(max(
                float(sigma2_before[i]) + self.gpr[i].posterior_var(x_arr),
                1e-12,
            ))
            for i in range(2)
        ]
        gains = []
        for z_vec in common_z:
            gpr_clone = [
                self._clone_gpr_for_exact_kg(model)
                for model in self.gpr
            ]
            var_clone = self._clone_variance_model_for_exact_kg()
            y = [
                float(mu_before[i] + pred_sd[i] * z_vec[i])
                for i in range(2)
            ]
            for i in range(2):
                gpr_clone[i].update(x_arr, y[i], sigma2_before[i])
            for i in range(2):
                var_clone.update(
                    i,
                    x_arr,
                    y[i],
                    mu_before[i],
                    gpr_clone[i],
                    self.problem,
                )
            future_value = self._terminal_value_from_models(
                gpr_clone, var_clone, terminal_pool)
            gains.append(current_value - future_value)
        return max(float(np.mean(gains)), 0.0)

    def _exact_posterior_update_scores(self, candidates, terminal_pool):
        """Monte Carlo exact posterior-update KG over a fixed terminal pool.

        This is intentionally optional and small-budget friendly.  It samples
        predictive observations, applies the same GPR update and HVD residual
        update as the main loop, and measures current terminal value minus
        updated terminal value.  Candidate-level work is embarrassingly
        parallel; thread workers avoid process pickling of SUMO/problem handles.
        """
        mc = self._effective_exact_kg_mc_samples()
        if mc <= 0 or len(candidates) == 0:
            return np.zeros(len(candidates), dtype=float)
        current_value = self._terminal_value_from_models(
            self.gpr, self.variance_model, terminal_pool)
        self._last_exact_kg_current_value = float(current_value)
        common_z = self.rng.standard_normal((mc, 2))
        out = np.zeros(len(candidates), dtype=float)
        jobs = max(1, int(self.config.exact_kg_jobs))
        jobs = min(jobs, len(candidates))
        stage_n = int(getattr(self, "_progress_stage_n", len(self.history)))
        step_started_at = float(
            getattr(self, "_progress_step_started_at", time.perf_counter()))
        run_started_at = float(
            getattr(self, "_progress_run_started_at", step_started_at))
        emit_updates = max(1, int(getattr(self.config, "progress_exact_updates", 10)))
        emit_every = max(1, int(np.ceil(len(candidates) / float(emit_updates))))
        self._progress_emit(
            n=stage_n,
            frac=0.35,
            kind="exact_kg_start",
            started_at=step_started_at,
            run_started_at=run_started_at,
            extra=f"candidates={len(candidates)} mc={int(mc)} jobs={int(jobs)}",
        )
        if jobs <= 1:
            for j, x in enumerate(candidates):
                out[j] = self._exact_posterior_update_score_one(
                    x, common_z, terminal_pool, current_value)
                done = j + 1
                if done == len(candidates) or done % emit_every == 0:
                    frac = 0.35 + 0.55 * (float(done) / float(len(candidates)))
                    self._progress_emit(
                        n=stage_n,
                        frac=frac,
                        kind="exact_kg_candidates",
                        started_at=step_started_at,
                        run_started_at=run_started_at,
                        extra=f"candidates_done={done}/{len(candidates)}",
                    )
            return out
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(
                    self._exact_posterior_update_score_one,
                    x,
                    common_z,
                    terminal_pool,
                    current_value,
                ): j
                for j, x in enumerate(candidates)
            }
            done = 0
            for future in as_completed(futures):
                out[futures[future]] = future.result()
                done += 1
                if done == len(candidates) or done % emit_every == 0:
                    frac = 0.35 + 0.55 * (float(done) / float(len(candidates)))
                    self._progress_emit(
                        n=stage_n,
                        frac=frac,
                        kind="exact_kg_candidates",
                        started_at=step_started_at,
                        run_started_at=run_started_at,
                        extra=f"candidates_done={done}/{len(candidates)}",
                    )
        return out

    def _evaluate_recommendation(self, x_best):
        true_obj = self.problem.true_objective(x_best)
        true_con = self.problem.true_constraint_mean(x_best)
        true_sig = self.problem.true_sigma(x_best)
        true_vector = None
        if hasattr(self.problem, "true_vector_objectives"):
            true_vector = [
                float(v)
                for v in self.problem.true_vector_objectives(x_best)
            ]
        true_margin = (
            true_con
            + norm.ppf(1 - self.problem.alpha) * true_sig[1]
            - self.problem.tau
        )
        true_best_x, true_best_obj = self.problem.true_best_feasible()
        true_best_vector = None
        if true_best_x is not None and hasattr(self.problem, "true_vector_objectives"):
            true_best_vector = [
                float(v)
                for v in self.problem.true_vector_objectives(true_best_x)
            ]
        regret = true_obj - true_best_obj if np.isfinite(true_best_obj) else np.nan
        out = {
            "x_recommended": list(map(int, x_best)),
            "true_objective": float(true_obj),
            "true_constraint_mean": float(true_con),
            "true_constraint_sigma": float(true_sig[1]),
            "true_chance_margin": float(true_margin),
            "true_feasible": bool(true_margin <= 0.0),
            "true_best_x": None if true_best_x is None else list(map(int, true_best_x)),
            "true_best_objective": float(true_best_obj),
            "simple_regret": float(regret),
        }
        if true_vector is not None:
            out["true_vector_objectives"] = true_vector
            if len(true_vector) >= 2:
                out["true_f1"] = float(true_vector[0])
                out["true_f2"] = float(true_vector[1])
        if true_best_vector is not None:
            out["true_best_vector_objectives"] = true_best_vector
            if len(true_best_vector) >= 2:
                out["true_best_f1"] = float(true_best_vector[0])
                out["true_best_f2"] = float(true_best_vector[1])
        return out

    def run(self, verbose=False):
        t_start = time.time()
        t_progress_start = time.perf_counter()
        start_n = self._initialize_or_resume(verbose=verbose)
        if self.final_log is not None and int(start_n) >= int(self.config.N):
            return self.final_log

        for n in range(int(start_n), self.config.N):
            iteration = n - self.config.n0
            row = {"iteration": iteration, "stage": n}
            t_iter = time.time()
            t_iter_progress = time.perf_counter()
            self._progress_stage_n = int(n)
            self._progress_step_started_at = float(t_iter_progress)
            self._progress_run_started_at = float(t_progress_start)

            t0 = time.time()
            rec_x, rec_details = self._solve_posterior_recommendation()
            row["t_posterior_solve"] = time.time() - t0
            row["recommendation_before"] = list(map(int, rec_x))
            row.update({f"rec_{k}": v for k, v in rec_details.items()})

            t0 = time.time()
            candidates, candidate_sources = self._generate_candidates(iteration)
            row["t_candidate_gen"] = time.time() - t0
            row["n_candidates"] = len(candidates)
            row["llm_prior"] = dict(self._last_llm_prior_info)

            t0 = time.time()
            score = self.acquisition.score(
                candidates,
                self.gpr[0],
                self.gpr[1],
                self.variance_model,
                self.problem,
                observed=self.history,
            )
            exact_mc_samples = self._effective_exact_kg_mc_samples()
            acquisition_mode = str(self.config.acquisition_mode or "additive").lower()
            if exact_mc_samples > 0:
                terminal_pool = self._recommendation_pool()
                exact_kg = self._exact_posterior_update_scores(
                    candidates, terminal_pool)
                score["exact_kg"] = exact_kg
                row["exact_kg_mc_samples"] = int(exact_mc_samples)
                row["exact_kg_jobs"] = int(max(1, self.config.exact_kg_jobs))
                row["exact_kg_parallel_backend"] = (
                    "thread"
                    if int(self.config.exact_kg_jobs) > 1 and len(candidates) > 1
                    else "serial"
                )
                row["acquisition_mode"] = acquisition_mode
                row["certified_terminal_value_before"] = float(getattr(
                    self,
                    "_last_exact_kg_current_value",
                    np.nan,
                ))
                blend = 0.0
                if acquisition_mode == "exact_mc" or self.config.exact_kg_use_score:
                    score["total"] = exact_kg
                else:
                    blend = float(np.clip(self.config.exact_kg_blend, 0.0, 1.0))
                    if acquisition_mode == "blend" and blend <= 0.0:
                        blend = 0.5
                if acquisition_mode == "blend" or (
                    acquisition_mode == "additive" and float(self.config.exact_kg_blend) > 0.0
                ):
                    score["total"] = (
                        (1.0 - blend) * score["total"]
                        + blend * exact_kg
                    )
            selected_idx = int(np.argmax(score["total"]))
            x_selected = candidates[selected_idx]
            row["t_kg_compute"] = time.time() - t0
            row["x_selected"] = list(map(int, x_selected))
            row["candidate_source_selected"] = candidate_sources.get(
                tuple(x_selected), "unknown")
            row["score_selected"] = float(score["total"][selected_idx])
            row["v_C_plus_selected"] = float(
                self.variance_model.predict_certification_variance(
                    1,
                    x_selected,
                    self.problem,
                )
            )
            selected_decomp = self.variance_model.predict_decomposition(
                1,
                x_selected,
                self.problem,
            )
            selected_cumulative = selected_decomp.get("cumulative") or {}
            row["v_C_plus_source"] = (
                "provider_cumulative"
                if selected_cumulative.get("provider_active")
                else "fallback_hvd"
            )
            row["selected_cumulative_blocks"] = selected_cumulative.get("fitted_blocks")
            row["kg_obj_selected"] = float(score["kg_obj"][selected_idx])
            row["kg_obj_scaled_selected"] = float(score["kg_obj_scaled"][selected_idx])
            row["kg_feas_selected"] = float(score["kg_feas"][selected_idx])
            row["kg_var_selected"] = float(score["kg_var"][selected_idx])
            row["kg_coupling_selected"] = float(score["kg_coupling"][selected_idx])
            row["kg_coupling_raw_selected"] = float(
                score["kg_coupling_raw"][selected_idx])
            row["kg_coupling_gate_selected"] = float(
                score["kg_coupling_gate"][selected_idx])
            if "exact_kg" in score:
                row["exact_kg_selected"] = float(score["exact_kg"][selected_idx])

            x_arr = np.asarray(x_selected, dtype=int)
            mu_before = [self.gpr[i].posterior_mean(x_arr) for i in range(2)]
            sigma2_before = [
                self.variance_model.predict_variance(i, x_arr, self.problem)
                for i in range(2)
            ]
            row["mu_before"] = [float(v) for v in mu_before]
            row["sigma2_before"] = [float(v) for v in sigma2_before]

            t0 = time.time()
            y = self._simulate_and_store(x_selected)
            row["t_simulate"] = time.time() - t0
            row["Y_observed"] = [float(v) for v in y]

            t0 = time.time()
            for i in range(2):
                self.gpr[i].update(x_arr, y[i], sigma2_before[i])
            hvd_details = []
            for i in range(2):
                hvd_details.append(self.variance_model.update(
                    i, x_arr, y[i], mu_before[i], self.gpr[i], self.problem))
            row["t_update"] = time.time() - t0
            row["hvd_update"] = hvd_details
            row["n_visited"] = len(self.gpr[0].sampled_set)

            t0 = time.time()
            eval_interval = int(self.config.evaluate_interval)
            if eval_interval > 0 and (
                iteration % eval_interval == 0 or n == self.config.N - 1
            ):
                rec_x_after, rec_after = self._solve_posterior_recommendation()
                eval_after = self._evaluate_recommendation(rec_x_after)
                row["recommendation_after"] = list(map(int, rec_x_after))
                row["eval"] = {**rec_after, **eval_after}
            row["t_eval"] = time.time() - t0
            row["t_total"] = time.time() - t_iter
            self.iteration_log.append(row)
            self._save_checkpoint(n + 1, reason="iteration")
            self._progress_emit(
                n=n,
                frac=1.0,
                kind="iteration_done",
                started_at=t_iter_progress,
                run_started_at=t_progress_start,
                extra=(
                    f"kg={row['t_kg_compute']:.3f}s "
                    f"cand={row['t_candidate_gen']:.3f}s "
                    f"sim={row['t_simulate']:.3f}s "
                    f"update={row['t_update']:.3f}s "
                    f"eval={row['t_eval']:.3f}s"
                ),
            )
            if verbose:
                print(
                    f"iter={iteration:03d} x={x_selected} "
                    f"score={row['score_selected']:.4g}"
                )

        candidate_source_counts = {}
        llm_status_counts = {}
        llm_gates = []
        for row in self.iteration_log:
            source = str(row.get("candidate_source_selected", "unknown"))
            candidate_source_counts[source] = candidate_source_counts.get(source, 0) + 1
            info = row.get("llm_prior") or {}
            status = str(info.get("status", "missing"))
            llm_status_counts[status] = llm_status_counts.get(status, 0) + 1
            if "gate" in info:
                try:
                    llm_gates.append(float(info.get("gate", 0.0)))
                except (TypeError, ValueError):
                    pass
        llm_prior_summary = {
            "enabled": bool(self.llm_prior is not None),
            "last": dict(self._last_llm_prior_info),
            "status_counts": llm_status_counts,
            "called_count": int(sum(
                count for status, count in llm_status_counts.items()
                if status not in {"disabled", "skipped", "not_called", "missing"}
            )),
            "ok_count": int(llm_status_counts.get("ok", 0)),
            "selected_count": int(candidate_source_counts.get("llm_prior", 0)),
            "gate_mean": float(np.mean(llm_gates)) if llm_gates else 0.0,
            "gate_max": float(np.max(llm_gates)) if llm_gates else 0.0,
        }

        final_x, final_post = self._solve_posterior_recommendation()
        final_eval = self._evaluate_recommendation(final_x)
        self.final_log = {
            **final_post,
            **final_eval,
            "total_time_sec": float(time.time() - t_start),
            "n_simulations": int(len(self.history)),
            "n_distinct_solutions": int(len(self.gpr[0].sampled_set)),
            "stage_times": summarize_stage_times(self.iteration_log),
            "candidate_source_counts": candidate_source_counts,
            "llm_prior": llm_prior_summary,
            "variance": self.variance_model.diagnostics(),
            "cumulative_risk_provider": (
                self.problem.cumulative_risk_provider_status()
                if hasattr(self.problem, "cumulative_risk_provider_status")
                else {"status": "missing_provider"}
            ),
            "mainline_high_dependence": bool(
                self.config.variance_mode == "factor"
                and self.config.certification_mode == "theory"
                and self.config.use_state_coupling
                and self.config.use_state_basis
                and str(self.config.acquisition_mode).lower() == "exact_mc"
            ),
            "numeric_backend": self.gpr[0].backend_status(),
            "config": asdict(self.config),
        }
        self._save_checkpoint(self.config.N, reason="final", force=True)
        return self.final_log
