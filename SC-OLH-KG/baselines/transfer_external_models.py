"""Adapters around checked-out official transfer-BO repositories.

External code stays in the ignored ``repo/clones`` tree.  An unavailable or
incompatible official runtime raises explicitly; callers never receive a
paper-core fallback while requesting ``implementation=official``.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import types

import numpy as np

from baselines.transfer_learned_models import OFFICIAL_PROVENANCE
from baselines.transfer_models import HyperBOSurrogate, _as_2d


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2] / "repo" / "clones"


def _repo_root():
    return Path(os.environ.get(
        "SCOLHKG_EXTERNAL_REPO_ROOT", DEFAULT_REPO_ROOT)).resolve()


def _prepend(path):
    path = str(Path(path).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def _task_fingerprint(tasks):
    payload = [
        {
            "name": task.name,
            "X": np.round(task.X, 14).tolist(),
            "y": np.round(task.y, 14).tolist(),
        }
        for task in tasks
    ]
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _adaptive_positive_definite_jitter(
    covariance,
    *,
    initial_jitter,
    max_attempts=16,
):
    """Return the minimally jittered covariance accepted by Cholesky."""

    import torch

    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    identity = torch.eye(
        covariance.shape[-1],
        dtype=covariance.dtype,
        device=covariance.device,
    )
    jitter = max(float(initial_jitter), 1e-12)
    last_error = None
    for retry in range(int(max_attempts)):
        candidate = covariance + jitter * identity
        try:
            torch.linalg.cholesky(candidate)
            return candidate, float(jitter), int(retry)
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            jitter *= 2.0
    raise RuntimeError(
        "unable to stabilize F-PACOH functional-prior covariance after "
        f"{int(max_attempts)} jitter attempts"
    ) from last_error


class OfficialFPACOHSurrogate:
    adaptation_kind = "posterior_conditioning"
    implementation_family = "safe_fpacoh"
    implementation_fidelity = "official_code_with_compatibility_shims"

    def __init__(self, *, seed=0, source_steps=1000):
        self.seed = int(seed)
        self.source_steps = max(1, int(source_steps))
        self.model = None
        self.n_target = 0

    def meta_fit(self, tasks):
        tasks = list(tasks)
        root = _repo_root() / "jonasrothfuss__f-pacoh-torch"
        if not root.is_dir():
            raise FileNotFoundError(f"missing official F-PACOH repo: {root}")
        _prepend(root)
        import torch
        import gpytorch
        import gpytorch.utils.broadcasting as broadcasting
        if not hasattr(broadcasting, "_mul_broadcast_shape"):
            broadcasting._mul_broadcast_shape = (
                lambda *shapes, **kwargs: torch.broadcast_shapes(*[
                    torch.Size(shape) for shape in shapes
                ])
            )
        from meta_bo.domain import ContinuousDomain
        from meta_bo.models.f_pacoh_map import FPACOH_MAP_GP
        from torch.distributions import MultivariateNormal, kl_divergence

        dimension = int(tasks[0].X.shape[1])
        self.model = FPACOH_MAP_GP(
            ContinuousDomain(np.zeros(dimension), np.ones(dimension)),
            learning_mode="both",
            num_iter_fit=self.source_steps,
            covar_module="SE",
            mean_module="constant",
            num_samples_kl=min(20, max(4, dimension)),
            task_batch_size=min(10, len(tasks)),
            normalize_data=True,
            random_state=np.random.RandomState(self.seed),
        )
        # The official repository targets an older GPyTorch distance API.
        # Replacing only the functional hyper-prior kernel preserves the
        # F-PACOH objective while restoring current-library semantics.
        self.model.prior_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=dimension))

        # The official recovery loop catches RuntimeError, while current
        # torch.distributions raises ValueError for the same non-PD matrix.
        # Preserve the functional KL and its adaptive diagonal-jitter intent.
        def stable_functional_kl(model, task_dict):
            with gpytorch.settings.debug(False):
                x_kl = model._sample_measurement_set(task_dict["x_train"])
                posterior = task_dict["model"](x_kl)
                prior_covariance = torch.reshape(
                    model.prior_covar_module(x_kl).evaluate(),
                    (x_kl.shape[0], x_kl.shape[0]),
                )
                covariance, jitter, retries = (
                    _adaptive_positive_definite_jitter(
                        prior_covariance,
                        initial_jitter=model.prior_kernel_noise,
                    )
                )
                model._scolhkg_fkl_max_jitter = max(
                    float(getattr(
                        model, "_scolhkg_fkl_max_jitter", 0.0)),
                    float(jitter),
                )
                model._scolhkg_fkl_retry_count = int(getattr(
                    model, "_scolhkg_fkl_retry_count", 0)) + int(retries)
                prior = MultivariateNormal(
                    torch.zeros(
                        x_kl.shape[0],
                        dtype=covariance.dtype,
                        device=covariance.device,
                    ),
                    covariance_matrix=covariance,
                )
                return kl_divergence(posterior, prior)

        self.model._f_kl = types.MethodType(
            stable_functional_kl, self.model)
        train = [
            (np.asarray(task.X).copy(), np.asarray(task.y).copy())
            for task in tasks
        ]
        self.model.meta_fit(
            train,
            verbose=False,
            n_iter=self.source_steps,
        )
        return self

    def adapt(self, X, y, noise):
        self.model.reset_to_prior()
        y = np.asarray(y, dtype=float).reshape(-1).copy()
        if len(y):
            self.model.add_data(np.asarray(X, dtype=float).copy(), y)
        self.n_target = int(len(y))

    def predict(self, X, full_cov=False):
        mean, std = self.model.predict(
            np.asarray(X, dtype=float), include_obs_noise=False)
        variance = np.maximum(np.asarray(std, dtype=float) ** 2, 1e-12)
        if full_cov:
            return np.asarray(mean, dtype=float), np.diag(variance)
        return np.asarray(mean, dtype=float), variance

    def diagnostics(self):
        return {
            "family": self.implementation_family,
            "adaptation_kind": self.adaptation_kind,
            "implementation_fidelity": self.implementation_fidelity,
            "official_provenance": OFFICIAL_PROVENANCE["safe_fpacoh"],
            "compatibility_shims": [
                "gpytorch_removed_mul_broadcast_shape",
                "current_gpytorch_rbf_functional_prior",
                "torch_valueerror_adaptive_functional_kl_jitter",
            ],
            "source_prior_frozen_online": True,
            "online_parameters_changed": ["target_gp_posterior"],
            "source_steps": int(self.source_steps),
            "n_target": int(self.n_target),
            "functional_kl_max_jitter": float(getattr(
                self.model, "_scolhkg_fkl_max_jitter", 0.0)),
            "functional_kl_jitter_retries": int(getattr(
                self.model, "_scolhkg_fkl_retry_count", 0)),
        }


class OfficialFSBOSurrogate:
    adaptation_kind = "end_to_end_gradient_finetuning"
    implementation_family = "fsbo"
    implementation_fidelity = "official_code_adapted_to_scalar_cbo"

    def __init__(
        self,
        *,
        seed=0,
        source_steps=1000,
        target_steps=100,
        checkpoint_dir=None,
    ):
        self.seed = int(seed)
        self.source_steps = max(1, int(source_steps))
        self.target_steps = max(1, int(target_steps))
        self.checkpoint_dir = Path(checkpoint_dir or (
            Path.cwd() / "checkpoints" / "official_fsbo" / f"seed{seed}"
        ))
        self.model = None
        self.dimension = None
        self.y_center = 0.0
        self.y_scale = 1.0
        self.n_target = 0

    @staticmethod
    def _compatibility_shims():
        if not hasattr(np, "asscalar"):
            np.asscalar = lambda value: np.asarray(
                value.detach().cpu() if hasattr(value, "detach") else value
            ).item()
        if "xgboost" not in sys.modules:
            # The official surrogate imports xgboost for an unused optional
            # path.  Do not introduce that dependency into FSBO GP training.
            sys.modules["xgboost"] = types.ModuleType("xgboost")

    def _modules(self):
        root = _repo_root() / "machinelearningnuremberg__FSBO"
        if not root.is_dir():
            raise FileNotFoundError(f"missing official FSBO repo: {root}")
        _prepend(root)
        self._compatibility_shims()
        from fsbo_modules import DeepKernelGP, FSBO
        from fsbo_utils import totorch
        return DeepKernelGP, FSBO, totorch

    def meta_fit(self, tasks):
        tasks = list(tasks)
        self.dimension = int(tasks[0].X.shape[1])
        values = np.concatenate([np.asarray(task.y) for task in tasks])
        self.y_center = float(np.mean(values))
        self.y_scale = max(float(np.std(values)), 1e-8)
        fingerprint = _task_fingerprint(tasks)
        metadata_path = self.checkpoint_dir / "scolhkg_archive.json"
        weights_path = self.checkpoint_dir / "weights"
        cached = False
        if metadata_path.is_file() and weights_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cached = metadata.get("fingerprint") == fingerprint
        if not cached:
            _, FSBO, _ = self._modules()
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            data = {
                str(index): {
                    "X": np.asarray(task.X, dtype=float).tolist(),
                    "y": np.asarray(task.y, dtype=float).reshape(-1, 1).tolist(),
                }
                for index, task in enumerate(tasks)
            }
            batch_size = min(32, max(2, min(len(task.y) for task in tasks) // 2))
            with redirect_stdout(io.StringIO()):
                model = FSBO(
                    train_data=data,
                    valid_data=copy.deepcopy(data),
                    checkpoint_path=str(self.checkpoint_dir),
                    batch_size=batch_size,
                )
                model.meta_train(
                    epochs=self.source_steps,
                    lr=1e-4,
                )
            metadata_path.write_text(json.dumps({
                "fingerprint": fingerprint,
                "source_steps": self.source_steps,
            }), encoding="utf-8")
        return self

    def adapt(self, X, y, noise):
        import torch
        DeepKernelGP, _, totorch = self._modules()
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.n_target = int(len(y))
        if not self.n_target:
            self.model = None
            return
        normalized = (y - self.y_center) / self.y_scale
        log_path = self.checkpoint_dir / f"target_seed{self.seed}.log"
        with redirect_stdout(io.StringIO()):
            self.model = DeepKernelGP(
                self.dimension,
                str(log_path),
                self.seed,
                epochs=self.target_steps,
                load_model=True,
                checkpoint=str(self.checkpoint_dir),
                verbose=False,
            )
            self.model.X_obs = totorch(X, self.model.device)
            self.model.y_obs = totorch(
                normalized.reshape(-1), self.model.device)
            self.model.train()
        self.model.model.eval()
        self.model.feature_extractor.eval()
        self.model.likelihood.eval()

    def predict(self, X, full_cov=False):
        if self.model is None:
            mean = np.full(len(_as_2d(X)), self.y_center)
            variance = np.full(len(mean), self.y_scale ** 2)
        else:
            _, _, totorch = self._modules()
            values = totorch(_as_2d(X), self.model.device)
            mean0, std0 = self.model.predict(values)
            mean = self.y_center + self.y_scale * np.asarray(mean0)
            variance = np.maximum(
                (self.y_scale * np.asarray(std0)) ** 2, 1e-12)
        if full_cov:
            return mean, np.diag(variance)
        return mean, variance

    def diagnostics(self):
        return {
            "family": self.implementation_family,
            "adaptation_kind": self.adaptation_kind,
            "implementation_fidelity": self.implementation_fidelity,
            "official_provenance": OFFICIAL_PROVENANCE["fsbo"],
            "compatibility_shims": [
                "numpy_removed_asscalar",
                "unused_xgboost_import_stub",
            ],
            "source_prior_frozen_online": False,
            "online_parameters_changed": [
                "deep_feature_extractor",
                "target_gp_kernel",
                "target_gp_likelihood",
            ],
            "source_steps": int(self.source_steps),
            "target_gradient_steps": int(self.target_steps),
            "n_target": int(self.n_target),
        }


class OfficialMALIBOSurrogate:
    adaptation_kind = "bayesian_utility_head_adaptation"
    implementation_family = "malibo"
    implementation_fidelity = "official_metablor_core_adapted_to_cbo"

    def __init__(self, *, seed=0, source_steps=512):
        self.seed = int(seed)
        self.source_steps = max(1, int(source_steps))
        self.classifier = None
        self.dimension = None
        self.n_target = 0

    @staticmethod
    def _reweight(X, y, gamma=1.0 / 3.0):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        threshold = float(np.quantile(np.unique(y), q=gamma))
        positive = y < threshold
        if len(X) <= 1 or not np.any(positive):
            return X, positive.astype(float), np.ones(len(X))
        X1 = X[positive]
        z1 = np.ones(len(X1), dtype=float)
        w1 = threshold - y[positive]
        w1 /= max(float(np.mean(w1)), 1e-12)
        X0 = X
        z0 = np.zeros(len(X0), dtype=float)
        values = np.concatenate([w1, np.ones(len(X0))])
        values /= max(float(np.mean(values)), 1e-12)
        return np.vstack([X1, X0]), np.r_[z1, z0], values

    def meta_fit(self, tasks):
        root = _repo_root() / "boschresearch__MALIBO"
        if not root.is_dir():
            raise FileNotFoundError(f"missing official MALIBO repo: {root}")
        _prepend(root)
        import torch
        from meta_learning.meta_classifier import MetaBLOR

        tasks = list(tasks)
        self.dimension = int(tasks[0].X.shape[1])
        torch.manual_seed(self.seed)
        self.classifier = MetaBLOR(
            input_dim=self.dimension,
            output_dim=1,
            device="cpu",
            num_layers=5,
            num_features=32,
            num_hidden_units=64,
        )
        metadata = {}
        for index, task in enumerate(tasks, start=1):
            X, labels, weights = self._reweight(task.X, task.y)
            metadata[index] = {
                "X": X,
                "Y": labels.reshape(-1, 1),
                "w": weights.reshape(-1, 1),
            }
        with redirect_stdout(io.StringIO()):
            self.classifier.meta_fit(
                metadata,
                num_epochs=self.source_steps,
                batch_size=min(128, sum(len(task.y) for task in tasks)),
            )
        return self

    def adapt(self, X, y, noise):
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.n_target = int(len(y))
        if self.n_target:
            features, labels, weights = self._reweight(X, y)
            self.classifier.fit(features, labels, sample_weight=weights)

    def utility_moments(self, X):
        probability = np.asarray(self.classifier.predict(
            _as_2d(X), sampling="max"), dtype=float).reshape(-1)
        variance = np.maximum(probability * (1.0 - probability), 1e-8)
        return probability, variance

    def acquisition_scores(self, X, incumbent=None, progress=None):
        return self.utility_moments(X)[0]

    def predict(self, X, full_cov=False):
        probability, variance = self.utility_moments(X)
        mean = -probability
        if full_cov:
            return mean, np.diag(variance)
        return mean, variance

    def diagnostics(self):
        return {
            "family": self.implementation_family,
            "adaptation_kind": self.adaptation_kind,
            "implementation_fidelity": self.implementation_fidelity,
            "official_provenance": OFFICIAL_PROVENANCE["malibo"],
            "source_prior_frozen_online": True,
            "online_parameters_changed": ["target_bayesian_utility_head"],
            "source_steps": int(self.source_steps),
            "n_target": int(self.n_target),
        }


class OfficialTransferGPBOSurrogate:
    """Pinned TransferGPBO RGPE, SHGP, or MTGP model."""

    implementation_fidelity = "official_transfergpbo_code"

    def __init__(self, family, *, seed=0):
        if family not in {"rgpe", "shgp", "mtgp"}:
            raise ValueError(f"unknown official TransferGPBO family {family!r}")
        self.family = family
        self.seed = int(seed)
        self.model = None
        self.dimension = None
        self.n_target = 0

    @staticmethod
    def _modules():
        overlay = os.environ.get("SCOLHKG_TRANSFERGPBO_OVERLAY", "").strip()
        if overlay:
            _prepend(overlay)
        root = _repo_root() / "boschresearch__transfergpbo"
        if not root.is_dir():
            raise FileNotFoundError(
                f"missing official TransferGPBO repo: {root}")
        _prepend(root)
        try:
            import GPy
            from transfergpbo.models import InputData, TaskData
            from transfergpbo.models.mtgp import MTGP
            from transfergpbo.models.rgpe import RGPE
            from transfergpbo.models.shgp import SHGP
        except ImportError as exc:
            raise RuntimeError(
                "official TransferGPBO needs its GPy/emukit overlay on "
                "PYTHONPATH before Python starts; NumPy ABIs cannot be swapped "
                "inside a running process"
            ) from exc
        return GPy, InputData, TaskData, RGPE, SHGP, MTGP

    def meta_fit(self, tasks):
        GPy, _, TaskData, RGPE, SHGP, MTGP = self._modules()
        tasks = list(tasks)
        self.dimension = int(tasks[0].X.shape[1])
        metadata = {
            index: TaskData(
                X=np.asarray(task.X, dtype=float),
                Y=np.asarray(task.y, dtype=float).reshape(-1, 1),
            )
            for index, task in enumerate(tasks)
        }
        np.random.seed(self.seed)
        if self.family == "rgpe":
            self.model = RGPE(
                n_samples=256,
                kernel=GPy.kern.Matern52(self.dimension, ARD=True),
                noise_variance=float(np.median(np.concatenate([
                    np.asarray(task.noise) for task in tasks
                ]))),
                normalize=True,
            )
            self.model.meta_fit(metadata)
        elif self.family == "shgp":
            self.model = SHGP(n_features=self.dimension)
            self.model.meta_fit(metadata, optimize=True)
        else:
            self.model = MTGP(
                kernel=GPy.kern.Matern52(self.dimension, ARD=True),
                noise_variance=float(np.median(np.concatenate([
                    np.asarray(task.noise) for task in tasks
                ]))),
                normalize=True,
            )
            self.model.meta_fit(metadata)
        return self

    def adapt(self, X, y, noise):
        _, _, TaskData, _, _, _ = self._modules()
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.n_target = int(len(y))
        if self.n_target:
            self.model.fit(
                TaskData(X=X, Y=y.reshape(-1, 1)),
                optimize=False,
            )

    def predict(self, X, full_cov=False):
        _, InputData, _, _, _, _ = self._modules()
        mean, variance = self.model.predict(
            InputData(X=_as_2d(X)),
            return_full=full_cov,
            with_noise=False,
        )
        mean = np.asarray(mean, dtype=float).reshape(-1)
        variance = np.asarray(variance, dtype=float)
        if full_cov:
            return mean, variance
        return mean, np.maximum(variance.reshape(-1), 1e-12)

    def diagnostics(self):
        adaptation = {
            "rgpe": "expert_reweighting_and_posterior_conditioning",
            "shgp": "source_target_discrepancy_posterior",
            "mtgp": "joint_multitask_posterior",
        }[self.family]
        return {
            "family": self.family,
            "adaptation_kind": adaptation,
            "implementation_fidelity": self.implementation_fidelity,
            "official_provenance": {
                "repository": "boschresearch/transfergpbo",
                "commit": "4bdf90498c0048d7c80327609961d795d82b6f43",
            },
            "isolated_dependency_overlay": os.environ.get(
                "SCOLHKG_TRANSFERGPBO_OVERLAY"),
            "source_prior_frozen_online": True,
            "online_parameters_changed": (
                ["target_gp_posterior", "source_expert_weights"]
                if self.family == "rgpe"
                else ["target_gp_posterior"]
            ),
            "n_target": int(self.n_target),
        }


class OfficialHyperBOSurrogate:
    """Pinned HyperBO GP-prior training and target conditioning."""

    adaptation_kind = "posterior_conditioning"
    implementation_family = "hyperbo_pretrained_gp_prior"
    implementation_fidelity = "official_hyperbo_code_with_gfile_shim"

    def __init__(self, *, seed=0, source_steps=10_000):
        self.seed = int(seed)
        self.source_steps = max(1, int(source_steps))
        self.params = None
        self.target_X = None
        self.target_y = None
        self.n_target = 0

    @staticmethod
    def _tensorflow_gfile_shim():
        if "tensorflow" in sys.modules:
            return
        import glob as glob_module

        class GFileShim:
            GFile = staticmethod(open)
            exists = staticmethod(os.path.exists)
            makedirs = staticmethod(
                lambda path: os.makedirs(path, exist_ok=True))
            glob = staticmethod(glob_module.glob)

        tensorflow = types.ModuleType("tensorflow")
        tensorflow_io = types.ModuleType("tensorflow.io")
        tensorflow_io.gfile = GFileShim
        tensorflow.io = tensorflow_io
        sys.modules["tensorflow"] = tensorflow
        sys.modules["tensorflow.io"] = tensorflow_io

    @classmethod
    def _modules(cls):
        overlay = os.environ.get("SCOLHKG_HYPERBO_OVERLAY", "").strip()
        if overlay:
            _prepend(overlay)
        root = _repo_root() / "google-research__hyperbo"
        if not root.is_dir():
            raise FileNotFoundError(f"missing official HyperBO repo: {root}")
        _prepend(root)
        cls._tensorflow_gfile_shim()
        try:
            import jax
            import jax.numpy as jnp
            from hyperbo.basics.definitions import GPParams
            from hyperbo.gp_utils import gp, kernel, mean, utils
        except ImportError as exc:
            raise RuntimeError(
                "official HyperBO needs the isolated JAX/Flax overlay"
            ) from exc
        jax.config.update("jax_enable_x64", True)
        return jax, jnp, GPParams, gp, kernel, mean, utils

    def meta_fit(self, tasks):
        jax, jnp, GPParams, gp, kernel, mean, utils = self._modules()
        tasks = list(tasks)
        dimension = int(tasks[0].X.shape[1])
        dataset = [
            (
                jnp.asarray(task.X, dtype=jnp.float64),
                jnp.asarray(task.y, dtype=jnp.float64).reshape(-1, 1),
            )
            for task in tasks
        ]
        params = GPParams(
            model={
                "constant": 0.0,
                "lengthscale": jnp.full(dimension, -1.5),
                "signal_variance": 0.0,
                "noise_variance": -4.0,
            },
            config={
                "method": "adam",
                "max_training_step": int(self.source_steps),
                "batch_size": int(sum(len(task.y) for task in tasks)),
                "learning_rate": 1e-3,
            },
        )
        model = gp.GP(
            dataset=dataset,
            mean_func=mean.constant,
            cov_func=kernel.squared_exponential,
            params=params,
            warp_func=utils.DEFAULT_WARP_FUNC,
        )
        key = jax.random.PRNGKey(self.seed)
        model.initialize_params(key)
        trained = model.train(key=key)
        self.params = GPParams(
            model=copy.deepcopy(trained.model),
            config={"max_training_step": 0},
            cache={},
        )
        return self

    def adapt(self, X, y, noise):
        _, jnp, _, _, _, _, _ = self._modules()
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.target_X = jnp.asarray(X, dtype=jnp.float64)
        self.target_y = jnp.asarray(y, dtype=jnp.float64).reshape(-1, 1)
        self.n_target = int(len(y))

    def predict(self, X, full_cov=False):
        _, jnp, _, gp, kernel, mean, utils = self._modules()
        query = jnp.asarray(_as_2d(X), dtype=jnp.float64)
        observed_X = self.target_X if self.n_target else None
        observed_y = self.target_y if self.n_target else None
        posterior_mean, posterior_variance = gp.predict(
            mean.constant,
            kernel.squared_exponential,
            self.params,
            observed_X,
            observed_y,
            query,
            warp_func=utils.DEFAULT_WARP_FUNC,
            full_cov=full_cov,
        )
        posterior_mean = np.asarray(posterior_mean).reshape(-1)
        posterior_variance = np.asarray(posterior_variance)
        if full_cov:
            return posterior_mean, posterior_variance
        return posterior_mean, np.maximum(
            posterior_variance.reshape(-1), 1e-12)

    def diagnostics(self):
        return {
            "family": self.implementation_family,
            "adaptation_kind": self.adaptation_kind,
            "implementation_fidelity": self.implementation_fidelity,
            "official_provenance": {
                "repository": "google-research/hyperbo",
                "commit": "e720fc1",
            },
            "compatibility_shims": ["tensorflow_io_gfile_only"],
            "source_prior_frozen_online": True,
            "online_parameters_changed": ["target_gp_posterior"],
            "source_steps": int(self.source_steps),
            "n_target": int(self.n_target),
        }


class OfficialMetaBOSurrogate:
    """Official NeuralAF trained by PPO on the finite source archive.

    The upstream trainer assumes a generative family of source functions and
    can therefore draw an unlimited number of source evaluations.  That is an
    unfair information advantage in our fixed-archive comparison.  This
    adapter keeps the official NeuralAF architecture and clipped PPO objective,
    but obtains every reward and GP state by replaying the disclosed archive.
    Reusing an observed archive row for optimization epochs has zero simulator
    cost; no unobserved source value or analytic optimum is queried.
    """

    adaptation_kind = "frozen_policy_with_target_posterior_state"
    implementation_family = "metabo_source_trained_acquisition_policy"
    implementation_fidelity = (
        "official_neuralaf_ppo_fixed_archive_extension"
    )

    def __init__(
        self,
        *,
        seed=0,
        source_steps=10_000,
        target_budget=20,
        checkpoint_dir=None,
    ):
        self.seed = int(seed)
        self.source_steps = max(1, int(source_steps))
        self.target_budget = max(1, int(target_budget))
        self.checkpoint_dir = Path(checkpoint_dir or (
            Path.cwd() / "checkpoints" / "official_metabo" / f"seed{seed}"
        ))
        self.posterior = HyperBOSurrogate(kernel="rbf")
        self.policy = None
        self.y_center = 0.0
        self.y_scale = 1.0
        self.n_target = 0
        self.source_transitions_trained = 0
        self.loaded_source_checkpoint = False

    @staticmethod
    def _modules():
        overlay = os.environ.get("SCOLHKG_TRANSFERGPBO_OVERLAY", "").strip()
        if overlay:
            _prepend(overlay)
        root = _repo_root() / "boschresearch__MetaBO"
        if not root.is_dir():
            raise FileNotFoundError(f"missing official MetaBO repo: {root}")
        _prepend(root)
        try:
            import torch
            from torch.distributions import Categorical
            from metabo.policies.policies import NeuralAF
        except ImportError as exc:
            raise RuntimeError(
                "official MetaBO needs PyTorch and the GPy compatibility "
                "overlay on PYTHONPATH before Python starts"
            ) from exc
        return torch, Categorical, NeuralAF

    @staticmethod
    def _policy_options():
        # The official GP-family experiment uses two 20-unit hidden layers.
        return {
            "activations": "relu",
            "arch_spec": [20, 20],
            "exclude_t_from_policy": True,
            "exclude_T_from_policy": True,
            "use_value_network": True,
            "t_idx": -2,
            "T_idx": -1,
            "arch_spec_value": [20, 20],
        }

    def _new_policy(self, cardinality):
        torch, _, NeuralAF = self._modules()
        torch.manual_seed(self.seed)
        observation_space = types.SimpleNamespace(
            shape=(int(cardinality), 6))
        action_space = types.SimpleNamespace(n=int(cardinality))
        return NeuralAF(
            observation_space=observation_space,
            action_space=action_space,
            deterministic=False,
            options=self._policy_options(),
        )

    def _state(self, model, X, incumbent, timestep):
        mean, variance = model.predict(X)
        normalized_gain_mean = -(
            np.asarray(mean, dtype=float) - self.y_center
        ) / self.y_scale
        normalized_std = np.sqrt(np.maximum(
            np.asarray(variance, dtype=float), 1e-12
        )) / self.y_scale
        normalized_incumbent = -(
            float(incumbent) - self.y_center
        ) / self.y_scale
        progress = float(timestep) / float(self.target_budget)
        return np.column_stack([
            normalized_gain_mean,
            normalized_std,
            np.full(len(X), normalized_incumbent),
            np.full(len(X), progress),
            np.full(len(X), float(timestep)),
            np.full(len(X), float(self.target_budget)),
        ]).astype(np.float32)

    def _collect_episode(self, task, rng):
        torch, Categorical, _ = self._modules()
        horizon = min(self.target_budget, len(task.y))
        model = HyperBOSurrogate(kernel="rbf")
        model.hyperparameters = copy.deepcopy(self.posterior.hyperparameters)
        model.normalization = tuple(self.posterior.normalization)
        from baselines.transfer_models import ExactGPSurrogate
        model.target_model = ExactGPSurrogate(
            kernel="rbf",
            normalization=model.normalization,
            **model.hyperparameters,
        )
        selected = []
        transitions = []
        source_best = float(np.min(task.y))
        source_scale = max(float(np.std(task.y)), 1e-8)
        incumbent = float(np.max(task.y))
        for timestep in range(horizon):
            if selected:
                indices = np.asarray(selected, dtype=int)
                model.adapt(
                    task.X[indices], task.y[indices], task.noise[indices])
            else:
                model.adapt(
                    np.empty((0, task.X.shape[1])),
                    np.empty(0),
                    np.empty(0),
                )
            state = self._state(model, task.X, incumbent, timestep)
            state_tensor = torch.from_numpy(state).unsqueeze(0)
            logits, values = self.policy.forward(state_tensor)
            unavailable = torch.zeros(len(task.y), dtype=torch.bool)
            if selected:
                unavailable[torch.as_tensor(selected, dtype=torch.long)] = True
            masked_logits = logits[0].masked_fill(unavailable, -1e9)
            distribution = Categorical(logits=masked_logits)
            action = int(distribution.sample().item())
            old_log_probability = float(
                distribution.log_prob(torch.tensor(action)).detach().item())
            selected.append(action)
            incumbent = min(incumbent, float(task.y[action]))
            relative_regret = max(
                (incumbent - source_best) / source_scale, 1e-6)
            reward = -float(np.log10(relative_regret))
            transitions.append({
                "state": state,
                "unavailable": unavailable.numpy(),
                "action": action,
                "reward": reward,
                "value": float(values[0].detach().item()),
                "old_log_probability": old_log_probability,
            })
        discounted = 0.0
        for row in reversed(transitions):
            discounted = float(row["reward"]) + 0.98 * discounted
            row["return"] = discounted
            row["advantage"] = discounted - float(row["value"])
        # Randomness is consumed only through the torch policy; advance the
        # NumPy stream so task scheduling is checkpoint-reproducible as well.
        rng.random()
        return transitions

    def _ppo_update(self, transitions, optimizer):
        torch, Categorical, _ = self._modules()
        states = torch.from_numpy(np.stack([
            row["state"] for row in transitions
        ]).astype(np.float32))
        unavailable = torch.from_numpy(np.stack([
            row["unavailable"] for row in transitions
        ])).bool()
        actions = torch.as_tensor([
            row["action"] for row in transitions
        ], dtype=torch.long)
        returns = torch.as_tensor([
            row["return"] for row in transitions
        ], dtype=torch.float32)
        advantages = torch.as_tensor([
            row["advantage"] for row in transitions
        ], dtype=torch.float32)
        old_log_probabilities = torch.as_tensor([
            row["old_log_probability"] for row in transitions
        ], dtype=torch.float32)
        if float(torch.std(advantages, unbiased=False)) > 0.0:
            advantages = (
                advantages - torch.mean(advantages)
            ) / torch.std(advantages, unbiased=False)
        for _ in range(4):
            logits, values = self.policy.forward(states)
            logits = logits.masked_fill(unavailable, -1e9)
            distribution = Categorical(logits=logits)
            log_probabilities = distribution.log_prob(actions)
            ratios = torch.exp(log_probabilities - old_log_probabilities)
            clipped = torch.clamp(ratios, 0.85, 1.15)
            policy_loss = -torch.mean(torch.minimum(
                ratios * advantages, clipped * advantages))
            value_loss = torch.mean((values - returns) ** 2)
            entropy_loss = -torch.mean(distribution.entropy())
            loss = policy_loss + value_loss + 0.01 * entropy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def meta_fit(self, tasks):
        torch, _, _ = self._modules()
        tasks = list(tasks)
        cardinalities = {len(task.y) for task in tasks}
        if len(cardinalities) != 1:
            raise ValueError(
                "official MetaBO fixed-archive replay requires an equal "
                "candidate count per source task"
            )
        values = np.concatenate([
            np.asarray(task.y, dtype=float) for task in tasks
        ])
        self.y_center = float(np.mean(values))
        self.y_scale = max(float(np.std(values)), 1e-8)
        self.posterior.meta_fit(tasks)
        self.policy = self._new_policy(cardinalities.pop())
        fingerprint = _task_fingerprint(tasks)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        weights_path = self.checkpoint_dir / "weights.pt"
        metadata_path = self.checkpoint_dir / "archive.json"
        if metadata_path.is_file() and weights_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("fingerprint") == fingerprint
                and int(metadata.get("source_steps", -1)) == self.source_steps
            ):
                self.policy.load_state_dict(torch.load(
                    weights_path, map_location="cpu", weights_only=True))
                self.source_transitions_trained = int(
                    metadata["source_transitions_trained"])
                self.loaded_source_checkpoint = True
                return self
        rng = np.random.default_rng(self.seed)
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        transitions_seen = 0
        task_index = 0
        while transitions_seen < self.source_steps:
            task = tasks[task_index % len(tasks)]
            task_index += 1
            episode = self._collect_episode(task, rng)
            remaining = self.source_steps - transitions_seen
            episode = episode[:remaining]
            self._ppo_update(episode, optimizer)
            transitions_seen += len(episode)
        self.source_transitions_trained = int(transitions_seen)
        torch.save(self.policy.state_dict(), weights_path)
        metadata_path.write_text(json.dumps({
            "fingerprint": fingerprint,
            "source_steps": self.source_steps,
            "source_transitions_trained": self.source_transitions_trained,
        }, sort_keys=True), encoding="utf-8")
        return self

    def adapt(self, X, y, noise):
        self.posterior.adapt(X, y, noise)
        self.n_target = int(len(np.asarray(y).reshape(-1)))

    def predict(self, X, full_cov=False):
        return self.posterior.predict(X, full_cov=full_cov)

    def acquisition_scores(self, X, incumbent=None, progress=None):
        torch, _, _ = self._modules()
        X = _as_2d(X)
        if incumbent is None:
            incumbent = self.y_center
        state = self._state(self.posterior, X, incumbent, self.n_target)
        with torch.no_grad():
            logits, _ = self.policy.forward(
                torch.from_numpy(state).unsqueeze(0))
        return logits[0].detach().cpu().numpy()

    def diagnostics(self):
        return {
            "family": self.implementation_family,
            "adaptation_kind": self.adaptation_kind,
            "implementation_fidelity": self.implementation_fidelity,
            "official_provenance": {
                "repository": "boschresearch/MetaBO",
                "commit": "3f458bd32db340fbe2d5f072a92cfd782072342c",
                "official_components": [
                    "metabo.policies.policies.NeuralAF",
                    "clipped_PPO_objective",
                ],
            },
            "fixed_archive_extension": {
                "reason": "equalize finite source simulator calls",
                "reward_truth": "observed_source_archive_rows_only",
                "analytic_source_oracle_used": False,
                "new_source_calls_during_training": 0,
            },
            "source_prior_frozen_online": True,
            "online_parameters_changed": ["target_gp_posterior"],
            "policy_parameters_changed_online": False,
            "source_steps": int(self.source_steps),
            "source_transitions_trained": int(
                self.source_transitions_trained),
            "loaded_source_checkpoint": bool(
                self.loaded_source_checkpoint),
            "n_target": int(self.n_target),
        }


class _OfficialExtensionHyperBO(HyperBOSurrogate):
    implementation_fidelity = "audited_common_cbo_extension"

    def diagnostics(self):
        values = super().diagnostics()
        values["implementation_fidelity"] = self.implementation_fidelity
        values["role"] = "constraint_or_heteroscedastic_risk_extension"
        return values


def official_model(method, *, role, config, seed):
    """Return an official core or fail rather than silently downgrade."""

    if method == "safe_fpacoh_cbo":
        return OfficialFPACOHSurrogate(
            seed=seed, source_steps=config.source_train_steps)
    if method == "fsbo_cbo":
        checkpoint = Path(config.checkpoint_path or (
            Path.cwd() / "checkpoints" / "official_fsbo.pkl"
        )).with_suffix("").parent / "official_models" / role
        return OfficialFSBOSurrogate(
            seed=seed,
            source_steps=config.source_train_steps,
            target_steps=config.target_finetune_steps,
            checkpoint_dir=checkpoint,
        )
    if method == "malibo_cbo":
        if role == "objective":
            return OfficialMALIBOSurrogate(
                seed=seed, source_steps=config.source_train_steps)
        return _OfficialExtensionHyperBO(kernel="rbf")
    if method == "metabo_cbo":
        if role == "objective":
            checkpoint = Path(config.checkpoint_path or (
                Path.cwd() / "checkpoints" / "official_metabo.pkl"
            )).with_suffix("").parent / "official_models" / role
            return OfficialMetaBOSurrogate(
                seed=seed,
                source_steps=config.source_train_steps,
                target_budget=config.N,
                checkpoint_dir=checkpoint,
            )
        return _OfficialExtensionHyperBO(kernel="rbf")
    if method == "rgpe_cbo":
        return OfficialTransferGPBOSurrogate("rgpe", seed=seed)
    if method == "stacked_transfer_gp_cbo":
        return OfficialTransferGPBOSurrogate("shgp", seed=seed)
    if method == "mtgp_cbo":
        return OfficialTransferGPBOSurrogate("mtgp", seed=seed)
    if method == "hyperbo_cbo":
        return OfficialHyperBOSurrogate(
            seed=seed, source_steps=config.source_train_steps)
    raise RuntimeError(
        f"official runtime for {method} is not configured; use paper_core "
        "only for smoke tests and do not report it as an official SOTA row"
    )


def external_runtime_report():
    root = _repo_root()
    rows = {}
    for name, directory in {
        "safe_fpacoh_cbo": "jonasrothfuss__f-pacoh-torch",
        "fsbo_cbo": "machinelearningnuremberg__FSBO",
        "malibo_cbo": "boschresearch__MALIBO",
        "hyperbo_cbo": "google-research__hyperbo",
        "metabo_cbo": "boschresearch__MetaBO",
        "rgpe_cbo": "boschresearch__transfergpbo",
        "stacked_transfer_gp_cbo": "boschresearch__transfergpbo",
        "mtgp_cbo": "boschresearch__transfergpbo",
    }.items():
        rows[name] = {
            "repository_present": bool((root / directory).is_dir()),
            "repository_path": str(root / directory),
            "official_adapter_configured": name in {
                "safe_fpacoh_cbo",
                "fsbo_cbo",
                "malibo_cbo",
                "rgpe_cbo",
                "stacked_transfer_gp_cbo",
                "mtgp_cbo",
                "hyperbo_cbo",
                "metabo_cbo",
            },
        }
    return rows
