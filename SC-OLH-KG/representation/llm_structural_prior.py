"""Evidence-gated LLM structural prior for SC-OLH-KG.

The LLM is deliberately constrained to propose generic descriptor-space
regions.  It never receives the target problem name, source code, true
objective, true constraint, true optimum, or hidden coordinates.  Candidate
points are still sampled and scored by SC-OLH-KG.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

import numpy as np
import requests
from scipy.stats import norm

from core.candidates import unique_candidates


DESCRIPTOR_KEYS = (
    "mean",
    "std",
    "min",
    "max",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "low_fraction",
    "high_fraction",
    "center_norm",
    "diff_mean_abs",
)


def policy_descriptor(problem, x):
    z = np.asarray(problem.normalize(x), dtype=float).reshape(-1)
    if len(z) == 0:
        z = np.zeros(1, dtype=float)
    diffs = np.diff(z) if len(z) > 1 else np.array([0.0], dtype=float)
    vals = {
        "mean": float(np.mean(z)),
        "std": float(np.std(z)),
        "min": float(np.min(z)),
        "max": float(np.max(z)),
        "q10": float(np.quantile(z, 0.10)),
        "q25": float(np.quantile(z, 0.25)),
        "q50": float(np.quantile(z, 0.50)),
        "q75": float(np.quantile(z, 0.75)),
        "q90": float(np.quantile(z, 0.90)),
        "low_fraction": float(np.mean(z <= 0.25)),
        "high_fraction": float(np.mean(z >= 0.75)),
        "center_norm": float(np.linalg.norm(z - 0.5) / np.sqrt(max(len(z), 1))),
        "diff_mean_abs": float(np.mean(np.abs(diffs))),
    }
    return np.asarray([vals[key] for key in DESCRIPTOR_KEYS], dtype=float), vals


def _json_from_text(text):
    text = str(text or "").strip()
    if not text:
        raise ValueError("empty LLM response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM response did not contain a JSON object")
    return json.loads(match.group(0))


def _clip01(value, default=0.0):
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return float(default)


@dataclass
class LLMRegion:
    name: str
    center: np.ndarray
    radius: float
    weight: float
    rationale: str = ""


class LLMStructuralPriorAdvisor:
    """OpenAI-compatible LLM client plus local evidence gate."""

    def __init__(
        self,
        *,
        base_url,
        model,
        api_key_env="SCOLHKG_LLM_API_KEY",
        timeout_sec=30.0,
        min_obs=8,
        gate_floor=0.05,
        max_observations=24,
        temperature=0.2,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self.api_key_env = str(api_key_env)
        self.timeout_sec = float(timeout_sec)
        self.min_obs = int(min_obs)
        self.gate_floor = float(gate_floor)
        self.max_observations = int(max_observations)
        self.temperature = float(temperature)
        self.last_result = {
            "status": "not_called",
            "gate": 0.0,
            "n_regions": 0,
        }

    def _observation_summary(self, problem, observations):
        rows = []
        for x, ys in list(observations.items())[-max(1, self.max_observations):]:
            y_arr = np.asarray(ys, dtype=float)
            y_mean = np.mean(y_arr, axis=0)
            desc_vec, desc = policy_descriptor(problem, x)
            del desc_vec
            chance_margin = (
                float(y_mean[1])
                + float(norm.ppf(1.0 - problem.alpha))
                * float(getattr(problem, "sigma_level", 0.0))
                - float(problem.tau)
            )
            rows.append({
                "descriptor": {key: round(float(desc[key]), 5) for key in DESCRIPTOR_KEYS},
                "mean_objective": round(float(y_mean[0]), 6),
                "mean_constraint": round(float(y_mean[1]), 6),
                "chance_margin_proxy": round(float(chance_margin), 6),
                "n_replications": int(len(y_arr)),
            })
        return rows

    def _build_prompt(self, problem, observations, iteration, budget_remaining):
        lo, hi = problem.int_bounds()
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        obs_rows = self._observation_summary(problem, observations)
        metadata = {
            "dimension": int(problem.d),
            "integer_bounds_summary": {
                "lo_min": int(np.min(lo)),
                "lo_max": int(np.max(lo)),
                "hi_min": int(np.min(hi)),
                "hi_max": int(np.max(hi)),
                "span_min": int(np.min(hi - lo)),
                "span_max": int(np.max(hi - lo)),
            },
            "risk_level_alpha": float(problem.alpha),
            "constraint_threshold_tau": float(problem.tau),
            "iteration": int(iteration),
            "budget_remaining": int(budget_remaining),
            "descriptor_keys": list(DESCRIPTOR_KEYS),
            "observations": obs_rows,
        }
        system = (
            "You are an evidence-gated structural-prior generator for black-box "
            "chance-constrained optimization. Do not propose final solutions. "
            "Return JSON only. Use only the anonymized metadata and observations. "
            "Do not infer or name benchmark identities. Suggest descriptor-space "
            "candidate regions that may improve feasible search."
        )
        user = {
            "task": "Propose generic normalized descriptor-space candidate regions.",
            "schema": {
                "abstain": "boolean",
                "confidence": "number in [0,1]",
                "candidate_region_priors": [
                    {
                        "name": "short string",
                        "descriptor_center": {
                            key: "number in [0,1]" for key in DESCRIPTOR_KEYS
                        },
                        "radius": "number in [0.02,1.0]",
                        "weight": "number in [0,1]",
                        "rationale": "short non-oracle reason",
                    }
                ],
                "acquisition_weights": {
                    "kg_objective": "number in [0,1]",
                    "kg_feasibility": "number in [0,1]",
                    "kg_variance": "number in [0,1]",
                    "kg_coupling": "number in [0,1]",
                },
            },
            "metadata": metadata,
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, sort_keys=True)},
        ]

    def _call_llm(self, messages):
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(f"missing API key env {self.api_key_env}")
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            },
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _parse_regions(self, payload):
        regions = []
        for idx, item in enumerate(payload.get("candidate_region_priors") or []):
            center_dict = item.get("descriptor_center") or {}
            center = np.asarray([
                _clip01(center_dict.get(key), 0.5)
                for key in DESCRIPTOR_KEYS
            ], dtype=float)
            radius = float(np.clip(float(item.get("radius", 0.25)), 0.02, 1.0))
            weight = _clip01(item.get("weight", 1.0), 1.0)
            regions.append(LLMRegion(
                name=str(item.get("name") or f"region_{idx}")[:64],
                center=center,
                radius=radius,
                weight=weight,
                rationale=str(item.get("rationale") or "")[:240],
            ))
        return regions

    def _observed_utility(self, problem, observations):
        desc = []
        util = []
        z_alpha = float(norm.ppf(1.0 - problem.alpha))
        sigma = float(getattr(problem, "sigma_level", 0.0))
        for x, ys in observations.items():
            y_mean = np.mean(np.asarray(ys, dtype=float), axis=0)
            d_vec, _ = policy_descriptor(problem, x)
            margin = float(y_mean[1]) + z_alpha * sigma - float(problem.tau)
            desc.append(d_vec)
            util.append(float(y_mean[0]) + 5.0 * max(margin, 0.0))
        if not desc:
            return np.empty((0, len(DESCRIPTOR_KEYS))), np.empty(0)
        return np.vstack(desc), np.asarray(util, dtype=float)

    def _evidence_gate(self, problem, observations, regions, confidence, abstain):
        if abstain or not regions:
            return 0.0, {"reason": "abstain_or_no_regions"}
        base = _clip01(confidence, 0.25)
        if len(observations) < max(1, self.min_obs):
            obs_factor = len(observations) / float(max(1, self.min_obs))
            return max(self.gate_floor, base * obs_factor), {
                "reason": "few_observations",
                "obs_factor": float(obs_factor),
            }
        desc, util = self._observed_utility(problem, observations)
        if len(util) == 0:
            return max(self.gate_floor, 0.25 * base), {"reason": "no_utility"}
        baseline = float(np.median(util))
        best_region = float("inf")
        for region in regions:
            d = np.linalg.norm(desc - region.center[None, :], axis=1)
            w = np.exp(-0.5 * (d / max(region.radius, 1e-6)) ** 2)
            if float(np.sum(w)) <= 1e-12:
                continue
            best_region = min(best_region, float(np.sum(w * util) / np.sum(w)))
        if not np.isfinite(best_region):
            return max(self.gate_floor, 0.25 * base), {"reason": "no_region_overlap"}
        scale = float(np.std(util) + 1e-8)
        advantage = (baseline - best_region) / scale
        gate = base / (1.0 + np.exp(-advantage))
        gate = max(self.gate_floor, float(np.clip(gate, 0.0, 1.0)))
        return gate, {
            "reason": "observed_descriptor_evidence",
            "baseline_utility": baseline,
            "best_region_utility": best_region,
            "advantage_z": float(advantage),
        }

    def propose(self, problem, observations, *, iteration, budget_remaining):
        started = time.time()
        try:
            messages = self._build_prompt(problem, observations, iteration, budget_remaining)
            text = self._call_llm(messages)
            payload = _json_from_text(text)
            regions = self._parse_regions(payload)
            confidence = _clip01(payload.get("confidence"), 0.25)
            abstain = bool(payload.get("abstain", False))
            gate, gate_details = self._evidence_gate(
                problem, observations, regions, confidence, abstain)
            self.last_result = {
                "status": "ok",
                "model": self.model,
                "n_regions": int(len(regions)),
                "confidence": float(confidence),
                "abstain": bool(abstain),
                "gate": float(gate),
                "gate_details": gate_details,
                "regions": [
                    {
                        "name": r.name,
                        "radius": float(r.radius),
                        "weight": float(r.weight),
                        "rationale": r.rationale,
                    }
                    for r in regions
                ],
                "time_sec": float(time.time() - started),
            }
            return regions, self.last_result
        except Exception as exc:  # noqa: BLE001 - advisor must not break BO.
            self.last_result = {
                "status": "failed",
                "error": str(exc)[:500],
                "gate": 0.0,
                "n_regions": 0,
                "time_sec": float(time.time() - started),
            }
            return [], self.last_result

    def inverse_candidates(self, problem, regions, *, n, rng, pool_size, gate=1.0):
        n = max(0, int(round(float(n) * float(np.clip(gate, 0.0, 1.0)))))
        if n <= 0 or not regions:
            return []
        rng = rng or np.random.default_rng()
        pool = [problem.sample_random(rng) for _ in range(max(int(pool_size), 4 * n, 16))]
        scored = []
        for x in unique_candidates(pool):
            d_vec, _ = policy_descriptor(problem, x)
            best = float("inf")
            for region in regions:
                dist = float(np.linalg.norm(d_vec - region.center))
                weighted = dist / max(region.radius * max(region.weight, 1e-6), 1e-6)
                best = min(best, weighted)
            scored.append((best, tuple(int(v) for v in x)))
        scored.sort(key=lambda item: item[0])
        return [x for _, x in scored[:n]]
