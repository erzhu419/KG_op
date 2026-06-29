"""
NSGA-II-D: Direct NSGA-II with 10x simulation budget.

No surrogate model. Each objective evaluation is a direct simulation call.
Serves as a computationally expensive gold-standard reference.

Constraint handling follows the constraint-domination principle of:
  Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
  "A fast and elitist multiobjective genetic algorithm: NSGA-II."
  IEEE Transactions on Evolutionary Computation, 6(2), 182-197.
  https://doi.org/10.1109/4235.996017
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpr_kg import pareto_filter, compute_hypervolume_2d
from methods.base_method import BaseMethod
from methods.nsga2_utils import (constrained_nsga2_one_generation,
                                  constrained_environmental_selection)
from experiments.config import HV_EVAL_INTERVAL, NSGA2_PARAMS


class NSGA2Direct(BaseMethod):
    """NSGA-II-D: Direct NSGA-II with 10x budget (gold standard).

    Constraint violation for each individual is defined as:
        CV(x) = max(0, Y[2] - tau)
    where Y[2] is the observed (noisy) constraint value.  The EA uses
    constraint-domination selection (Deb et al. 2002) throughout.
    """

    name = "NSGA-II-D"

    def __init__(self, budget_multiplier=10):
        self.budget_multiplier = budget_multiplier

    def run(self, problem, N, n0, seed, hv_eval_interval=HV_EVAL_INTERVAL):
        ref = problem.ref_point
        if seed is not None:
            np.random.seed(seed)

        t_start = time.time()
        d = problem.d
        lo, hi = problem.int_bounds()
        L = int(hi[0])
        total_budget = N * self.budget_multiplier  # 10x budget
        pop_size = NSGA2_PARAMS['pop_size']

        # Initialize random population
        population = np.array([list(problem.sample_random()) for _ in range(pop_size)])

        # Evaluate initial population (each costs 1 simulation)
        all_sims = {}  # x_tuple -> Y (most recent)
        objectives_3 = np.zeros((pop_size, 3))
        n_sims = 0

        for i in range(pop_size):
            x = tuple(population[i])
            Y = problem.simulate(population[i])
            all_sims[x] = Y
            objectives_3[i] = Y
            n_sims += 1

        # f1, f2 used as objectives; f3 constraint violation for sorting
        objectives = objectives_3[:, :2]
        # CV(x) = max(0, observed_f3 - tau)  [Deb et al. 2002, constraint-domination]
        cv_pop = np.maximum(objectives_3[:, 2] - problem.tau, 0.0)

        hv_history = []
        time_per_iter = []

        # Main EA loop
        while n_sims < total_budget:
            t_gen = time.time()

            # Generate offspring using constrained NSGA-II selection
            # (Deb et al. 2002, Section III-B)
            offspring = constrained_nsga2_one_generation(
                population, objectives, cv_pop, 1, L, pop_size,
                crossover_eta=NSGA2_PARAMS['crossover_eta'],
                mutation_eta=NSGA2_PARAMS['mutation_eta'],
                crossover_prob=NSGA2_PARAMS['crossover_prob'],
            )

            # Evaluate offspring by direct simulation
            offspring_obj_3 = np.zeros((len(offspring), 3))
            actual_offspring_len = len(offspring)
            for i in range(len(offspring)):
                if n_sims >= total_budget:
                    actual_offspring_len = i
                    break
                x = tuple(offspring[i])
                Y = problem.simulate(offspring[i])
                all_sims[x] = Y
                offspring_obj_3[i] = Y
                n_sims += 1

            if actual_offspring_len == 0:
                break

            offspring = offspring[:actual_offspring_len]
            offspring_obj_3 = offspring_obj_3[:actual_offspring_len]
            offspring_obj = offspring_obj_3[:, :2]
            cv_offspring = np.maximum(offspring_obj_3[:, 2] - problem.tau, 0.0)

            # Environmental selection with constraint-domination (Deb et al. 2002)
            combined_pop = np.vstack([population, offspring])
            combined_obj = np.vstack([objectives, offspring_obj])
            combined_cv = np.concatenate([cv_pop, cv_offspring])
            population, objectives, cv_pop = constrained_environmental_selection(
                combined_pop, combined_obj, combined_cv, pop_size)

            t_compute = time.time() - t_gen
            time_per_iter.append(t_compute)

            # HV evaluation at intervals (based on simulation budget)
            if n_sims % (hv_eval_interval * self.budget_multiplier) == 0 \
                    or n_sims >= total_budget:
                pf_objs, pf_sols = self._current_pareto(all_sims, problem)
                hv = compute_hypervolume_2d(pf_objs, ref) \
                    if len(pf_objs) > 0 else 0.0
                # Scale x-axis to match other methods' budget
                equiv_budget = n_sims / self.budget_multiplier
                hv_history.append((equiv_budget, float(hv)))

        total_time = time.time() - t_start

        # Final Pareto front from all evaluated solutions
        pf_objs, pf_sols = self._current_pareto(all_sims, problem)

        return self._make_result(
            pareto_solutions=pf_sols,
            problem=problem,
            hv_history=hv_history,
            total_time=total_time,
            time_per_iter=time_per_iter,
            n_simulations=n_sims,
            ref_point=ref,
        )

    def _current_pareto(self, all_sims, problem):
        """Extract feasible Pareto front from all simulated solutions.

        Feasibility is determined by the observed (noisy) constraint value
        Y[2] <= tau.  This is consistent with direct NSGA-II having no
        probabilistic constraint model.
        """
        feasible_objs = []
        feasible_sols = []

        for x_tuple, Y in all_sims.items():
            if Y[2] <= problem.tau:
                feasible_objs.append([Y[0], Y[1]])
                feasible_sols.append(x_tuple)

        if len(feasible_objs) == 0:
            return np.empty((0, 2)), []

        objs = np.array(feasible_objs)
        pf, pf_idx = pareto_filter(objs, return_indices=True)
        return pf, [feasible_sols[i] for i in pf_idx]
