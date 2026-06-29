"""
NSGA-II core utilities for discrete optimization.

Implements the essential NSGA-II components:
- Non-dominated sorting
- Crowding distance assignment
- Tournament selection
- Integer SBX crossover
- Polynomial mutation for integers
"""

import numpy as np


def non_dominated_sort(objectives):
    """Fast non-dominated sorting (Deb et al. 2002).

    Args:
        objectives: np.array of shape (N, M), each row is an objective vector (minimization).

    Returns:
        list of lists: fronts[0] is the first front (indices), fronts[1] second, etc.
    """
    N = len(objectives)
    domination_count = np.zeros(N, dtype=int)
    dominated_set = [[] for _ in range(N)]
    fronts = [[]]

    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if _dominates(objectives[p], objectives[q]):
                dominated_set[p].append(q)
            elif _dominates(objectives[q], objectives[p]):
                domination_count[p] += 1

        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]  # Remove last empty front


def _dominates(a, b):
    """Check if a dominates b (all <= and at least one <)."""
    return np.all(a <= b) and np.any(a < b)


def crowding_distance(objectives, front_indices):
    """Compute crowding distances for solutions in a front.

    Args:
        objectives: np.array of shape (N, M).
        front_indices: list of indices in this front.

    Returns:
        dict: index -> crowding distance.
    """
    n = len(front_indices)
    if n <= 2:
        return {idx: float('inf') for idx in front_indices}

    M = objectives.shape[1]
    distances = {idx: 0.0 for idx in front_indices}

    for m in range(M):
        sorted_idx = sorted(front_indices, key=lambda i: objectives[i, m])
        distances[sorted_idx[0]] = float('inf')
        distances[sorted_idx[-1]] = float('inf')

        obj_range = objectives[sorted_idx[-1], m] - objectives[sorted_idx[0], m]
        if obj_range < 1e-15:
            continue

        for k in range(1, n - 1):
            distances[sorted_idx[k]] += (
                objectives[sorted_idx[k + 1], m] - objectives[sorted_idx[k - 1], m]
            ) / obj_range

    return distances


def tournament_selection(population, objectives, fronts, crowd_dist, n_select):
    """Binary tournament selection based on rank and crowding distance.

    Args:
        population: np.array of shape (N, d) - integer decision vectors.
        objectives: np.array of shape (N, M).
        fronts: list of fronts from non_dominated_sort.
        crowd_dist: dict index -> crowding distance.
        n_select: number of parents to select.

    Returns:
        np.array of shape (n_select, d) - selected parents.
    """
    N = len(population)
    rank = np.zeros(N, dtype=int)
    for r, front in enumerate(fronts):
        for idx in front:
            rank[idx] = r

    selected = []
    for _ in range(n_select):
        i, j = np.random.randint(N, size=2)
        if rank[i] < rank[j]:
            selected.append(i)
        elif rank[j] < rank[i]:
            selected.append(j)
        elif crowd_dist.get(i, 0) > crowd_dist.get(j, 0):
            selected.append(i)
        else:
            selected.append(j)

    return population[selected]


def integer_sbx_crossover(parent1, parent2, L_min, L_max, eta=20, prob=0.9):
    """SBX crossover adapted for integer variables.

    Args:
        parent1, parent2: 1D integer arrays.
        L_min, L_max: bounds for each dimension.
        eta: distribution index (higher = more like parents).
        prob: crossover probability.

    Returns:
        child1, child2: integer arrays.
    """
    d = len(parent1)
    child1 = parent1.copy().astype(float)
    child2 = parent2.copy().astype(float)

    if np.random.rand() > prob:
        return np.round(child1).astype(int), np.round(child2).astype(int)

    for i in range(d):
        if np.random.rand() > 0.5:
            continue
        if abs(parent1[i] - parent2[i]) < 1e-14:
            continue

        p1, p2 = float(min(parent1[i], parent2[i])), float(max(parent1[i], parent2[i]))
        delta = p2 - p1

        u = np.random.rand()
        if u <= 0.5:
            beta = (2.0 * u) ** (1.0 / (eta + 1))
        else:
            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1))

        # Use sorted parents (y1 <= y2) per Deb & Agrawal (1995) SBX definition
        child1[i] = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        child2[i] = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)

    child1 = np.clip(np.round(child1), L_min, L_max).astype(int)
    child2 = np.clip(np.round(child2), L_min, L_max).astype(int)
    return child1, child2


def polynomial_mutation(individual, L_min, L_max, eta=20, prob=None):
    """Polynomial mutation adapted for integer variables.

    Args:
        individual: 1D integer array.
        L_min, L_max: bounds.
        eta: distribution index.
        prob: mutation probability per gene (default: 1/d).

    Returns:
        mutated individual (integer array).
    """
    d = len(individual)
    if prob is None:
        prob = 1.0 / d

    child = individual.copy().astype(float)
    for i in range(d):
        if np.random.rand() > prob:
            continue

        delta_max = float(L_max - L_min)
        if delta_max < 1e-14:
            continue

        u = np.random.rand()
        if u < 0.5:
            delta = (2.0 * u) ** (1.0 / (eta + 1)) - 1.0
        else:
            delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1))

        child[i] = child[i] + delta * delta_max

    return np.clip(np.round(child), L_min, L_max).astype(int)


def nsga2_one_generation(population, objectives, L_min, L_max, pop_size,
                          crossover_eta=20, mutation_eta=20, crossover_prob=0.9):
    """Run one NSGA-II generation: selection, crossover, mutation, environmental selection.

    Args:
        population: np.array (N, d) current population (integers).
        objectives: np.array (N, M) objectives for current population.
        L_min, L_max: integer bounds.
        pop_size: target population size after selection.
        crossover_eta, mutation_eta: distribution indices.
        crossover_prob: crossover probability.

    Returns:
        offspring: np.array (pop_size, d) new population (needs evaluation).
    """
    N = len(population)
    fronts = non_dominated_sort(objectives)
    cd = {}
    for front in fronts:
        cd.update(crowding_distance(objectives, front))

    # Selection
    parents = tournament_selection(population, objectives, fronts, cd, pop_size)

    # Crossover + mutation
    offspring = []
    for k in range(0, pop_size, 2):
        p1 = parents[k]
        p2 = parents[min(k + 1, pop_size - 1)]
        c1, c2 = integer_sbx_crossover(p1, p2, L_min, L_max,
                                         eta=crossover_eta, prob=crossover_prob)
        c1 = polynomial_mutation(c1, L_min, L_max, eta=mutation_eta)
        c2 = polynomial_mutation(c2, L_min, L_max, eta=mutation_eta)
        offspring.append(c1)
        if len(offspring) < pop_size:
            offspring.append(c2)

    return np.array(offspring[:pop_size])


def environmental_selection(combined_pop, combined_obj, pop_size):
    """Environmental selection: keep best pop_size from combined population.

    Uses non-dominated sorting + crowding distance.

    Args:
        combined_pop: np.array (2*N, d).
        combined_obj: np.array (2*N, M).
        pop_size: target size.

    Returns:
        selected_pop, selected_obj: arrays of size pop_size.
    """
    fronts = non_dominated_sort(combined_obj)
    selected = []

    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
        else:
            remaining = pop_size - len(selected)
            cd = crowding_distance(combined_obj, front)
            sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
            selected.extend(sorted_front[:remaining])
            break

    selected = np.array(selected)
    return combined_pop[selected], combined_obj[selected]


# =============================================================================
# Constrained NSGA-II utilities
# Reference: Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
#   "A fast and elitist multiobjective genetic algorithm: NSGA-II."
#   IEEE Transactions on Evolutionary Computation, 6(2), 182-197.
#   https://doi.org/10.1109/4235.996017
#   (Section III-B defines the constraint-domination principle)
#
# Open-source reference implementation: pymoo (Blank & Deb, 2020)
#   https://github.com/anyoptimization/pymoo
#   https://pymoo.org/constraints/feas_first.html
# =============================================================================

def _constraint_dominates(obj_a, cv_a, obj_b, cv_b):
    """Return True iff solution a constraint-dominates solution b (Deb et al. 2002).

    The three-case constraint-domination rule (Table I in Deb et al. 2002):
      1. a is feasible (CV=0) and b is infeasible → a dominates b.
      2. Both infeasible: smaller total constraint violation wins.
      3. Both feasible: standard Pareto dominance on objectives.

    Args:
        obj_a, obj_b: 1-D objective arrays (minimisation).
        cv_a, cv_b: scalar total constraint violation (>= 0; 0 = feasible).
    """
    feas_a = cv_a <= 1e-12
    feas_b = cv_b <= 1e-12
    if feas_a and not feas_b:
        return True
    if not feas_a and not feas_b:
        return cv_a < cv_b - 1e-12
    if feas_a and feas_b:
        return _dominates(obj_a, obj_b)
    return False


def constrained_non_dominated_sort(objectives, constraint_violations):
    """Constrained non-dominated sorting (Deb et al. 2002, NSGA-II).

    Uses the constraint-domination principle instead of Pareto dominance,
    so infeasible solutions are ranked behind all feasible ones.

    Args:
        objectives: np.array (N, M).
        constraint_violations: np.array (N,) — total CV >= 0; 0 = feasible.

    Returns:
        List of fronts (each front is a list of indices, best front first).
    """
    N = len(objectives)
    domination_count = np.zeros(N, dtype=int)
    dominated_set = [[] for _ in range(N)]
    fronts = [[]]

    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if _constraint_dominates(objectives[p], constraint_violations[p],
                                      objectives[q], constraint_violations[q]):
                dominated_set[p].append(q)
            elif _constraint_dominates(objectives[q], constraint_violations[q],
                                        objectives[p], constraint_violations[p]):
                domination_count[p] += 1

        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]


def constrained_tournament_selection(population, objectives, fronts, crowd_dist,
                                      n_select):
    """Binary tournament using constrained rank + crowding distance.

    Args:
        population: np.array (N, d).
        objectives: np.array (N, M).
        fronts: result of constrained_non_dominated_sort.
        crowd_dist: dict index -> crowding distance.
        n_select: number of parents to select.

    Returns:
        np.array (n_select, d) selected parents.
    """
    N = len(population)
    rank = np.zeros(N, dtype=int)
    for r, front in enumerate(fronts):
        for idx in front:
            rank[idx] = r

    selected = []
    for _ in range(n_select):
        i, j = np.random.randint(N, size=2)
        if rank[i] < rank[j]:
            selected.append(i)
        elif rank[j] < rank[i]:
            selected.append(j)
        elif crowd_dist.get(i, 0) > crowd_dist.get(j, 0):
            selected.append(i)
        else:
            selected.append(j)

    return population[selected]


def constrained_environmental_selection(combined_pop, combined_obj, combined_cv,
                                         pop_size):
    """Environmental selection with constraint-domination (Deb et al. 2002).

    Args:
        combined_pop: np.array (2N, d).
        combined_obj: np.array (2N, M).
        combined_cv: np.array (2N,) total constraint violations (>= 0).
        pop_size: target population size.

    Returns:
        selected_pop, selected_obj, selected_cv (each of size pop_size).
    """
    fronts = constrained_non_dominated_sort(combined_obj, combined_cv)
    selected = []

    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
        else:
            remaining = pop_size - len(selected)
            cd = crowding_distance(combined_obj, front)
            sorted_front = sorted(front, key=lambda idx: cd[idx], reverse=True)
            selected.extend(sorted_front[:remaining])
            break

    selected = np.array(selected)
    return combined_pop[selected], combined_obj[selected], combined_cv[selected]


def constrained_nsga2_one_generation(population, objectives, constraint_violations,
                                      L_min, L_max, pop_size,
                                      crossover_eta=20, mutation_eta=20,
                                      crossover_prob=0.9):
    """One NSGA-II generation with constraint-domination selection (Deb et al. 2002).

    Replaces the unconstrained non_dominated_sort with constrained_non_dominated_sort
    so infeasible individuals are pushed behind feasible ones during selection.

    Args:
        population: np.array (N, d) integer decision vectors.
        objectives: np.array (N, M) objective values.
        constraint_violations: np.array (N,) total CV per individual (>= 0).
        L_min, L_max: integer bounds.
        pop_size: offspring count to produce.
        crossover_eta, mutation_eta: SBX / polynomial distribution indices
            (Deb & Agrawal 1995; Deb & Goyal 1996).
        crossover_prob: per-individual SBX activation probability.

    Returns:
        offspring: np.array (pop_size, d) new individuals (needs evaluation).
    """
    fronts = constrained_non_dominated_sort(objectives, constraint_violations)
    cd = {}
    for front in fronts:
        cd.update(crowding_distance(objectives, front))

    parents = constrained_tournament_selection(
        population, objectives, fronts, cd, pop_size)

    offspring = []
    for k in range(0, pop_size, 2):
        p1 = parents[k]
        p2 = parents[min(k + 1, pop_size - 1)]
        c1, c2 = integer_sbx_crossover(p1, p2, L_min, L_max,
                                        eta=crossover_eta, prob=crossover_prob)
        c1 = polynomial_mutation(c1, L_min, L_max, eta=mutation_eta)
        c2 = polynomial_mutation(c2, L_min, L_max, eta=mutation_eta)
        offspring.append(c1)
        if len(offspring) < pop_size:
            offspring.append(c2)

    return np.array(offspring[:pop_size])
