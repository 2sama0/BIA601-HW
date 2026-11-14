"""
Genetic Algorithm (Chapter 3) for exam timetabling.

  Encoding     : Value encoding, chromosome = slot of each exam.
  Initialization: random population.
  Fitness      : cost from problem.py (minimise).
  Selection    : Tournament selection (as in the course TSP GA, tournament_size)
                 + Elitism (as in genetic_optimize: the best 'elite' fraction survives).
  Crossover    : 1-point crossover with probability pc (course: 0.6..0.9).
  Mutation     : each gene changes to a random slot with probability pm
                 (course: between 1/pop_size and 1/chromosome_length).
  Termination  : evaluation budget (number of generations).
"""
import numpy as np

from common import Tracker

DEFAULTS = dict(pop_size=100, elite=0.1, pc=0.8, pm=None, tournament=3)


def run(problem, budget, rng, **params):
    p = {**DEFAULTS, **params}
    E, T, N = problem.E, problem.T, p["pop_size"]
    pm = p["pm"] if p["pm"] is not None else 1.0 / E
    n_elite = max(1, int(p["elite"] * N))
    tr = Tracker(problem, budget, p.pop("time_limit", None))

    pop = problem.random_solutions(N, rng)
    cost, hard, soft = problem.evaluate_many(pop)
    tr.update(pop, cost, hard, soft)
    tr.log(0)

    gen = 0
    while not tr.done():
        gen += 1
        order = np.argsort(cost)
        elite = pop[order[:n_elite]]

        n_child = min(N - n_elite, budget - tr.used)
        # --- selection (tournament) ---
        def select(k):
            cand = rng.integers(0, N, size=(k, p["tournament"]))
            winners = cand[np.arange(k), cost[cand].argmin(1)]
            return pop[winners]
        p1, p2 = select(n_child), select(n_child)

        # --- 1-point crossover ---
        children = p1.copy()
        do_x = rng.random(n_child) < p["pc"]
        cut = rng.integers(1, E, size=n_child)
        mask = (np.arange(E)[None, :] >= cut[:, None]) & do_x[:, None]
        children[mask] = p2[mask]

        # --- mutation ---
        mut = rng.random(children.shape) < pm
        children[mut] = rng.integers(0, T, size=mut.sum())

        c_cost, c_hard, c_soft = problem.evaluate_many(children)
        tr.update(children, c_cost, c_hard, c_soft)

        # --- new generation = elite + children ---
        e_idx = order[:n_elite]
        pop = np.vstack([elite, children])
        cost = np.concatenate([cost[e_idx], c_cost])
        hard = np.concatenate([hard[e_idx], c_hard])
        soft = np.concatenate([soft[e_idx], c_soft])
        if len(pop) < N:  # last (partial) generation only
            break
        tr.log(gen)
    tr.log(gen)
    return tr.result("GA", {**p, "pm": pm})
