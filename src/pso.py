"""
Particle Swarm Optimization (Chapter 4) for exam timetabling.

  Particle position : real vector x in [0, T)^E ; decoded slot = floor(x_i)
  Velocity update   : v = w*v + c1*rand1*(pbest - x) + c2*rand2*(gbest - x)
                      (the course formula is the case w = 1; w < 1 is the usual
                       'inertia weight' that stops velocities from exploding)
  Position update   : x = x + v      (clipped to the search space)
  pbest / gbest     : best position of each particle / of the whole swarm.
  Termination       : evaluation budget.
  No crossover / mutation (difference from GA, see course section 1.3).
"""
import numpy as np

from common import Tracker

DEFAULTS = dict(swarm_size=100, w=0.7, c1=1.5, c2=1.5, vmax_frac=0.25)


def run(problem, budget, rng, **params):
    p = {**DEFAULTS, **params}
    E, T, N = problem.E, problem.T, p["swarm_size"]
    vmax = p["vmax_frac"] * T
    tr = Tracker(problem, budget, p.pop("time_limit", None))

    def decode(X):
        return np.minimum(X.astype(np.int64), T - 1)

    x = rng.uniform(0, T, size=(N, E))
    v = rng.uniform(-vmax, vmax, size=(N, E))
    cost, hard, soft = problem.evaluate_many(decode(x))
    tr.update(decode(x), cost, hard, soft)
    pbest, pbest_cost = x.copy(), cost.copy()
    g = pbest_cost.argmin()
    gbest, gbest_cost = pbest[g].copy(), pbest_cost[g]
    tr.log(0)

    it = 0
    while not tr.done():
        it += 1
        n = min(N, budget - tr.used)
        r1, r2 = rng.random((N, E)), rng.random((N, E))
        v = p["w"] * v + p["c1"] * r1 * (pbest - x) + p["c2"] * r2 * (gbest - x)
        v = np.clip(v, -vmax, vmax)
        x = np.clip(x + v, 0, T - 1e-9)

        cost, hard, soft = problem.evaluate_many(decode(x[:n]))
        tr.update(decode(x[:n]), cost, hard, soft)
        better = cost < pbest_cost[:n]
        pbest[:n][better] = x[:n][better]
        pbest_cost[:n][better] = cost[better]
        g = pbest_cost.argmin()
        if pbest_cost[g] < gbest_cost:
            gbest, gbest_cost = pbest[g].copy(), pbest_cost[g]
        tr.log(it)
    return tr.result("PSO", p)
