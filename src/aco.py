"""
Ant Colony Optimization (Chapter 5) for exam timetabling.

  Construct solutions : each ant schedules the exams one by one (hardest first:
                        most conflicts first). For exam i it chooses slot t with
                        probability  ~  tau[i,t]^alpha * eta[i,t]^beta
                        eta = 1 / (1 + extra cost of putting exam i in slot t,
                                       given the exams already placed)   (heuristic)
  Pheromone update    : evaporation  tau = (1 - rho) * tau
                        deposit      tau[i, s_i] += Q / cost    (ant-cycle, each ant)
  Parameters follow the course ACO(ant_count, generations, alpha, beta, rho, q).
  Termination         : evaluation budget (each built ant = one evaluation).
"""
import numpy as np

from common import Tracker
from problem import HARD_W, W_CONSECUTIVE, W_SAME_DAY

DEFAULTS = dict(ant_count=20, alpha=1.0, beta=4.0, rho=0.1, q=None, tau0=1.0)


def _construct(problem, tau, order, alpha, beta, rng):
    E, T, D, P = problem.E, problem.T, problem.D, problem.P
    C, size, seats = problem.C, problem.size, problem.seats
    sol = np.full(E, -1, dtype=np.int64)
    load = np.zeros(T)
    placed = []
    for i in order:
        # students of exam i already examined in each slot
        x = np.zeros(T)
        if placed:
            idx = np.array(placed)
            np.add.at(x, sol[idx], C[i, idx])
        xd = x.reshape(D, P)
        day_total = xd.sum(1, keepdims=True)
        left = np.zeros_like(xd);  left[:, 1:] = xd[:, :-1]
        right = np.zeros_like(xd); right[:, :-1] = xd[:, 1:]
        cons = left + right
        other = day_total - xd - cons
        overflow = np.clip(load + size[i] - seats, 0, None) - np.clip(load - seats, 0, None)
        delta = (HARD_W * (xd + overflow.reshape(D, P))
                 + W_CONSECUTIVE * cons + W_SAME_DAY * other).ravel()
        eta = 1.0 / (1.0 + delta)
        w = tau[i] ** alpha * eta ** beta
        w_sum = w.sum()
        t = rng.choice(T, p=w / w_sum) if w_sum > 0 else rng.integers(T)
        sol[i] = t
        load[t] += size[i]
        placed.append(i)
    return sol


def run(problem, budget, rng, **params):
    p = {**DEFAULTS, **params}
    E, T, M = problem.E, problem.T, p["ant_count"]
    tr = Tracker(problem, budget, p.pop("time_limit", None))
    tau = np.full((E, T), p["tau0"])
    order = np.argsort(-problem.degree, kind="stable")  # most constrained exam first
    q = p["q"]

    it = 0
    while not tr.done():
        n = min(M, budget - tr.used)
        ants = np.array([_construct(problem, tau, order, p["alpha"], p["beta"], rng)
                         for _ in range(n)])
        cost, hard, soft = problem.evaluate_many(ants)
        tr.update(ants, cost, hard, soft)
        if q is None:  # scale Q to the problem: first iteration's best cost
            q = float(cost.min())
        # evaporation + deposit
        tau *= (1 - p["rho"])
        for k in range(n):
            tau[np.arange(E), ants[k]] += q / max(cost[k], 1.0)
        tau = np.maximum(tau, 1e-6)
        tr.log(it)
        it += 1
    return tr.result("ACO", {**p, "q": q})
