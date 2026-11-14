"""
Baseline: Random Optimizer (as in Chapter 3, 'randomoptimize').
Generates random timetables and keeps the best one.
It shows what the intelligent algorithms must beat.
"""
from common import Tracker

DEFAULTS = dict(batch=100)


def run(problem, budget, rng, **params):
    p = {**DEFAULTS, **params}
    tr = Tracker(problem, budget, p.pop("time_limit", None))
    it = 0
    while not tr.done():
        n = min(p["batch"], budget - tr.used)
        S = problem.random_solutions(n, rng)
        tr.update(S, *problem.evaluate_many(S))
        tr.log(it)
        it += 1
    return tr.result("Random", p)
