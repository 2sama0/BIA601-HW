"""
اختبار حساسية الـ Hyperparameters (على المسألة المتوسطة)
Small sensitivity study used to choose the default hyperparameters.
Each setting: 3 runs (seeds 0,1,2), same budget as the main experiment.

    python tune.py        ->  results/tuning.csv
"""
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from problem import ExamProblem  # noqa: E402
import ga, pso, aco  # noqa: E402

MODULES = {"GA": ga, "PSO": pso, "ACO": aco}

SIZE, BUDGET, SEEDS = "medium", 20_000, (0, 1, 2)
GRID = [
    ("GA", {}), ("GA", {"pm": 0.5 / 40}), ("GA", {"pm": 2 / 40}),
    ("GA", {"tournament": 2}), ("GA", {"tournament": 5}),
    ("GA", {"pc": 0.6}), ("GA", {"elite": 0.2}), ("GA", {"pop_size": 50}),
    ("PSO", {}), ("PSO", {"w": 1.0}), ("PSO", {"w": 0.5}),
    ("PSO", {"swarm_size": 50}), ("PSO", {"vmax_frac": 0.1}),
    ("PSO", {"c1": 2.0, "c2": 1.0}),
    ("ACO", {}), ("ACO", {"beta": 2.0}), ("ACO", {"rho": 0.3}),
    ("ACO", {"alpha": 2.0}),
]
_prob = None


def job(args):
    global _prob
    name, params, seed = args
    mod = MODULES[name]
    if _prob is None:
        _prob = ExamProblem(os.path.join(HERE, "data", SIZE))
    r = mod.run(_prob, BUDGET, np.random.default_rng(seed), **params)
    return name, str(params or "defaults"), seed, r["cost"]


if __name__ == "__main__":
    jobs = [(n, p, s) for n, p in GRID for s in SEEDS]
    with Pool() as pool:
        out = pool.map(job, jobs, chunksize=1)
    rows = {}
    for name, params, _, cost in out:
        rows.setdefault((name, params), []).append(cost)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "tuning.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "setting", "mean_cost", "min_cost", "max_cost"])
        for (name, params), cs in rows.items():
            w.writerow([name, params, round(np.mean(cs), 1), min(cs), max(cs)])
            print(f"{name:4s} {params:32s} mean={np.mean(cs):9.1f}")
