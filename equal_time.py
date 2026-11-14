"""
تجربة إضافية: مقارنة بنفس الزمن بدل نفس عدد التقييمات
Extra experiment: give GA and PSO the same WALL-CLOCK time that ACO needed
in the main experiment (read from results/summary.csv), instead of the same
number of fitness evaluations. Answers: is ACO better only because each of
its evaluations costs much more time?

    python equal_time.py --runs 5     ->  results/equal_time.csv
(run after run_experiments.py)
"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from problem import ExamProblem  # noqa: E402
import ga, pso  # noqa: E402

ALGS = {"GA": ga, "PSO": pso}
BASE_SEED = 1000


def job(args):
    size, alg, run, seconds = args
    prob = ExamProblem(os.path.join(HERE, "data", size))
    r = ALGS[alg].run(prob, 10**12, np.random.default_rng(BASE_SEED + run), time_limit=seconds)
    print(f"{size:6s} {alg:4s} run {run}  cost={r['cost']:.0f} hard={r['hard']:.0f} "
          f"evals={r['evaluations']}", flush=True)
    return dict(size=size, algorithm=alg, run=run, seconds=seconds, cost=r["cost"],
                hard=r["hard"], evaluations=r["evaluations"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()
    with open(os.path.join(HERE, "results", "summary.csv")) as f:
        aco_time = {r["size"]: float(r["mean_time_s"]) for r in csv.DictReader(f)
                    if r["algorithm"] == "ACO"}
    jobs = [(s, a, r, aco_time[s]) for s in aco_time for a in ALGS for r in range(args.runs)]
    with Pool() as pool:
        out = pool.map(job, jobs, chunksize=1)
    rows = []
    for s in aco_time:
        for a in ALGS:
            rs = [o for o in out if o["size"] == s and o["algorithm"] == a]
            c = np.array([o["cost"] for o in rs])
            rows.append(dict(size=s, algorithm=a, seconds=round(aco_time[s], 1), runs=len(rs),
                             mean_cost=round(c.mean(), 1), std_cost=round(c.std(ddof=1), 1),
                             best_cost=c.min(), feasible_runs=sum(o["hard"] == 0 for o in rs),
                             mean_evaluations=int(np.mean([o["evaluations"] for o in rs]))))
    with open(os.path.join(HERE, "results", "equal_time.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
