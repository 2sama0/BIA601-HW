"""
تشغيل التجارب والمقارنة بين الخوارزميات
Runs every algorithm on every problem size several times (different seeds),
with the SAME budget of fitness evaluations, then writes tables and charts.

    python run_experiments.py                 # full experiment (10 runs)
    python run_experiments.py --runs 3 --sizes small medium   # quick check

Outputs (results/):
    raw_results.json      every run (best solution, cost, time, convergence curve)
    summary.csv           mean / std / best / worst per algorithm and size
    memory.csv            peak memory per algorithm and size (tracemalloc)
    convergence_<size>.png, boxplot.png, time.png
    best_timetable_<size>.txt   best timetable found, readable
"""
import argparse
import csv
import json
import os
import sys
import time
import tracemalloc
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from problem import ExamProblem  # noqa: E402
import random_search, ga, pso, aco  # noqa: E402

ALGORITHMS = {"Random": random_search, "GA": ga, "PSO": pso, "ACO": aco}
BUDGET = {"small": 10_000, "medium": 20_000, "large": 30_000}  # fitness evaluations
BASE_SEED = 1000
RESULTS = os.path.join(HERE, "results")

_problems = {}


def get_problem(size):
    if size not in _problems:
        _problems[size] = ExamProblem(os.path.join(HERE, "data", size))
    return _problems[size]


def one_run(job):
    size, alg, run = job
    prob = get_problem(size)
    rng = np.random.default_rng(BASE_SEED + run)  # same seeds for every algorithm
    res = ALGORITHMS[alg].run(prob, BUDGET[size], rng)
    res.update(size=size, run=run, seed=BASE_SEED + run)
    print(f"{size:6s} {alg:6s} run {run:2d}  cost={res['cost']:9.0f}  hard={res['hard']:.0f}  "
          f"time={res['time']:6.2f}s", flush=True)
    return res


def memory_run(job):
    size, alg = job
    prob = get_problem(size)
    tracemalloc.start()
    ALGORITHMS[alg].run(prob, 2_000, np.random.default_rng(0))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dict(size=size, algorithm=alg, peak_kb=peak / 1024)


# ---------------------------------------------------------------- analysis
def evals_to_reach(history, target):
    for _, ev, best in history:
        if best <= target:
            return ev
    return None


def summarise(results, sizes):
    rows = []
    for size in sizes:
        for alg in ALGORITHMS:
            rs = [r for r in results if r["size"] == size and r["algorithm"] == alg]
            if not rs:
                continue
            c = np.array([r["cost"] for r in rs])
            feas = [r for r in rs if r["hard"] == 0]
            ff = [r["first_feasible"] for r in rs if r["first_feasible"] is not None]
            # convergence speed: evaluations needed to get within 5% of the run's final result
            conv = [evals_to_reach(r["history"], r["cost"] * 1.05) for r in rs]
            rows.append(dict(
                size=size, algorithm=alg, runs=len(rs),
                mean_cost=c.mean(), std_cost=c.std(ddof=1) if len(c) > 1 else 0.0,
                best_cost=c.min(), worst_cost=c.max(),
                cv_percent=100 * c.std(ddof=1) / c.mean() if len(c) > 1 else 0.0,
                feasible_runs=len(feas),
                mean_soft_feasible=np.mean([r["soft"] for r in feas]) if feas else None,
                mean_hard=np.mean([r["hard"] for r in rs]),
                mean_first_feasible_evals=np.mean(ff) if ff else None,
                mean_evals_to_95pct=np.mean(conv),
                mean_time_s=np.mean([r["time"] for r in rs]),
                budget=BUDGET[size]))
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})


def write_timetable(prob, res, path):
    with open(path, "w") as f:
        f.write(f"Best timetable for '{prob.name}' by {res['algorithm']} (run {res['run']})\n")
        f.write(f"cost={res['cost']:.0f}  hard violations={res['hard']:.0f}  "
                f"soft penalty={res['soft']:.0f}\n\n")
        s = np.array(res["best"])
        load = np.bincount(s, weights=prob.size, minlength=prob.T)
        for day, period, exams in prob.describe(s):
            t = (day - 1) * prob.P + (period - 1)
            names = ", ".join(f"Course_{e:03d}" for e in exams) or "-"
            f.write(f"Day {day:2d}  Period {period}  seats {int(load[t]):4d}/{prob.seats}:  {names}\n")


# ---------------------------------------------------------------- charts
COLORS = {"Random": "#8a8a85", "GA": "#2a78d6", "PSO": "#eb6834", "ACO": "#1baf7a"}
STYLES = {"Random": ":", "GA": "-", "PSO": "--", "ACO": "-."}


def plot_all(results, sizes, mem_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e3", "grid.linewidth": 0.8,
                         "axes.edgecolor": "#9a9a95"})

    for size in sizes:
        # --- convergence: iteration -> best fitness (mean over runs) ---
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        grid = np.linspace(0, BUDGET[size], 300)
        for alg in ALGORITHMS:
            rs = [r for r in results if r["size"] == size and r["algorithm"] == alg]
            if not rs:
                continue
            curves = []
            for r in rs:
                h = np.array(r["history"])
                curves.append(np.interp(grid, h[:, 1], h[:, 2], left=h[0, 2]))
            curves = np.array(curves)
            med = np.median(curves, 0)
            ax.plot(grid, med, STYLES[alg], color=COLORS[alg], lw=2, label=alg)
            ax.fill_between(grid, np.percentile(curves, 25, 0), np.percentile(curves, 75, 0),
                            color=COLORS[alg], alpha=0.12, lw=0)
            ax.annotate(f"{alg} {med[-1]:,.0f}", (grid[-1], med[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color="#333")
        ax.set_yscale("log")
        ax.set_xlabel("Fitness evaluations (iteration × population size)")
        ax.set_ylabel("Best cost so far (log scale, lower is better)")
        ax.set_title(f"Convergence – {size} (median of runs, band = 25–75%)", loc="left")
        ax.legend(frameon=False, ncol=4, loc="upper right")
        ax.set_xlim(0, BUDGET[size] * 1.18)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, f"convergence_{size}.png"), dpi=150)
        plt.close(fig)

    # --- boxplots of final cost (stability), one panel per size ---
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.2 * len(sizes) + 0.8, 3.8), squeeze=False)
    for ax, size in zip(axes[0], sizes):
        algs = [a for a in ALGORITHMS if any(r["size"] == size and r["algorithm"] == a for r in results)]
        data = [[r["cost"] for r in results if r["size"] == size and r["algorithm"] == a] for a in algs]
        bp = ax.boxplot(data, tick_labels=algs, patch_artist=True, widths=0.55,
                        medianprops=dict(color="#222", lw=1.5))
        for patch, a in zip(bp["boxes"], algs):
            patch.set_facecolor(COLORS[a]); patch.set_alpha(0.55); patch.set_edgecolor(COLORS[a])
        ax.set_yscale("log")
        ax.set_title(size, loc="left")
    axes[0][0].set_ylabel("Final cost (log scale)")
    fig.suptitle("Final cost over 10 repeated runs (stability)", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "boxplot.png"), dpi=150)
    plt.close(fig)

    # --- execution time ---
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    algs = list(ALGORITHMS)
    x = np.arange(len(sizes))
    wdt = 0.8 / len(algs)
    for k, alg in enumerate(algs):
        vals = [np.mean([r["time"] for r in results if r["size"] == s and r["algorithm"] == alg] or [0])
                for s in sizes]
        bars = ax.bar(x + (k - (len(algs) - 1) / 2) * wdt, vals, wdt * 0.92,
                      color=COLORS[alg], label=alg)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}" if v >= 1 else f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=7, color="#333")
    ax.set_xticks(x, sizes)
    ax.set_yscale("log")
    ax.set_ylabel("Mean time per run (s, log scale)")
    ax.set_title("Execution time (same evaluation budget)", loc="left")
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "time.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--sizes", nargs="+", default=list(BUDGET))
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    jobs = [(s, a, r) for s in args.sizes for a in ALGORITHMS for r in range(args.runs)]
    jobs.sort(key=lambda j: j[1] != "ACO")  # start the slowest first
    t0 = time.time()
    with Pool(args.jobs) as pool:
        results = pool.map(one_run, jobs, chunksize=1)
        mem_rows = pool.map(memory_run, [(s, a) for s in args.sizes for a in ALGORITHMS])
    print(f"total wall time {time.time() - t0:.0f}s")

    with open(os.path.join(RESULTS, "raw_results.json"), "w") as f:
        json.dump(results, f)
    rows = summarise(results, args.sizes)
    write_csv(os.path.join(RESULTS, "summary.csv"), rows)
    write_csv(os.path.join(RESULTS, "memory.csv"), mem_rows)
    for size in args.sizes:
        best = min((r for r in results if r["size"] == size), key=lambda r: r["cost"])
        write_timetable(get_problem(size), best, os.path.join(RESULTS, f"best_timetable_{size}.txt"))
    plot_all(results, args.sizes, mem_rows)

    for r in rows:
        print(f"{r['size']:6s} {r['algorithm']:6s} mean={r['mean_cost']:10.1f} std={r['std_cost']:9.1f} "
              f"best={r['best_cost']:8.0f} feasible={r['feasible_runs']}/{r['runs']} "
              f"time={r['mean_time_s']:.2f}s")


if __name__ == "__main__":
    main()
