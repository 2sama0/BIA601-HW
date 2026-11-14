"""Shared helpers: result record + convergence tracker."""
import time


class Tracker:
    """Keeps the best solution found and the convergence curve
    (iteration -> best cost so far, and evaluations used)."""

    def __init__(self, problem, budget, time_limit=None):
        self.problem = problem
        self.budget = budget
        self.time_limit = time_limit  # optional: stop after this many seconds
        self.start_evals = problem.evaluations
        self.best = None
        self.best_cost = float("inf")
        self.best_hard = None
        self.best_soft = None
        self.history = []          # (iteration, evaluations, best_cost)
        self.first_feasible = None  # evaluations when hard == 0 first reached
        self.t0 = time.perf_counter()

    @property
    def used(self):
        return self.problem.evaluations - self.start_evals

    def done(self):
        if self.time_limit is not None and time.perf_counter() - self.t0 >= self.time_limit:
            return True
        return self.used >= self.budget

    def update(self, S, costs, hard, soft):
        k = int(costs.argmin())
        if costs[k] < self.best_cost:
            self.best_cost = float(costs[k])
            self.best = S[k].copy()
            self.best_hard = float(hard[k])
            self.best_soft = float(soft[k])
        if self.first_feasible is None and (hard == 0).any():
            self.first_feasible = self.used

    def log(self, iteration):
        self.history.append((iteration, self.used, self.best_cost))

    def result(self, name, params):
        return dict(algorithm=name, params=params, best=self.best.tolist(),
                    cost=self.best_cost, hard=self.best_hard, soft=self.best_soft,
                    evaluations=self.used, first_feasible=self.first_feasible,
                    time=time.perf_counter() - self.t0, history=self.history)
