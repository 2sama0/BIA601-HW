"""
تعريف المسألة ودالة الهدف (Objective / Cost function)

Solution representation (Value Encoding, as in Chapter 3):
    s = [s_0, s_1, ..., s_{E-1}],  s_i in {0..T-1}  = time-slot of exam i
    slot t  ->  day = t // P,  period = t % P

Hard constraints (must be 0 for a feasible timetable):
    H1  clash:     a student has two exams in the same slot
    H2  capacity:  students seated in a slot > seats_per_slot

Soft constraints (student comfort, minimise):
    S1  two exams in consecutive periods of the same day   (weight 3 per student)
    S2  two exams in the same day, not consecutive         (weight 1 per student)

Cost (smaller is better, like schedule_fitness in Chapter 3):
    cost = HARD_W * (H1 + H2) + S1 + S2
"""
import csv
import json
import os

import numpy as np

HARD_W = 1000
W_CONSECUTIVE = 3
W_SAME_DAY = 1


class ExamProblem:
    def __init__(self, data_dir):
        with open(os.path.join(data_dir, "config.json")) as f:
            cfg = json.load(f)
        self.name = cfg["name"]
        self.E = cfg["n_exams"]
        self.D = cfg["days"]
        self.P = cfg["periods_per_day"]
        self.T = self.D * self.P
        self.seats = cfg["seats_per_slot"]

        self.size = np.zeros(self.E)
        with open(os.path.join(data_dir, "exams.csv")) as f:
            for row in csv.DictReader(f):
                self.size[int(row["exam_id"])] = int(row["n_students"])

        # conflict matrix C[i, j] = number of students taking both exam i and exam j
        by_student = {}
        with open(os.path.join(data_dir, "enrollments.csv")) as f:
            for row in csv.DictReader(f):
                by_student.setdefault(int(row["student_id"]), []).append(int(row["exam_id"]))
        C = np.zeros((self.E, self.E))
        for exams in by_student.values():
            for a in exams:
                for b in exams:
                    if a != b:
                        C[a, b] += 1
        self.C = C
        self.n_students = len(by_student)
        # list of conflicting pairs (i<j) -> fast vectorised evaluation
        I, J = np.nonzero(np.triu(C, 1))
        self.I, self.J, self.W = I, J, C[I, J]
        self.degree = (C > 0).sum(1)
        self.evaluations = 0  # counter (fair budget between algorithms)

    # ------------------------------------------------------------------
    def evaluate_many(self, S):
        """S: (n, E) int array of solutions -> (cost, hard, soft) arrays of length n."""
        S = np.asarray(S, dtype=np.int64)
        if S.ndim == 1:
            S = S[None, :]
        n = S.shape[0]
        self.evaluations += n
        a, b = S[:, self.I], S[:, self.J]
        same_slot = a == b
        same_day = (a // self.P) == (b // self.P)
        consecutive = same_day & (np.abs(a - b) == 1)
        other_same_day = same_day & ~same_slot & ~consecutive

        clash = same_slot @ self.W
        # capacity overflow per slot
        load = np.zeros((n, self.T))
        np.add.at(load, (np.repeat(np.arange(n), self.E), S.ravel()), np.tile(self.size, n))
        overflow = np.clip(load - self.seats, 0, None).sum(1)

        hard = clash + overflow
        soft = W_CONSECUTIVE * (consecutive @ self.W) + W_SAME_DAY * (other_same_day @ self.W)
        return HARD_W * hard + soft, hard, soft

    def evaluate(self, s):
        c, h, so = self.evaluate_many(np.asarray(s)[None, :])
        return float(c[0]), float(h[0]), float(so[0])

    def random_solutions(self, n, rng):
        return rng.integers(0, self.T, size=(n, self.E))

    # ------------------------------------------------------------------
    def describe(self, s):
        """Readable timetable: list of (day, period, [exam ids])."""
        out = []
        for t in range(self.T):
            exams = [i for i in range(self.E) if s[i] == t]
            out.append((t // self.P + 1, t % self.P + 1, exams))
        return out
