"""
توليد بيانات مسألة جدولة الامتحانات (Small / Medium / Large)
Synthetic, reproducible exam-timetabling instances.

Model:
  - Courses are grouped into "programs" (e.g. a year of a major).
  - Each student belongs to one program, takes most of their courses from it,
    and sometimes one elective from another program.
  - This creates a realistic conflict graph: dense inside programs, sparse across them.

Output per size (data/<size>/):
  exams.csv        exam_id, name, n_students
  enrollments.csv  student_id, exam_id
  config.json      days, periods_per_day, seats_per_slot, ...
"""
import csv
import json
import os
import random

SIZES = {
    #          exams students days periods programs seed
    "small":  dict(n_exams=15,  n_students=250,  days=5,  periods=2, n_programs=3,  seed=1),
    "medium": dict(n_exams=40,  n_students=1000, days=6,  periods=3, n_programs=6,  seed=2),
    "large":  dict(n_exams=100, n_students=3000, days=10, periods=3, n_programs=12, seed=3),
}


def generate(name, n_exams, n_students, days, periods, n_programs, seed, out_dir):
    rnd = random.Random(seed)
    exams = list(range(n_exams))
    # split exams into programs (round-robin keeps groups balanced)
    programs = [exams[p::n_programs] for p in range(n_programs)]

    n_slots = days * periods
    # A hidden reference timetable guarantees that a feasible solution exists
    # (no clashes): exams of the same program get different slots, and a
    # student only takes an elective whose hidden slot differs from his courses.
    hidden = [0] * n_exams
    for prog in programs:
        for e, t in zip(prog, rnd.sample(range(n_slots), len(prog))):
            hidden[e] = t

    enroll = []  # (student, exam)
    for s in range(n_students):
        prog = programs[rnd.randrange(n_programs)]
        k = min(len(prog), rnd.randint(3, 5))
        chosen = set(rnd.sample(prog, k))
        if rnd.random() < 0.3:  # one elective from another program
            used = {hidden[e] for e in chosen}
            options = [e for e in exams if e not in chosen and hidden[e] not in used]
            if options:
                chosen.add(rnd.choice(options))
        for e in sorted(chosen):
            enroll.append((s, e))

    size = [0] * n_exams
    for _, e in enroll:
        size[e] += 1

    # seats per slot: 10% more than the busiest slot of the hidden timetable
    hidden_load = [0] * n_slots
    for e in exams:
        hidden_load[hidden[e]] += size[e]
    seats = int(1.1 * max(hidden_load)) + 1

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exams.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exam_id", "name", "n_students"])
        for e in exams:
            w.writerow([e, f"Course_{e:03d}", size[e]])
    with open(os.path.join(out_dir, "enrollments.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["student_id", "exam_id"])
        w.writerows(enroll)
    cfg = dict(name=name, n_exams=n_exams, n_students=n_students, days=days,
               periods_per_day=periods, n_slots=n_slots, seats_per_slot=seats,
               n_enrollments=len(enroll), seed=seed,
               # known clash-free timetable (used only to prove feasibility,
               # never given to the algorithms)
               reference_solution=hidden)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for name, p in SIZES.items():
        cfg = generate(name, out_dir=os.path.join(here, name), **p)
        print(cfg)
