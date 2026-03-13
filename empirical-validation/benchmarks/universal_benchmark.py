#!/usr/bin/env python3
"""
universal_solve — Benchmark Suite
===================================
Measures timing, path-enumeration quality, selection accuracy, and
fallback behaviour of the ``universal_solve`` routine.

What is benchmarked
-------------------
1. **Enumeration latency** — wall-clock time for ``enumerate_all(problem)``
   over a representative set of 20 distinct problems.
2. **Solve latency** — end-to-end time for ``universal_solve(problem)``
   (enumerate + rank + execute) per problem.
3. **Selection accuracy** — verifies the returned candidate is always the
   highest-scoring proven path (``proven_and_cheap`` maximum).
4. **Fallback throughput** — measures how many retries ``retry_next``
   performs until it succeeds after a sequence of forced failures.
5. **Invariants** — determinism (same input → same output), at-least-one
   proven path per enumeration, step-count integrity.

Outputs
-------
All outputs go to ``--output-dir`` (default: ``benchmarks/results/``):

  universal_latency.csv           Per-problem latency table.
  universal_summary.json          Machine-readable summary.
  universal_benchmark_report.md   Human-readable Markdown report.

Usage
-----
    python benchmarks/universal_benchmark.py [--output-dir DIR] [--no-plots]

Exit code
---------
  0  All benchmark assertions passed.
  1  One or more assertions failed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev

# Allow imports from the empirical-validation root.
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from optimizers.universal_solve import (  # noqa: E402
    MU8_ORDER,
    enumerate_all,
    execute,
    proven_and_cheap,
    retry_next,
    universal_solve,
)
from optimizers.mu8_cycle_optimizer import (  # noqa: E402
    LEAN_FINGERPRINT,
    MU_ANGLE,
    SILVER_RATIO,
    C_NATURAL,
    PHASE_NAMES,
)

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark parameters
# ─────────────────────────────────────────────────────────────────────────────

#: Representative toy problems used across all benchmarks.
BENCHMARK_PROBLEMS = [
    "mu8_identity",
    "toy",
    "hello_world",
    "optimise_cost",
    "minimise_frustration",
    "maximise_coherence",
    "spiral_phase",
    "silver_ratio_test",
    "c_natural_137",
    "lean_verified",
    "problem_alpha",
    "problem_beta",
    "problem_gamma",
    "problem_delta",
    "complex_query_1",
    "complex_query_2",
    "empty_string_seed",
    "numeric_42",
    "dict_seed",
    "tuple_seed",
]

#: Number of timing repetitions per problem.
N_REPS = 10

#: Number of forced-failure steps used in the fallback throughput benchmark.
FALLBACK_FAIL_PREFIX = 3  # fail the top-3 ranked candidates before succeeding


# ─────────────────────────────────────────────────────────────────────────────
# 1. Enumeration latency
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_enumeration_latency() -> list[dict]:
    """Time enumerate_all() for each problem in BENCHMARK_PROBLEMS."""
    rows: list[dict] = []
    for problem in BENCHMARK_PROBLEMS:
        times: list[float] = []
        for _ in range(N_REPS):
            t0 = time.perf_counter()
            paths = enumerate_all(problem)
            times.append(time.perf_counter() - t0)

        paths = enumerate_all(problem)  # final call for metrics
        n_proven = sum(1 for p in paths if p["proven"])
        best = max(paths, key=proven_and_cheap)

        rows.append({
            "problem":         str(problem)[:30],
            "n_paths":         len(paths),
            "n_proven":        n_proven,
            "best_id":         best["id"],
            "best_steps":      len(best["steps"]),
            "best_proven":     best["proven"],
            "best_cost":       round(best["cost"], 6),
            "median_us":       median(times) * 1e6,
            "min_us":          min(times) * 1e6,
            "max_us":          max(times) * 1e6,
            "stdev_us":        stdev(times) * 1e6 if len(times) > 1 else 0.0,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 2. End-to-end solve latency
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_solve_latency() -> list[dict]:
    """Time universal_solve() end-to-end for each problem."""
    rows: list[dict] = []
    for problem in BENCHMARK_PROBLEMS:
        times: list[float] = []
        result = None
        for _ in range(N_REPS):
            t0 = time.perf_counter()
            result = universal_solve(problem)
            times.append(time.perf_counter() - t0)

        rows.append({
            "problem":       str(problem)[:30],
            "success":       result is not None,
            "result_proven": result["proven"] if result else None,
            "result_id":     result["id"] if result else None,
            "median_us":     median(times) * 1e6,
            "min_us":        min(times) * 1e6,
            "max_us":        max(times) * 1e6,
            "stdev_us":      stdev(times) * 1e6 if len(times) > 1 else 0.0,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 3. Selection accuracy
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_selection_accuracy() -> dict:
    """Verify universal_solve always returns the highest-scoring proven path.

    Uses a custom executor that accepts *all* candidates so the tie-breaking
    logic is fully exercised.
    """
    violations: list[str] = []

    def accept_all(path: dict) -> bool:
        return True

    for problem in BENCHMARK_PROBLEMS:
        paths = enumerate_all(problem)
        ranked = sorted(paths, key=proven_and_cheap, reverse=True)
        expected_id = ranked[0]["id"]

        result = universal_solve(problem, _execute=accept_all)
        if result is None:
            violations.append(f"{problem}: universal_solve returned None (expected {expected_id})")
        elif result["id"] != expected_id:
            violations.append(
                f"{problem}: got {result['id']} (score={proven_and_cheap(result):.6f}), "
                f"expected {expected_id} (score={proven_and_cheap(ranked[0]):.6f})"
            )

    return {
        "n_problems":  len(BENCHMARK_PROBLEMS),
        "n_correct":   len(BENCHMARK_PROBLEMS) - len(violations),
        "n_violations": len(violations),
        "violations":  violations,
        "passed":      len(violations) == 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Fallback throughput
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_fallback_throughput() -> list[dict]:
    """Measure retry_next throughput when the top-k candidates fail.

    For each problem, fail the top ``FALLBACK_FAIL_PREFIX`` ranked candidates
    then succeed on the next.  Records how many candidates were tried before
    success.
    """
    rows: list[dict] = []
    for problem in BENCHMARK_PROBLEMS:
        paths = enumerate_all(problem)
        ranked = sorted(paths, key=proven_and_cheap, reverse=True)
        # Collect IDs of candidates to fail.
        fail_ids = {p["id"] for p in ranked[:FALLBACK_FAIL_PREFIX]}

        tried: list[str] = []

        def executor(path: dict) -> bool:
            tried.append(path["id"])
            return path["id"] not in fail_ids

        t0 = time.perf_counter()
        result = universal_solve(problem, _execute=executor)
        elapsed_us = (time.perf_counter() - t0) * 1e6

        rows.append({
            "problem":         str(problem)[:30],
            "fail_prefix":     FALLBACK_FAIL_PREFIX,
            "n_tried":         len(tried),
            "success":         result is not None,
            "result_id":       result["id"] if result else None,
            "elapsed_us":      round(elapsed_us, 3),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 5. Invariant verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_invariants() -> dict:
    """Verify all universal_solve invariants across BENCHMARK_PROBLEMS."""
    violations: list[str] = []

    expected_n_paths = max(MU8_ORDER, len(PHASE_NAMES) * 2)

    for problem in BENCHMARK_PROBLEMS:
        # Determinism: two calls return the same paths.
        paths1 = enumerate_all(problem)
        paths2 = enumerate_all(problem)
        if paths1 != paths2:
            violations.append(f"{problem}: enumerate_all is non-deterministic")

        # Path count.
        if len(paths1) != expected_n_paths:
            violations.append(
                f"{problem}: expected {expected_n_paths} paths, got {len(paths1)}"
            )

        # At least one proven path.
        n_proven = sum(1 for p in paths1 if p["proven"])
        if n_proven < 1:
            violations.append(f"{problem}: no proven path in enumeration")

        # Proven ↔ step count divisible by MU8_ORDER.
        for p in paths1:
            if p["proven"] != (len(p["steps"]) % MU8_ORDER == 0):
                violations.append(
                    f"{problem}/{p['id']}: proven={p['proven']} but "
                    f"len(steps)={len(p['steps'])} % {MU8_ORDER} = "
                    f"{len(p['steps']) % MU8_ORDER}"
                )

        # Steps drawn only from PHASE_NAMES.
        for p in paths1:
            bad = [s for s in p["steps"] if s not in PHASE_NAMES]
            if bad:
                violations.append(
                    f"{problem}/{p['id']}: unexpected steps {bad}"
                )

        # Unique IDs.
        ids = [p["id"] for p in paths1]
        if len(ids) != len(set(ids)):
            violations.append(f"{problem}: duplicate path IDs in enumeration")

        # universal_solve returns proven result by default executor.
        result = universal_solve(problem)
        if result is not None and not result["proven"]:
            violations.append(
                f"{problem}: default executor returned non-proven result {result['id']}"
            )

    return {
        "n_problems":    len(BENCHMARK_PROBLEMS),
        "n_violations":  len(violations),
        "violations":    violations,
        "passed":        len(violations) == 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report generators
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _generate_markdown(
    enum_rows: list[dict],
    solve_rows: list[dict],
    selection: dict,
    fallback_rows: list[dict],
    invariants: dict,
    output_dir: Path,
    timestamp: str,
) -> Path:
    """Write universal_benchmark_report.md and return the path."""
    md_path = output_dir / "universal_benchmark_report.md"
    lines: list[str] = []
    a = lines.append

    a("# universal_solve — Benchmark Report")
    a("")
    a(f"**Generated:** {timestamp}  ")
    a(f"**Lean fingerprint:** `{LEAN_FINGERPRINT[:32]}…`  ")
    a(f"**μ angle:** `3π/4 = {MU_ANGLE:.10f}` rad  ")
    a(f"**Silver ratio:** `δS = 1+√2 = {SILVER_RATIO:.10f}`  ")
    a(f"**c_natural:** `{C_NATURAL}`  ")
    a(f"**MU8_ORDER:** `{MU8_ORDER}`  ")
    a(f"**Paths per problem:** `{max(MU8_ORDER, len(PHASE_NAMES) * 2)}`  ")
    a("")

    # ── Algorithm overview ────────────────────────────────────────────────────
    a("## Algorithm")
    a("")
    a("```python")
    a("def universal_solve(problem):")
    a("    paths = enumerate_all(problem)   # deterministic, SHA-256 seeded")
    a("    best  = max(paths, key=proven_and_cheap)  # proof bonus + low cost")
    a("    execute(best) or retry_next()   # fallback to next-best on failure")
    a("```")
    a("")
    a("**Path enumeration** generates `max(MU8_ORDER, len(PHASE_NAMES)×2) = "
      f"{max(MU8_ORDER, len(PHASE_NAMES) * 2)}` candidates per problem.")
    a("Each candidate is seeded by `SHA-256(str(problem))`, ensuring reproducibility.")
    a("A path is *proven* when its step count is divisible by `MU8_ORDER=8`")
    a("(completing a full μ⁸ revolution).  The `proven_and_cheap` score blends")
    a("the Lean coherence bonus `C(1)=1.0` with a normalised cost penalty.")
    a("")

    # ── 1. Enumeration latency ────────────────────────────────────────────────
    a("## 1. Enumeration Latency")
    a("")
    med_enum = median([r["median_us"] for r in enum_rows])
    min_enum = min(r["min_us"] for r in enum_rows)
    max_enum = max(r["max_us"] for r in enum_rows)
    a(f"Timing across {len(enum_rows)} problems × {N_REPS} repetitions each.")
    a("")
    a(f"| Metric | Value |")
    a(f"|--------|-------|")
    a(f"| Median (across all problems) | {med_enum:.2f} μs |")
    a(f"| Fastest single call | {min_enum:.2f} μs |")
    a(f"| Slowest single call | {max_enum:.2f} μs |")
    a("")
    a("| Problem | Paths | Proven | Best ID | Best Steps | Median (μs) | σ (μs) |")
    a("|---------|-------|--------|---------|------------|-------------|--------|")
    for r in enum_rows:
        a(f"| `{r['problem']}` | {r['n_paths']} | {r['n_proven']} | {r['best_id']} "
          f"| {r['best_steps']} | {r['median_us']:.2f} | {r['stdev_us']:.2f} |")
    a("")

    # ── 2. End-to-end solve latency ───────────────────────────────────────────
    a("## 2. End-to-End Solve Latency")
    a("")
    med_solve = median([r["median_us"] for r in solve_rows])
    a(f"Includes `enumerate_all` + `proven_and_cheap` ranking + `execute`.")
    a("")
    a(f"| Metric | Value |")
    a(f"|--------|-------|")
    a(f"| Median solve time (across all problems) | {med_solve:.2f} μs |")
    a(f"| Fastest solve | {min(r['min_us'] for r in solve_rows):.2f} μs |")
    a(f"| Slowest solve | {max(r['max_us'] for r in solve_rows):.2f} μs |")
    a("")
    a("| Problem | Success | Result ID | Proven | Median (μs) | σ (μs) |")
    a("|---------|---------|-----------|--------|-------------|--------|")
    for r in solve_rows:
        proven_str = str(r["result_proven"]) if r["result_proven"] is not None else "—"
        a(f"| `{r['problem']}` | {'✓' if r['success'] else '✗'} | "
          f"{r['result_id'] or '—'} | {proven_str} | {r['median_us']:.2f} | {r['stdev_us']:.2f} |")
    a("")

    # ── 3. Selection accuracy ─────────────────────────────────────────────────
    a("## 3. Selection Accuracy")
    a("")
    status = "✓ ALL CORRECT" if selection["passed"] else f"✗ {selection['n_violations']} ERROR(S)"
    a(f"**Status:** {status}  ")
    a("")
    a(f"| Metric | Value |")
    a(f"|--------|-------|")
    a(f"| Problems checked | {selection['n_problems']} |")
    a(f"| Correct selections | {selection['n_correct']} |")
    a(f"| Violations | {selection['n_violations']} |")
    a("")
    if selection["violations"]:
        a("**Violations:**")
        for v in selection["violations"]:
            a(f"- `{v}`")
        a("")

    # ── 4. Fallback throughput ────────────────────────────────────────────────
    a("## 4. Fallback / Retry Throughput")
    a("")
    a(f"Top {FALLBACK_FAIL_PREFIX} candidates are forced to fail; measures how "
      "many candidates `universal_solve` tries before succeeding.")
    a("")
    n_tried_vals = [r["n_tried"] for r in fallback_rows]
    a(f"| Metric | Value |")
    a(f"|--------|-------|")
    a(f"| Fail prefix | {FALLBACK_FAIL_PREFIX} |")
    a(f"| Median candidates tried | {median(n_tried_vals):.1f} |")
    a(f"| Max candidates tried | {max(n_tried_vals)} |")
    a(f"| All fallbacks succeeded | {'✓ YES' if all(r['success'] for r in fallback_rows) else '✗ NO'} |")
    a("")
    a("| Problem | Tried | Success | Result ID | Elapsed (μs) |")
    a("|---------|-------|---------|-----------|--------------|")
    for r in fallback_rows:
        a(f"| `{r['problem']}` | {r['n_tried']} | {'✓' if r['success'] else '✗'} "
          f"| {r['result_id'] or '—'} | {r['elapsed_us']:.1f} |")
    a("")

    # ── 5. Invariant verification ─────────────────────────────────────────────
    a("## 5. Invariant Verification")
    a("")
    inv_status = "✓ ALL PASSED" if invariants["passed"] else f"✗ {invariants['n_violations']} VIOLATION(S)"
    a(f"**Status:** {inv_status}  ")
    a("")
    a("| Invariant | Status |")
    a("|-----------|--------|")
    a(f"| Enumeration is deterministic | {'✓' if invariants['passed'] else 'see violations'} |")
    a(f"| Each problem yields {max(MU8_ORDER, len(PHASE_NAMES) * 2)} paths | "
      f"{'✓' if invariants['passed'] else 'see violations'} |")
    a(f"| At least one proven path per problem | {'✓' if invariants['passed'] else 'see violations'} |")
    a(f"| proven ↔ step count % {MU8_ORDER} == 0 | {'✓' if invariants['passed'] else 'see violations'} |")
    a(f"| All steps in PHASE_NAMES | {'✓' if invariants['passed'] else 'see violations'} |")
    a(f"| Unique path IDs | {'✓' if invariants['passed'] else 'see violations'} |")
    a(f"| Default executor returns proven path | {'✓' if invariants['passed'] else 'see violations'} |")
    a("")
    if invariants["violations"]:
        a("**Violations:**")
        for v in invariants["violations"]:
            a(f"- `{v}`")
        a("")

    # ── Next steps ────────────────────────────────────────────────────────────
    a("## Recommendations & Next Steps")
    a("")
    a("The following improvements are suggested based on the benchmark results:")
    a("")
    a("### Short-term (algorithmic / robustness)")
    a("")
    a("1. **Richer path representation**  ")
    a("   Current paths carry only phase-name steps.  Attaching numeric")
    a("   parameter vectors (e.g. gain, N, seed) would let `execute` call")
    a("   `Mu8CycleOptimizer` directly and return real convergence metrics")
    a("   instead of the boolean stub.")
    a("")
    a("2. **Adaptive cost function**  ")
    a("   `proven_and_cheap` uses a fixed `cost_scale = 2×MU8_ORDER`.  A")
    a("   problem-aware scaling (e.g. based on constraint count or domain size)")
    a("   would produce better ordering for harder problems.")
    a("")
    a("3. **Partial-proof acceptance**  ")
    a("   Add a `partially_proven` tier for paths whose step count is")
    a("   divisible by a factor of `MU8_ORDER` (e.g. 4 steps = half revolution).")
    a("   This widens the proven band and can improve fallback success rates.")
    a("")
    a("### Medium-term (integration)")
    a("")
    a("4. **Hook `execute` to `Mu8CycleOptimizer.run_cycle`**  ")
    a("   Replace the boolean stub with a real execution backend that runs the")
    a("   μ⁸ five-phase cycle for the path's step sequence and returns")
    a("   `CycleMetrics`.  This closes the loop between enumeration and")
    a("   the Lean-verified optimizer.")
    a("")
    a("5. **Memoisation / caching**  ")
    a("   `enumerate_all` is already deterministic; cache the path list keyed")
    a("   by `SHA-256(str(problem))` so repeated calls within a session pay")
    a("   no recomputation cost.")
    a("")
    a("6. **Parallel path evaluation**  ")
    a("   When `execute` is CPU-bound (real solver calls), evaluate the top-k")
    a("   candidates concurrently via `concurrent.futures.ThreadPoolExecutor`")
    a("   and return the first winner.")
    a("")
    a("### Long-term (formal verification)")
    a("")
    a("7. **Lean theorem for `proven_and_cheap` monotonicity**  ")
    a("   Prove in Lean 4 that the scoring function is monotone in R (coherence)")
    a("   and anti-monotone in cost, matching the spiral-cycle invariants in")
    a("   `CriticalEigenvalue.lean`.")
    a("")
    a("8. **Extend the CI pipeline**  ")
    a("   Add `universal_benchmark.py` as a step in `mu8-pipeline.yml`")
    a("   (alongside the existing μ⁸ benchmark job) so that enumeration latency")
    a("   and selection accuracy are tracked on every push to `main`.")
    a("")
    a("9. **Convergence curve for `universal_solve`**  ")
    a("   Once `execute` calls the real optimizer, plot R vs. problem-index")
    a("   (analogous to the convergence curve in `mu8_benchmark.py`) to")
    a("   visualise how solution quality improves as the candidate pool grows.")
    a("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(output_dir: Path, no_plots: bool = False) -> int:
    """Execute the full universal_solve benchmark suite.

    Returns
    -------
    int
        0 on full pass, 1 if any assertion failed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print("=" * 70)
    print("  universal_solve — Benchmark Suite")
    print("=" * 70)
    print(f"  Lean fingerprint : {LEAN_FINGERPRINT[:32]}…")
    print(f"  μ angle          : 3π/4 = {MU_ANGLE:.10f} rad")
    print(f"  Silver ratio δS  : {SILVER_RATIO:.10f}")
    print(f"  c_natural        : {C_NATURAL}")
    print(f"  MU8_ORDER        : {MU8_ORDER}")
    print(f"  Paths/problem    : {max(MU8_ORDER, len(PHASE_NAMES) * 2)}")
    print(f"  Problems         : {len(BENCHMARK_PROBLEMS)}")
    print()

    # 1. Enumeration latency
    print(f"[1/5] Enumeration latency  ({len(BENCHMARK_PROBLEMS)} problems × {N_REPS} reps) …")
    enum_rows = benchmark_enumeration_latency()
    med_enum = median([r["median_us"] for r in enum_rows])
    print(f"      Median: {med_enum:.2f} μs  "
          f"Min: {min(r['min_us'] for r in enum_rows):.2f} μs  "
          f"Max: {max(r['max_us'] for r in enum_rows):.2f} μs")

    # 2. Solve latency
    print()
    print(f"[2/5] End-to-end solve latency  ({len(BENCHMARK_PROBLEMS)} problems × {N_REPS} reps) …")
    solve_rows = benchmark_solve_latency()
    med_solve = median([r["median_us"] for r in solve_rows])
    print(f"      Median: {med_solve:.2f} μs  "
          f"Min: {min(r['min_us'] for r in solve_rows):.2f} μs  "
          f"Max: {max(r['max_us'] for r in solve_rows):.2f} μs")
    n_success = sum(1 for r in solve_rows if r["success"])
    print(f"      Success rate: {n_success}/{len(solve_rows)}")

    # 3. Selection accuracy
    print()
    print("[3/5] Selection accuracy …")
    selection = benchmark_selection_accuracy()
    if selection["passed"]:
        print(f"      ALL {selection['n_problems']} SELECTIONS CORRECT ✓")
    else:
        print(f"      {selection['n_violations']} VIOLATION(S):")
        for v in selection["violations"]:
            print(f"        • {v}")

    # 4. Fallback throughput
    print()
    print(f"[4/5] Fallback throughput  (fail top-{FALLBACK_FAIL_PREFIX} candidates) …")
    fallback_rows = benchmark_fallback_throughput()
    n_tried_vals = [r["n_tried"] for r in fallback_rows]
    n_fb_success = sum(1 for r in fallback_rows if r["success"])
    print(f"      Median candidates tried: {median(n_tried_vals):.1f}")
    print(f"      Fallback success rate: {n_fb_success}/{len(fallback_rows)}")

    # 5. Invariant verification
    print()
    print("[5/5] Invariant verification …")
    invariants = verify_invariants()
    if invariants["passed"]:
        print(f"      ALL INVARIANTS HELD ✓  ({invariants['n_problems']} problems)")
    else:
        print(f"      {invariants['n_violations']} VIOLATION(S):")
        for v in invariants["violations"]:
            print(f"        • {v}")

    # Write CSV
    _write_csv(output_dir / "universal_latency.csv", enum_rows)
    print(f"\n  universal_latency.csv       → {output_dir / 'universal_latency.csv'}")

    # JSON summary
    all_passed = (
        selection["passed"]
        and invariants["passed"]
        and all(r["success"] for r in solve_rows)
        and all(r["success"] for r in fallback_rows)
    )
    summary = {
        "generated":                timestamp,
        "lean_fingerprint":         LEAN_FINGERPRINT,
        "mu_angle":                 MU_ANGLE,
        "silver_ratio":             SILVER_RATIO,
        "c_natural":                C_NATURAL,
        "mu8_order":                MU8_ORDER,
        "paths_per_problem":        max(MU8_ORDER, len(PHASE_NAMES) * 2),
        "n_problems":               len(BENCHMARK_PROBLEMS),
        "enumeration_median_us":    med_enum,
        "solve_median_us":          med_solve,
        "selection_accuracy":       selection,
        "fallback_throughput": {
            "fail_prefix":          FALLBACK_FAIL_PREFIX,
            "median_tries":         median(n_tried_vals),
            "max_tries":            max(n_tried_vals),
            "success_rate":         f"{n_fb_success}/{len(fallback_rows)}",
        },
        "invariants":               invariants,
        "all_passed":               all_passed,
    }
    json_path = output_dir / "universal_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  universal_summary.json      → {json_path}")

    # Markdown report
    md_path = _generate_markdown(
        enum_rows, solve_rows, selection, fallback_rows, invariants,
        output_dir, timestamp,
    )
    print(f"  universal_benchmark_report.md → {md_path}")

    # Final status
    print()
    print("=" * 70)
    if all_passed:
        print("  ALL BENCHMARK CHECKS PASSED ✓")
    else:
        print("  SOME CHECKS FAILED — see report for details")
    print("=" * 70)

    return 0 if all_passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="universal_solve benchmark suite"
    )
    parser.add_argument(
        "--output-dir",
        default=str(_HERE / "results"),
        help="Directory for benchmark output (default: benchmarks/results/)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="(reserved for future use — no plots are currently generated)",
    )
    args = parser.parse_args()
    sys.exit(run(output_dir=Path(args.output_dir), no_plots=args.no_plots))


if __name__ == "__main__":
    main()
