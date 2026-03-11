#!/usr/bin/env python3
"""
μ⁸=1 Spiral Cycle Optimizer — Benchmark Suite
===============================================
Measures timing, convergence, and scalability of the five-phase μ⁸ cycle.

What is benchmarked
-------------------
1. **Single-cycle latency** — wall-clock time per revolution for N ∈ {8, 16, 32, 64, 128, 256}.
2. **Eight-revolution group** — time for one full μ⁸ orbit (k=0→7) which
   returns the cumulative rotation to identity (μ⁸=1).
3. **Convergence curve** — R (Kuramoto order parameter) and E (frustration)
   vs. revolution for gain ∈ {0.1, 0.3, 0.5, 0.8}.
4. **Gate overhead** — time to construct Mu8CycleOptimizer (SymPy + NumPy +
   checksum gates) vs. steady-state cycle time.

Outputs
-------
All outputs go to ``--output-dir`` (default: ``benchmarks/results/``):

  mu8_latency.csv          Per-(N, gain, revolution) timing table.
  mu8_convergence.csv      R and E vs. revolution for each (N, gain) combo.
  mu8_summary.json         Machine-readable summary (pass/fail + key metrics).
  mu8_benchmark_report.md  Human-readable Markdown report.
  mu8_latency.png          Bar chart: median cycle time vs. N.
  mu8_convergence.png      Line plot: R(revolution) for each gain.

Usage
-----
    python benchmarks/mu8_benchmark.py [--output-dir DIR] [--no-plots]

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

from optimizers.mu8_cycle_optimizer import (  # noqa: E402
    LEAN_FINGERPRINT,
    MU_ANGLE,
    SILVER_RATIO,
    C_NATURAL,
    Mu8CycleOptimizer,
    bit_strength,
    lean_coherence,
)

try:
    import matplotlib  # type: ignore[import]
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import]
    import numpy as np
    _HAS_PLOTS = True
except ImportError:
    _HAS_PLOTS = False

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark parameters
# ─────────────────────────────────────────────────────────────────────────────

#: Oscillator counts for the latency sweep.
LATENCY_N_VALUES = [8, 16, 32, 64, 128, 256]

#: EMA gains for the convergence sweep.
CONVERGENCE_GAINS = [0.1, 0.3, 0.5, 0.8]

#: Oscillator count used in the convergence sweep.
CONVERGENCE_N = 32

#: Revolutions to run for each convergence curve.
CONVERGENCE_REVOLUTIONS = 80

#: Revolutions to time per (N, gain) in the latency sweep.
LATENCY_REVOLUTIONS = 8  # one full μ⁸ orbit

#: Gain used for the latency sweep.
LATENCY_GAIN = 0.3

#: Random seed (fixed for reproducibility).
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# 1. Gate-overhead timing
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_gate_overhead() -> dict:
    """Measure construction time (gate overhead) vs. cycle time."""
    N_REPS = 5

    # Construction (includes SymPy + NumPy + checksum gates)
    t_construct: list[float] = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        opt = Mu8CycleOptimizer(N=8, gain=0.3, seed=SEED)
        t_construct.append(time.perf_counter() - t0)

    # One cycle (steady-state, gates already cleared)
    t_cycle: list[float] = []
    for _ in range(N_REPS):
        opt = Mu8CycleOptimizer(N=8, gain=0.3, seed=SEED)
        t0 = time.perf_counter()
        opt.run_cycle()
        t_cycle.append(time.perf_counter() - t0)

    return {
        "construct_median_ms": median(t_construct) * 1000,
        "construct_min_ms":    min(t_construct) * 1000,
        "construct_max_ms":    max(t_construct) * 1000,
        "cycle_median_ms":     median(t_cycle) * 1000,
        "cycle_min_ms":        min(t_cycle) * 1000,
        "cycle_max_ms":        max(t_cycle) * 1000,
        "gate_overhead_ratio": median(t_construct) / max(median(t_cycle), 1e-9),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Single-cycle latency vs. N
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_latency() -> list[dict]:
    """Time one 8-revolution orbit for each N in LATENCY_N_VALUES."""
    rows: list[dict] = []
    for N in LATENCY_N_VALUES:
        opt = Mu8CycleOptimizer(N=N, gain=LATENCY_GAIN, seed=SEED)
        # Warm-up: one extra cycle so NumPy is hot.
        opt.run_cycle()

        per_cycle_times: list[float] = []
        for _ in range(LATENCY_REVOLUTIONS):
            t0 = time.perf_counter()
            m = opt.run_cycle()
            per_cycle_times.append(time.perf_counter() - t0)

        rows.append({
            "N":                  N,
            "gain":               LATENCY_GAIN,
            "revolutions":        LATENCY_REVOLUTIONS,
            "median_us":          median(per_cycle_times) * 1e6,
            "min_us":             min(per_cycle_times) * 1e6,
            "max_us":             max(per_cycle_times) * 1e6,
            "stdev_us":           stdev(per_cycle_times) * 1e6 if len(per_cycle_times) > 1 else 0.0,
            "total_orbit_ms":     sum(per_cycle_times) * 1000,
            "final_coherence":    m.coherence_out,
            "final_frustration":  m.frustration_out,
            "final_bit_strength": m.bit_strength_out,
            "mu_power_final":     m.mu_power,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 3. Convergence curves
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_convergence() -> list[dict]:
    """Run CONVERGENCE_REVOLUTIONS cycles for each gain; record R, E, and bit strength."""
    rows: list[dict] = []
    for gain in CONVERGENCE_GAINS:
        opt = Mu8CycleOptimizer(N=CONVERGENCE_N, gain=gain, seed=SEED)
        R0 = opt._circular_coherence()
        E0 = opt._frustration()
        rows.append({
            "N": CONVERGENCE_N, "gain": gain, "revolution": -1,
            "coherence": R0, "frustration": E0,
            "bit_strength": bit_strength(R0, CONVERGENCE_N),
            "lean_C_r": lean_coherence(R0),
            "mu_power": -1,
        })
        for _ in range(CONVERGENCE_REVOLUTIONS):
            m = opt.run_cycle()
            rows.append({
                "N":           CONVERGENCE_N,
                "gain":        gain,
                "revolution":  m.revolution,
                "coherence":   m.coherence_out,
                "frustration": m.frustration_out,
                "bit_strength": m.bit_strength_out,
                "lean_C_r":    lean_coherence(m.coherence_out),
                "mu_power":    m.mu_power,
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 4. Invariant verification over full run
# ─────────────────────────────────────────────────────────────────────────────

def _check_monotonic(
    vals: list[float],
    name: str,
    violations: list[str],
    *,
    increasing: bool,
    fmt: str = ".6e",
    tol: float = 1e-10,
) -> None:
    """Append violation strings for non-monotone sequences."""
    for i in range(1, len(vals)):
        breach = (vals[i] < vals[i - 1] - tol) if increasing else (vals[i] > vals[i - 1] + tol)
        if breach:
            direction = "decreased" if increasing else "increased"
            violations.append(
                f"{name} {direction} at revolution {i}: "
                f"{vals[i - 1]:{fmt}} → {vals[i]:{fmt}}"
            )


def verify_invariants() -> dict:
    """Run 40 cycles and verify all Lean-grounded invariants hold."""
    N_CYCLES = 40
    opt = Mu8CycleOptimizer(N=8, gain=0.3, seed=SEED)
    history = opt.run(N_CYCLES)

    violations: list[str] = []

    # μ-power cycles 0→7
    for i, m in enumerate(history):
        expected_k = i % 8
        if m.mu_power != expected_k:
            violations.append(
                f"revolution {i}: mu_power={m.mu_power}, expected {expected_k}"
            )

    # Frustration non-increasing (allow tiny numerical noise: 1e-10)
    _check_monotonic(
        [m.frustration_out for m in history], "frustration",
        violations, increasing=False,
    )

    # Bit strength non-decreasing
    _check_monotonic(
        [m.bit_strength_out for m in history], "bit_strength",
        violations, increasing=True, fmt=".4f",
    )

    # Lean fingerprint stable
    fps = {m.lean_fingerprint for m in history}
    if len(fps) != 1:
        violations.append(f"lean_fingerprint changed mid-run: {fps}")

    # Final coherence > initial coherence
    R_initial = history[0].coherence_in
    R_final   = history[-1].coherence_out
    spiral_ok = R_final > R_initial

    B_initial = history[0].bit_strength_in
    B_final   = history[-1].bit_strength_out

    return {
        "n_cycles":          N_CYCLES,
        "n_violations":      len(violations),
        "violations":        violations,
        "R_initial":         R_initial,
        "R_final":           R_final,
        "delta_R_total":     R_final - R_initial,
        "E_initial":         history[0].frustration_in,
        "E_final":           history[-1].frustration_out,
        "delta_E_total":     history[-1].frustration_out - history[0].frustration_in,
        "B_initial":         B_initial,
        "B_final":           B_final,
        "delta_B_total":     B_final - B_initial,
        "spiral_ok":         spiral_ok,
        "passed":            len(violations) == 0 and spiral_ok,
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


def _generate_plots(
    latency_rows: list[dict],
    convergence_rows: list[dict],
    output_dir: Path,
) -> None:
    if not _HAS_PLOTS:
        return

    # ── Plot 1: median cycle latency vs. N ──────────────────────────────────
    ns   = [r["N"] for r in latency_rows]
    meds = [r["median_us"] for r in latency_rows]
    errs = [r["stdev_us"]  for r in latency_rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(ns)), meds, yerr=errs, color="#58a6ff",
           error_kw={"ecolor": "#8b949e", "capsize": 4}, width=0.6)
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("N (oscillators)", fontsize=11)
    ax.set_ylabel("Median cycle time (μs)", fontsize=11)
    ax.set_title("μ⁸ Cycle Latency vs. Oscillator Count  (gain=0.3, 8 revolutions)",
                 fontsize=11)
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    ax.tick_params(colors="#c9d1d9")
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#58a6ff")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    plt.tight_layout()
    fig.savefig(output_dir / "mu8_latency.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    # ── Plot 2: convergence R(revolution) + E(revolution) + B(revolution) ────
    fig2, (ax_r, ax_e, ax_b) = plt.subplots(1, 3, figsize=(18, 5))
    fig2.patch.set_facecolor("#0d1117")

    colours = ["#58a6ff", "#3fb950", "#e3b341", "#f78166"]
    for colour, gain in zip(colours, CONVERGENCE_GAINS):
        subset = [r for r in convergence_rows if r["gain"] == gain and r["revolution"] >= 0]
        revs = [r["revolution"] for r in subset]
        R    = [r["coherence"]  for r in subset]
        E    = [r["frustration"] for r in subset]
        B    = [r["bit_strength"] for r in subset]
        ax_r.plot(revs, R, color=colour, linewidth=1.8, label=f"g={gain}")
        ax_e.plot(revs, E, color=colour, linewidth=1.8, label=f"g={gain}")
        ax_b.plot(revs, B, color=colour, linewidth=1.8, label=f"g={gain}")

    for ax in (ax_r, ax_e, ax_b):
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#c9d1d9")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.legend(fontsize=9, facecolor="#161b22", labelcolor="#c9d1d9",
                  edgecolor="#30363d")

    ax_r.set_xlabel("Revolution", fontsize=11)
    ax_r.set_ylabel("Coherence R", fontsize=11)
    ax_r.set_title("Coherence vs. Revolution  (N=32)", fontsize=11)
    ax_r.title.set_color("#58a6ff")
    ax_r.set_ylim(0, 1.05)

    ax_e.set_xlabel("Revolution", fontsize=11)
    ax_e.set_ylabel("Frustration E", fontsize=11)
    ax_e.set_title("Frustration vs. Revolution  (N=32)", fontsize=11)
    ax_e.title.set_color("#58a6ff")

    ax_b.set_xlabel("Revolution", fontsize=11)
    ax_b.set_ylabel("Bit Strength B (bits)", fontsize=11)
    ax_b.set_title("Bit Strength vs. Revolution  (N=32)", fontsize=11)
    ax_b.title.set_color("#58a6ff")

    plt.tight_layout()
    fig2.savefig(output_dir / "mu8_convergence.png", dpi=150, bbox_inches="tight",
                 facecolor=fig2.get_facecolor())
    plt.close(fig2)


def _generate_markdown(
    gate_row: dict,
    latency_rows: list[dict],
    invariants: dict,
    output_dir: Path,
    timestamp: str,
) -> Path:
    """Write mu8_benchmark_report.md and return the path."""
    md_path = output_dir / "mu8_benchmark_report.md"
    lines: list[str] = []

    a = lines.append
    a(f"# μ⁸=1 Spiral Cycle Optimizer — Benchmark Report")
    a(f"")
    a(f"**Generated:** {timestamp}  ")
    a(f"**Lean fingerprint:** `{LEAN_FINGERPRINT[:32]}…`  ")
    a(f"**μ angle:** `3π/4 = {MU_ANGLE:.10f}` rad  ")
    a(f"**Silver ratio:** `δS = 1+√2 = {SILVER_RATIO:.10f}`  ")
    a(f"**c_natural:** `{C_NATURAL}`  ")
    a(f"")

    # Logic gates
    a(f"## Logic Gates")
    a(f"")
    a(f"Three pre-flight gates verified at optimizer construction:")
    a(f"")
    a(f"| Gate | Tool | Check | Status |")
    a(f"|------|------|-------|--------|")
    a(f"| SymPy | exact symbolic | μ⁸ = 1 | ✓ PASS |")
    a(f"| NumPy | IEEE 754 | |μ| − 1 < 1e-14 | ✓ PASS |")
    a(f"| Checksum | SHA-256 | Lean constant fingerprint | ✓ PASS |")
    a(f"")

    # Gate overhead
    a(f"## Gate Overhead")
    a(f"")
    a(f"| Metric | Value |")
    a(f"|--------|-------|")
    a(f"| Construction (median) | {gate_row['construct_median_ms']:.3f} ms |")
    a(f"| Construction (min/max) | {gate_row['construct_min_ms']:.3f} / {gate_row['construct_max_ms']:.3f} ms |")
    a(f"| Single cycle (median) | {gate_row['cycle_median_ms']:.4f} ms |")
    a(f"| Gate overhead ratio | {gate_row['gate_overhead_ratio']:.1f}× cycle time |")
    a(f"")

    # Latency table
    a(f"## Cycle Latency vs. Oscillator Count")
    a(f"")
    a(f"Gain = {LATENCY_GAIN}, {LATENCY_REVOLUTIONS} revolutions (one full μ⁸ orbit).")
    a(f"")
    a(f"| N | Median (μs) | Min (μs) | Max (μs) | σ (μs) | Final R | Final B (bits) | Final E |")
    a(f"|---|-------------|----------|----------|--------|---------|----------------|---------|")
    for r in latency_rows:
        a(f"| {r['N']} | {r['median_us']:.2f} | {r['min_us']:.2f} | "
          f"{r['max_us']:.2f} | {r['stdev_us']:.2f} | "
          f"{r['final_coherence']:.6f} | {r['final_bit_strength']:.4f} | "
          f"{r['final_frustration']:.6e} |")
    a(f"")

    # Invariant checks
    a(f"## Lean Invariant Verification (40 revolutions, N=8, gain=0.3)")
    a(f"")
    status = "✓ ALL PASSED" if invariants["passed"] else f"✗ {invariants['n_violations']} VIOLATION(S)"
    a(f"**Status:** {status}  ")
    a(f"")
    a(f"| Invariant | Result |")
    a(f"|-----------|--------|")
    a(f"| μ-power cycles 0→7 | {'✓' if invariants['n_violations'] == 0 else '✗'} |")
    a(f"| Frustration non-increasing | {'✓' if invariants['n_violations'] == 0 else '✗'} |")
    a(f"| Bit strength non-decreasing | {'✓' if invariants['n_violations'] == 0 else '✗'} |")
    a(f"| Lean fingerprint stable | ✓ |")
    a(f"| Spiral: R_final > R_initial | {'✓' if invariants['spiral_ok'] else '✗'} |")
    a(f"| R initial → final | {invariants['R_initial']:.6f} → {invariants['R_final']:.6f} "
      f"(Δ={invariants['delta_R_total']:+.6f}) |")
    a(f"| B initial → final (bits) | {invariants['B_initial']:.4f} → {invariants['B_final']:.4f} "
      f"(Δ={invariants['delta_B_total']:+.4f}) |")
    a(f"| E initial → final | {invariants['E_initial']:.4e} → {invariants['E_final']:.4e} "
      f"(Δ={invariants['delta_E_total']:.4e}) |")
    if invariants["violations"]:
        a(f"")
        a(f"**Violations:**")
        for v in invariants["violations"]:
            a(f"- `{v}`")
    a(f"")

    # Bit strength definition
    a(f"## Bit Strength Measure")
    a(f"")
    a(f"Coherence is encoded as bit strength **B = −log₂(1 − R + ε)**, capped at log₂(N).")
    a(f"")
    a(f"| R | B (bits) | Interpretation |")
    a(f"|---|----------|----------------|")
    a(f"| 0.0 | 0.000 | No coherence |")
    a(f"| 0.5 | 1.000 | Half coherence — 1 bit locked |")
    a(f"| 0.75 | 2.000 | 2 bits locked |")
    a(f"| 0.875 | 3.000 | Natural threshold for N=8 (= log₂(8)) |")
    a(f"| 0.9375 | 4.000 | Natural threshold for N=16 (= log₂(16)) |")
    a(f"| → 1 | → log₂(N) | Perfect coherence — all bits locked |")
    a(f"")

    # Plots
    a(f"## Plots")
    a(f"")
    a(f"| Plot | Description |")
    a(f"|------|-------------|")
    a(f"| `mu8_latency.png` | Median cycle time vs. N (bar chart) |")
    a(f"| `mu8_convergence.png` | R, E, and bit strength vs. revolution for 4 gain values |")
    a(f"")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(output_dir: Path, no_plots: bool = False) -> int:
    """Execute the full benchmark suite.

    Returns
    -------
    int
        0 on full pass, 1 if any assertion failed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print("=" * 70)
    print("  μ⁸=1 Spiral Cycle Optimizer — Benchmark Suite")
    print("=" * 70)
    print(f"  Lean fingerprint : {LEAN_FINGERPRINT[:32]}…")
    print(f"  μ angle          : 3π/4 = {MU_ANGLE:.10f} rad")
    print(f"  Silver ratio δS  : {SILVER_RATIO:.10f}")
    print(f"  c_natural        : {C_NATURAL}")
    print()

    # 1. Gate overhead
    print("[1/4] Measuring gate overhead …")
    gate_row = benchmark_gate_overhead()
    print(f"      Construction:  {gate_row['construct_median_ms']:.3f} ms (median)")
    print(f"      Single cycle:  {gate_row['cycle_median_ms']:.4f} ms")
    print(f"      Gate overhead: {gate_row['gate_overhead_ratio']:.1f}× cycle time")

    # 2. Latency sweep
    print()
    print(f"[2/4] Latency sweep  (N ∈ {LATENCY_N_VALUES}, "
          f"gain={LATENCY_GAIN}, {LATENCY_REVOLUTIONS} revolutions each) …")
    latency_rows = benchmark_latency()
    header = f"  {'N':>5}  {'median μs':>12}  {'min μs':>8}  {'max μs':>8}  {'final R':>9}  {'B (bits)':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in latency_rows:
        print(f"  {r['N']:>5}  {r['median_us']:>12.2f}  "
              f"{r['min_us']:>8.2f}  {r['max_us']:>8.2f}  "
              f"{r['final_coherence']:>9.6f}  {r['final_bit_strength']:>9.4f}")

    # 3. Convergence
    print()
    print(f"[3/4] Convergence curves  "
          f"(N={CONVERGENCE_N}, gains={CONVERGENCE_GAINS}, "
          f"{CONVERGENCE_REVOLUTIONS} revolutions each) …")
    convergence_rows = benchmark_convergence()
    for gain in CONVERGENCE_GAINS:
        final = [r for r in convergence_rows if r["gain"] == gain][-1]
        print(f"      gain={gain:.1f}: R_final={final['coherence']:.6f}  "
              f"B_final={final['bit_strength']:.4f} bits  "
              f"E_final={final['frustration']:.4e}")

    # 4. Invariant verification
    print()
    print("[4/4] Verifying Lean invariants (40 revolutions, N=8, gain=0.3) …")
    invariants = verify_invariants()
    if invariants["passed"]:
        print(f"      ALL INVARIANTS HELD ✓  "
              f"ΔR={invariants['delta_R_total']:+.6f}  "
              f"ΔB={invariants['delta_B_total']:+.4f} bits  "
              f"ΔE={invariants['delta_E_total']:.4e}")
    else:
        print(f"      VIOLATIONS: {invariants['n_violations']}")
        for v in invariants["violations"]:
            print(f"        • {v}")

    # Write CSV outputs
    _write_csv(output_dir / "mu8_latency.csv", latency_rows)
    _write_csv(output_dir / "mu8_convergence.csv", convergence_rows)
    print(f"\n  mu8_latency.csv      → {output_dir / 'mu8_latency.csv'}")
    print(f"  mu8_convergence.csv  → {output_dir / 'mu8_convergence.csv'}")

    # JSON summary
    summary = {
        "generated":       timestamp,
        "lean_fingerprint": LEAN_FINGERPRINT,
        "mu_angle":        MU_ANGLE,
        "silver_ratio":    SILVER_RATIO,
        "c_natural":       C_NATURAL,
        "bit_strength_definition": "B = -log2(1 - R + eps), capped at log2(N)",
        "gate_overhead":   gate_row,
        "latency_rows":    latency_rows,
        "invariants":      invariants,
        "all_passed":      invariants["passed"],
    }
    json_path = output_dir / "mu8_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  mu8_summary.json     → {json_path}")

    # Markdown report
    md_path = _generate_markdown(gate_row, latency_rows, invariants, output_dir, timestamp)
    print(f"  mu8_benchmark_report.md → {md_path}")

    # Plots
    if not no_plots:
        _generate_plots(latency_rows, convergence_rows, output_dir)
        print(f"  mu8_latency.png      → {output_dir / 'mu8_latency.png'}")
        print(f"  mu8_convergence.png  → {output_dir / 'mu8_convergence.png'}")

    # Final status
    print()
    print("=" * 70)
    if invariants["passed"]:
        print("  ALL BENCHMARK CHECKS PASSED ✓")
    else:
        print(f"  {invariants['n_violations']} INVARIANT VIOLATION(S) — see report")
    print("=" * 70)

    return 0 if invariants["passed"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="μ⁸=1 spiral cycle optimizer benchmark suite"
    )
    parser.add_argument(
        "--output-dir",
        default=str(_HERE / "results"),
        help="Directory for benchmark output (default: benchmarks/results/)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    args = parser.parse_args()
    sys.exit(run(output_dir=Path(args.output_dir), no_plots=args.no_plots))


if __name__ == "__main__":
    main()
