#!/usr/bin/env python3
"""
plot_scaling.py — Visualise Kernel scaling falsification results.

Reads CSV files produced by benchmark_scaling_falsification and generates:
  1. loglog_coh_vs_n.png   — convergence rounds vs N on log-log axes
  2. speedup_vs_n.png      — speedup ratio vs N (semi-log)
  3. adversarial.png       — adversarial config comparison
  4. multi_seed_ci.png     — histogram of exponents over 20 seeds
  5. norm_inspection.png   — |β|, r_eff, G_eff over steps

Usage (one command from the repo root after running the benchmark):
    python3 plot_scaling.py

All PNGs are written to the current directory.
"""

import csv
import math
import os
import sys

# ---------------------------------------------------------------------------
# Lightweight plotting that works without matplotlib (falls back to ASCII).
# If matplotlib is available, use it; otherwise emit ASCII art tables.
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _read_csv(path):
    """Return list-of-dicts from a CSV file; empty list if file missing."""
    if not os.path.isfile(path):
        print(f"  [warn] {path} not found — skipping plot", file=sys.stderr)
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# 1. Log-log: convergence rounds vs N
# ---------------------------------------------------------------------------
def plot_loglog(csv_path="loglog_plot.csv", out="loglog_coh_vs_n.png"):
    rows = _read_csv(csv_path)
    if not rows:
        return

    log2_ns   = [float(r["log2_n"])       for r in rows]
    log2_cohs = [float(r["log2_coh_avg"]) for r in rows]
    log2_brts = [float(r["log2_brute_avg"]) for r in rows]
    log2_sqrts = [float(r["log2_sqrt_n"]) for r in rows]

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(log2_ns, log2_cohs,  "o-", label="Coherent search")
        ax.plot(log2_ns, log2_brts,  "s--", label="Brute-force (O(n))")
        ax.plot(log2_ns, log2_sqrts, "k:", label="√n reference")
        ax.set_xlabel("log₂(N)")
        ax.set_ylabel("log₂(convergence rounds)")
        ax.set_title("Convergence Rounds vs N  (log–log)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  ✓ {out}")
    else:
        # ASCII fallback
        print(f"\n  [ASCII] log-log plot ({csv_path}):")
        print(f"  {'log2_n':>8}  {'log2_coh':>10}  {'log2_brute':>12}  {'log2_sqrt':>10}")
        for r in rows:
            print(f"  {float(r['log2_n']):8.2f}  "
                  f"{float(r['log2_coh_avg']):10.4f}  "
                  f"{float(r['log2_brute_avg']):12.4f}  "
                  f"{float(r['log2_sqrt_n']):10.4f}")


# ---------------------------------------------------------------------------
# 2. Speedup vs N (semi-log x-axis)
# ---------------------------------------------------------------------------
def plot_speedup(csv_path="scaling_extended.csv", out="speedup_vs_n.png"):
    rows = _read_csv(csv_path)
    if not rows:
        return

    ns      = [float(r["n"])       for r in rows]
    speedups = [float(r["speedup"]) for r in rows]
    sqrts   = [math.sqrt(n) for n in ns]
    # Normalised speedup / sqrt(n)
    ratios  = [sp / math.sqrt(n) for sp, n in zip(speedups, ns)]

    if HAS_MPL:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.semilogx(ns, speedups, "o-")
        # Theoretical constant c ≈ 2.6 from Dirichlet-kernel resonance analysis
        # (see test_coherent_search.cpp §9 Constant-Factor Analysis).
        ax1.semilogx(ns, [2.6 * math.sqrt(n) for n in ns], "k:", label="2.6·√n theory")
        ax1.set_xlabel("N")
        ax1.set_ylabel("Speedup = brute_avg / coh_avg")
        ax1.set_title("Speedup vs N")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.semilogx(ns, ratios, "s-")
        # Theoretical constant: speedup/√N ≈ 2.6 (brute_avg/coh_avg / √N)
        # from Dirichlet-kernel analysis in test_coherent_search.cpp §9.
        ax2.axhline(y=2.6, color="k", linestyle=":", label="theory ≈ 2.6")
        ax2.set_xlabel("N")
        ax2.set_ylabel("speedup / √N")
        ax2.set_title("Normalised speedup (should ≈ const)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  ✓ {out}")
    else:
        print(f"\n  [ASCII] speedup table ({csv_path}):")
        print(f"  {'n':>12}  {'speedup':>10}  {'speedup/√n':>12}")
        for n, sp, rt in zip(ns, speedups, ratios):
            print(f"  {n:12.0f}  {sp:10.2f}  {rt:12.4f}")


# ---------------------------------------------------------------------------
# 3. Adversarial configurations
# ---------------------------------------------------------------------------
def plot_adversarial(csv_path="adversarial.csv", out="adversarial.png"):
    rows = _read_csv(csv_path)
    if not rows:
        return

    configs = sorted(set(r["config"] for r in rows))
    data = {c: [] for c in configs}
    log2_ns_seen = []

    # Group by config, keeping log2_n order
    log2_ns_map = {}
    for r in rows:
        k = int(r["log2_n"])
        c = r["config"]
        data[c].append((k, float(r["mean_steps"])))
        if k not in log2_ns_map:
            log2_ns_map[k] = True
            log2_ns_seen.append(k)
    log2_ns_seen = sorted(set(log2_ns_seen))

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(8, 5))
        markers = {"clustered": "o", "boundary": "s", "worst_case": "^"}
        for c in configs:
            pts = sorted(data[c])
            xs = [float(k) for k, _ in pts]   # log₂(2^k) = k by definition
            ys = [math.log2(v) for _, v in pts]
            ax.plot(xs, ys, marker=markers.get(c, "x"), label=c)

        # √n reference
        xs_ref = [float(k) for k in log2_ns_seen]
        ax.plot(xs_ref, [0.5 * x for x in xs_ref], "k:", label="√n reference")
        ax.set_xlabel("log₂(N)")
        ax.set_ylabel("log₂(mean steps)")
        ax.set_title("Adversarial Target Configurations")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  ✓ {out}")
    else:
        print(f"\n  [ASCII] adversarial configs ({csv_path}):")
        print(f"  {'log2_n':>8}  {'config':>12}  {'mean_steps':>12}")
        for r in rows:
            print(f"  {int(r['log2_n']):8d}  {r['config']:>12}  "
                  f"{float(r['mean_steps']):12.2f}")


# ---------------------------------------------------------------------------
# 4. Multi-seed CI histogram
# ---------------------------------------------------------------------------
def plot_multi_seed_ci(csv_path="multi_seed_ci.csv",
                       out="multi_seed_ci.png"):
    rows = _read_csv(csv_path)
    if not rows:
        return

    slopes = []
    for r in rows:
        try:
            v = float(r.get("slope") or list(r.values())[1])
            if 0.0 < v < 2.0:
                slopes.append(v)
        except (ValueError, IndexError):
            pass

    if not slopes:
        return

    mean_s = sum(slopes) / len(slopes)
    std_s = math.sqrt(sum((s - mean_s) ** 2 for s in slopes) / max(len(slopes) - 1, 1))

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(slopes, bins=10, edgecolor="black", alpha=0.7)
        ax.axvline(x=0.5,  color="red",    linestyle="--", label="target 0.5")
        ax.axvline(x=0.45, color="orange", linestyle=":",  label="CI bounds [0.45, 0.55]")
        ax.axvline(x=0.55, color="orange", linestyle=":")
        ax.axvline(x=mean_s, color="blue", linestyle="-", label=f"mean={mean_s:.4f}")
        ax.set_xlabel("Scaling exponent (slope)")
        ax.set_ylabel("Count")
        ax.set_title(f"Multi-seed exponent distribution (n={len(slopes)})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  ✓ {out}")
    else:
        print(f"\n  [ASCII] multi-seed slopes: mean={mean_s:.4f} std={std_s:.4f}")
        print(f"  slopes: {[f'{s:.4f}' for s in slopes]}")


# ---------------------------------------------------------------------------
# 5. Normalisation inspection
# ---------------------------------------------------------------------------
def plot_norm_inspection(csv_path="norm_inspection.csv",
                         out="norm_inspection.png"):
    rows = _read_csv(csv_path)
    if not rows:
        return

    steps    = [int(r["step"])         for r in rows]
    beta_mags = [float(r["beta_mag"])  for r in rows]
    r_effs   = [float(r["r_eff"])      for r in rows]
    g_effs   = [float(r["g_eff"])      for r in rows]

    if HAS_MPL:
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

        axes[0].plot(steps, beta_mags)
        axes[0].set_ylabel("|β|")
        axes[0].set_title("Normalisation Inspection — no clipping expected")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(steps, r_effs)
        axes[1].axhline(y=1.0, color="r", linestyle=":", label="ideal r_eff=1")
        axes[1].set_ylabel("r_eff")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(steps, g_effs)
        axes[2].axhline(y=1.0, color="r", linestyle=":", label="ideal G_eff=1")
        axes[2].set_ylabel("G_eff = 1/r_eff")
        axes[2].set_xlabel("Step")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  ✓ {out}")
    else:
        print(f"\n  [ASCII] first 10 normalisation steps ({csv_path}):")
        print(f"  {'step':>6}  {'beta_mag':>14}  {'r_eff':>10}  {'g_eff':>10}")
        for r in rows[:10]:
            print(f"  {int(r['step']):6d}  {float(r['beta_mag']):14.10f}  "
                  f"{float(r['r_eff']):10.8f}  {float(r['g_eff']):10.8f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  plot_scaling.py — Kernel Falsification Visualiser  ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    if not HAS_MPL:
        print("  [info] matplotlib not found — emitting ASCII summaries only.\n"
              "         Install matplotlib for PNG output: pip install matplotlib\n")

    plot_loglog()
    plot_speedup()
    plot_adversarial()
    plot_multi_seed_ci()
    plot_norm_inspection()

    print("\n  Done.  PNG files (if matplotlib available):")
    for f in ["loglog_coh_vs_n.png", "speedup_vs_n.png", "adversarial.png",
              "multi_seed_ci.png", "norm_inspection.png"]:
        status = "present" if os.path.isfile(f) else "not generated"
        print(f"    {f:<30} [{status}]")
    print()


if __name__ == "__main__":
    main()
