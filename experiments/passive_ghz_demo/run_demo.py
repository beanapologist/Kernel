#!/usr/bin/env python3
"""
run_demo.py — End-to-end Passive GHz Demo simulation.

Since no hardware is present, this script:
  1. Generates I/Q waveforms (palindromic and control variants).
  2. Synthesises realistic ringdown traces for all three runs:
       baseline    — reference ringdown with κ_base
       pal         — post-palindromic-drive ringdown with κ_pal
       ctrl        — post-control-drive ringdown with κ_ctrl
  3. Runs the full analysis pipeline (κ extraction, Δκ, plots).
  4. Writes a Markdown summary report: demo_report.md

Synthetic model
---------------
  β(t) = β0 · exp(−κ/2 · t) · exp(i ω_res t) + noise
  where ω_res is a small detuning added to make the oscillation visible.
  Noise ~ CN(0, σ) with σ = 0.01 × β0.

Physical parameters (representative superconducting resonator)
  κ_base  = 2π × 1.00 MHz   (10 µs ring-down time)
  κ_pal   = 2π × 1.02 MHz   (+2 % shift — expected to be ~0 if palindromic
                              drives are energy-neutral, but simulated nonzero
                              to show Δκ detection capability)
  κ_ctrl  = 2π × 1.01 MHz   (+1 % shift from drive heating)
  ω_res   = 2π × 0.05 MHz   (small residual detuning after IQ mixing)
  β0      = 1.0
  σ       = 0.01

Recording parameters
  fs_rec  = 500 MHz
  T_rec   = 10 µs

Usage
-----
  python run_demo.py [--seed N] [--output-dir DIR]

Outputs (written to output_dir, default = 'demo_output/')
  waveforms.npz
  ringdown_baseline.npz
  ringdown_pal.npz
  ringdown_ctrl.npz
  ringdown_fits.png
  kappa_summary.png
  phase_tracking.png
  demo_report.md
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warn] matplotlib not found — skipping plots.")

# ---------------------------------------------------------------------------
# Bring the experiment package into scope
# ---------------------------------------------------------------------------
DEMO_DIR = Path(__file__).parent
sys.path.insert(0, str(DEMO_DIR))

from generate_waveforms import (          # noqa: E402
    build_waveforms,
    compute_phase_schedule,
    SAMPLES_PER_STEP,
    D,
    MU_CYCLE,
    FS,
    T_STEP,
)

# ---------------------------------------------------------------------------
# Physical constants for simulation
#
# κ/(2π) = 100 kHz  →  field amplitude decay time τ = 2/κ ≈ 3.18 µs.
# With T_rec = 15 µs we capture ≈ 4.7 τ, giving a clean exponential fit.
# Variants differ by +2 % (pal) and +1 % (ctrl) to demonstrate Δκ detection.
# ---------------------------------------------------------------------------
KAPPA_BASE = 2 * math.pi * 0.100e6   # rad/s  — baseline decay rate
KAPPA_PAL  = 2 * math.pi * 0.102e6   # rad/s  — post-palindromic-drive  (+2 %)
KAPPA_CTRL = 2 * math.pi * 0.101e6   # rad/s  — post-control-drive       (+1 %)
OMEGA_RES  = 2 * math.pi * 0.010e6   # rad/s  — residual detuning (10 kHz)
BETA0      = 1.0                      # initial field amplitude
NOISE_SIGMA = 0.005                   # complex noise std (per quadrature, SNR≈46 dB)

FS_REC   = 50e6     # recording sample rate, Hz  (50 MS/s is ample for 100 kHz κ)
T_REC    = 15e-6    # recording window duration, seconds  (≈ 4.7 τ)


# ---------------------------------------------------------------------------
# Synthetic trace generator
# ---------------------------------------------------------------------------

def generate_ringdown(
    kappa: float,
    rng: np.random.Generator,
    fs_rec: float = FS_REC,
    t_rec: float = T_REC,
    beta0: float = BETA0,
    omega_res: float = OMEGA_RES,
    noise_sigma: float = NOISE_SIGMA,
) -> dict:
    """
    Synthesise a noisy IQ ringdown trace.

    β(t) = β0 · exp(−κ/2 · t) · exp(i ω_res t)  +  CN noise
    """
    n_samples = round(fs_rec * t_rec)
    t = np.arange(n_samples) / fs_rec
    envelope = beta0 * np.exp(-0.5 * kappa * t)
    phase = omega_res * t
    beta_clean = envelope * (np.cos(phase) + 1j * np.sin(phase))
    noise = rng.normal(0, noise_sigma, n_samples) + 1j * rng.normal(0, noise_sigma, n_samples)
    beta_noisy = beta_clean + noise
    return {
        "t":     t,
        "I_rec": beta_noisy.real,
        "Q_rec": beta_noisy.imag,
        "fs_rec": fs_rec,
        "kappa_true": kappa,
    }


# ---------------------------------------------------------------------------
# κ fitting (self-contained copy so run_demo.py has no import dependency on
# analysis_ringdown.py — users can run it standalone)
# ---------------------------------------------------------------------------

def fit_kappa(
    t: np.ndarray,
    beta: np.ndarray,
    t_start: float,
    t_end: float,
) -> tuple[float, float]:
    mask = (t >= t_start) & (t <= t_end)
    t_w = t[mask]
    log_amp = np.log(np.abs(beta[mask]) + 1e-30)
    A_mat = np.column_stack([t_w, np.ones(len(t_w))])
    result = np.linalg.lstsq(A_mat, log_amp, rcond=None)
    slope, _ = result[0]
    kappa = -2.0 * slope
    residuals = log_amp - A_mat @ result[0]
    n = len(t_w)
    sigma2 = np.sum(residuals ** 2) / max(n - 2, 1)
    t_var = np.sum((t_w - t_w.mean()) ** 2)
    slope_se = math.sqrt(sigma2 / max(t_var, 1e-30))
    return float(kappa), float(2.0 * slope_se)


def bootstrap_kappa(
    t: np.ndarray,
    beta: np.ndarray,
    t_start: float,
    t_end: float,
    n_boot: int = 500,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(99)
    mask = (t >= t_start) & (t <= t_end)
    t_w, b_w = t[mask], beta[mask]
    n = len(t_w)
    ks = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            k, _ = fit_kappa(t_w[idx], b_w[idx], t_w[idx].min(), t_w[idx].max())
            ks.append(k)
        except Exception:
            pass
    ks = np.array(ks)
    return float(np.nanmean(ks)), float(np.nanstd(ks))


# ---------------------------------------------------------------------------
# Phase tracking analysis
# ---------------------------------------------------------------------------

def phase_tracking_analysis(
    phi_per_step: np.ndarray,
    mu_increment: float = 3 * math.pi / 4,
) -> tuple[np.ndarray, float]:
    """
    Compute step-synchronous phase tracking error from the generated schedule.
    Compares actual Δφ between consecutive steps to the expected μ increment.
    """
    phase_diff = np.diff(phi_per_step)
    error = (phase_diff - mu_increment + math.pi) % (2 * math.pi) - math.pi
    return error, float(np.sqrt(np.mean(error ** 2)))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_ringdown_fits(runs: dict, t_fit_end: float, out_path: str) -> None:
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    colors = {"baseline": "#1f77b4", "pal": "#d62728", "ctrl": "#2ca02c"}
    for ax, (label, info) in zip(axes, runs.items()):
        t, beta, kappa_fit, kappa_true = (
            info["t"], info["beta"], info["kappa_fit"], info["kappa_true"]
        )
        log_amp = np.log(np.abs(beta) + 1e-30)
        ax.plot(t * 1e6, log_amp, color=colors[label], alpha=0.55, lw=0.8, label="data")
        mask = t <= t_fit_end
        slope = -kappa_fit / 2.0
        intercept = np.mean(log_amp[mask]) - slope * np.mean(t[mask])
        ax.plot(
            t[mask] * 1e6, slope * t[mask] + intercept,
            "k--", lw=1.5, label=f"fit  κ={kappa_fit/1e6:.3f} Mrad/s",
        )
        ax.axvline(t_fit_end * 1e6, color="orange", lw=1, ls=":", label="fit end")
        ax.set_title(f"{label}  (true κ={kappa_true/1e6:.3f})", fontsize=10)
        ax.set_xlabel("Time (µs)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("log|β|")
    fig.suptitle("Ringdown fits — Passive GHz Demo (synthetic data)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_kappa_summary(kappa_results: dict, out_path: str) -> None:
    if not HAS_MPL:
        return
    labels = list(kappa_results.keys())
    vals   = [kappa_results[k]["mean"] / 1e6 for k in labels]
    errs   = [kappa_results[k]["std"]  / 1e6 for k in labels]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, vals, yerr=errs, capsize=8,
                  color=colors[: len(labels)], alpha=0.85, ecolor="black")
    ax.set_ylabel("κ (Mrad/s)")
    ax.set_title("κ summary — post-drive ringdown (synthetic data)")
    ax.axhline(KAPPA_BASE / 1e6, color="grey", ls="--", lw=1, label="κ baseline")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v, e in zip(bars, vals, errs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + e + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_phase_tracking(error_pal: np.ndarray, error_ctrl: np.ndarray, out_path: str) -> None:
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    n_pal  = len(error_pal)
    n_ctrl = len(error_ctrl)
    axes[0].plot(range(n_pal),  np.rad2deg(error_pal),  color="#d62728", lw=0.8, alpha=0.8)
    axes[0].axhline(0, color="k", lw=0.8, ls="--")
    axes[0].set_ylabel("Phase error (deg)")
    axes[0].set_title("Step-synchronous phase tracking error — palindromic")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(n_ctrl), np.rad2deg(error_ctrl), color="#2ca02c", lw=0.8, alpha=0.8)
    axes[1].axhline(0, color="k", lw=0.8, ls="--")
    axes[1].set_ylabel("Phase error (deg)")
    axes[1].set_xlabel("Step index")
    axes[1].set_title("Step-synchronous phase tracking error — control")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Phase tracking (ideal synthesised waveform)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_waveform_preview(I_pal, Q_pal, I_ctrl, Q_ctrl, n_steps_show: int, out_path: str) -> None:
    if not HAS_MPL:
        return
    sps = SAMPLES_PER_STEP
    n_show = n_steps_show * sps
    t_ns = np.arange(n_show) / FS * 1e9

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].plot(t_ns, I_pal[:n_show],  color="#d62728", lw=0.8, label="I pal", alpha=0.9)
    axes[0].plot(t_ns, Q_pal[:n_show],  color="#ff7f0e", lw=0.8, label="Q pal", alpha=0.9, ls="--")
    axes[0].plot(t_ns, I_ctrl[:n_show], color="#2ca02c", lw=0.8, label="I ctrl", alpha=0.6)
    axes[0].plot(t_ns, Q_ctrl[:n_show], color="#17becf", lw=0.8, label="Q ctrl", alpha=0.6, ls="--")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Waveform preview — first {n_steps_show} steps ({n_show} samples)")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)

    amp_pal  = np.sqrt(I_pal[:n_show]**2  + Q_pal[:n_show]**2)
    amp_ctrl = np.sqrt(I_ctrl[:n_show]**2 + Q_ctrl[:n_show]**2)
    axes[1].plot(t_ns, amp_pal,  color="#d62728", lw=0.8, label="|ε| pal")
    axes[1].plot(t_ns, amp_ctrl, color="#2ca02c", lw=0.8, label="|ε| ctrl", ls="--")
    axes[1].set_ylabel("Drive amplitude |ε|")
    axes[1].set_xlabel("Time (ns)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(
    kappa_results: dict,
    delta_kappa: float,
    delta_kappa_se: float,
    n_steps: int,
    a_drive: float,
    rms_phase_pal: float,
    rms_phase_ctrl: float,
    plots: list[str],
    out_path: str,
) -> None:
    sep = "-" * 60
    lines = [
        "# Passive GHz Demo — Drive-only Variant 1",
        "## Synthetic Demo Report",
        "",
        f"*Generated by `run_demo.py` — synthetic data, no hardware required.*",
        "",
        sep,
        "## Simulation parameters",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| f_s (AWG) | 2.4 GS/s |",
        f"| T_step    | 40 ns |",
        f"| Samples/step | {SAMPLES_PER_STEP} |",
        f"| N_steps   | {n_steps} |",
        f"| A_drive   | {a_drive} |",
        f"| D (precession period) | {D:,} steps |",
        f"| κ_base (true) | {KAPPA_BASE/1e6:.4f} Mrad/s |",
        f"| κ_pal  (true) | {KAPPA_PAL/1e6:.4f} Mrad/s |",
        f"| κ_ctrl (true) | {KAPPA_CTRL/1e6:.4f} Mrad/s |",
        f"| Noise σ   | {NOISE_SIGMA} (per quadrature) |",
        f"| f_s (rec) | {FS_REC/1e6:.0f} MS/s |",
        f"| T_rec     | {T_REC*1e6:.0f} µs |",
        "",
        sep,
        "## κ extraction results",
        "",
        "| Run | κ fitted (Mrad/s) | ± (bootstrap) | κ true (Mrad/s) |",
        "|-----|-------------------|---------------|-----------------|",
    ]
    for label, res in kappa_results.items():
        k_true = {"baseline": KAPPA_BASE, "pal": KAPPA_PAL, "ctrl": KAPPA_CTRL}[label]
        lines.append(
            f"| {label:<8} | {res['mean']/1e6:>17.4f} | "
            f"{res['std']/1e6:>13.4f} | {k_true/1e6:>15.4f} |"
        )
    lines += [
        "",
        f"**Δκ (pal − ctrl)** = {delta_kappa/1e6:.4f} ± {delta_kappa_se/1e6:.4f} Mrad/s",
        f"  (true Δκ = {(KAPPA_PAL - KAPPA_CTRL)/1e6:.4f} Mrad/s)",
        "",
        sep,
        "## Phase tracking",
        "",
        f"| Variant | RMS phase error |",
        f"|---------|-----------------|",
        f"| palindromic | {math.degrees(rms_phase_pal):.4f} ° |",
        f"| control     | {math.degrees(rms_phase_ctrl):.4f} ° |",
        "",
        "> **Note**: phase errors are identically zero for an ideal synthesised waveform.",
        "> Non-zero errors in real hardware arise from mixer nonlinearity and DAC distortion.",
        "",
        sep,
        "## Output plots",
        "",
    ]
    for p in plots:
        name = Path(p).name
        lines.append(f"- `{name}`")
    lines += [
        "",
        sep,
        "## Physical interpretation",
        "",
        "- **κ is extracted only from the post-drive ringdown** (drive OFF).",
        "- Constant drive amplitude |ε| does **not** imply constant |β| inside the",
        "  resonator — β depends on drive history, detuning, and κ.",
        "- In this simulation Δκ > 0 by construction; on real hardware, palindromic",
        "  phase drives are designed to be energy-neutral (Δκ ≈ 0 is the null hypothesis).",
        "- Bootstrap error bars should shrink with repeated traces.",
        "",
        "*End of demo report.*",
    ]
    text = "\n".join(lines) + "\n"
    Path(out_path).write_text(text)
    print(f"  Saved {out_path}")
    # Also print to stdout
    print()
    print(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run Passive GHz Demo end-to-end with synthetic traces."
    )
    parser.add_argument("--seed",       type=int,   default=42,            help="RNG seed (default 42).")
    parser.add_argument("--n-steps",    type=int,   default=200,           help="AWG steps (default 200).")
    parser.add_argument("--a-drive",    type=float, default=0.5,           help="Drive amplitude (default 0.5).")
    parser.add_argument("--output-dir", type=str,   default="demo_output", help="Output directory.")
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 60)
    print("  Passive GHz Demo — Drive-only Variant 1  (synthetic)")
    print("=" * 60)

    # 1. Generate waveforms
    print("\n[1/5] Generating waveforms …")
    wf_path = out / "waveforms.npz"
    wf_data = build_waveforms(n_steps=args.n_steps, a_drive=args.a_drive)
    np.savez(str(wf_path), **wf_data)
    print(f"  Saved {wf_path}  ({args.n_steps} steps × {SAMPLES_PER_STEP} samples)")

    # 2. Generate synthetic ringdown traces
    print("\n[2/5] Generating synthetic ringdown traces …")
    traces: dict[str, dict] = {}
    kappa_map = {"baseline": KAPPA_BASE, "pal": KAPPA_PAL, "ctrl": KAPPA_CTRL}
    for run, kappa in kappa_map.items():
        tr = generate_ringdown(kappa, rng)
        tr_path = out / f"ringdown_{run}.npz"
        np.savez(str(tr_path), **{k: v for k, v in tr.items() if k != "kappa_true"})
        traces[run] = tr
        print(f"  {run:>10}: κ_true={kappa/1e6:.4f} Mrad/s  →  {tr_path.name}")

    # 3. Fit κ from ringdown traces
    print("\n[3/5] Fitting κ …")
    t_fit_start = 0.0
    t_fit_end   = 12e-6   # 12 µs ≈ 3.8 τ for κ/(2π)=100 kHz
    kappa_results: dict[str, dict] = {}
    runs_plot: dict = {}

    for run, tr in traces.items():
        t    = tr["t"]
        beta = tr["I_rec"] + 1j * tr["Q_rec"]
        k_fit, k_se = fit_kappa(t, beta, t_fit_start, t_fit_end)
        k_bs, k_bs_std = bootstrap_kappa(t, beta, t_fit_start, t_fit_end, n_boot=500, rng=rng)
        kappa_results[run] = {"mean": k_bs, "std": k_bs_std}
        runs_plot[run] = {"t": t, "beta": beta, "kappa_fit": k_fit, "kappa_true": tr["kappa_true"]}
        print(f"  {run:>10}: κ_fit={k_fit/1e6:.4f}  κ_bs={k_bs/1e6:.4f} ± {k_bs_std/1e6:.4f} Mrad/s")

    k_pal,  se_pal  = kappa_results["pal"]["mean"],  kappa_results["pal"]["std"]
    k_ctrl, se_ctrl = kappa_results["ctrl"]["mean"], kappa_results["ctrl"]["std"]
    delta_kappa    = k_pal - k_ctrl
    delta_kappa_se = math.sqrt(se_pal**2 + se_ctrl**2)
    print(f"\n  Δκ (pal − ctrl) = {delta_kappa/1e6:.4f} ± {delta_kappa_se/1e6:.4f} Mrad/s")
    print(f"  (true Δκ        = {(KAPPA_PAL - KAPPA_CTRL)/1e6:.4f} Mrad/s)")

    # 4. Phase tracking
    print("\n[4/5] Phase tracking analysis …")
    phi_pal  = wf_data["phi_per_step_pal"]
    phi_ctrl = wf_data["phi_per_step_ctrl"]
    err_pal,  rms_pal  = phase_tracking_analysis(phi_pal)
    err_ctrl, rms_ctrl = phase_tracking_analysis(phi_ctrl)
    print(f"  RMS phase error — pal:  {math.degrees(rms_pal):.6f} °")
    print(f"  RMS phase error — ctrl: {math.degrees(rms_ctrl):.6f} °")

    # 5. Plots and report
    print("\n[5/5] Generating plots and report …")
    plots_generated = []

    p_ringdown = str(out / "ringdown_fits.png")
    plot_ringdown_fits(runs_plot, t_fit_end, p_ringdown)
    if HAS_MPL:
        plots_generated.append(p_ringdown)

    p_kappa = str(out / "kappa_summary.png")
    plot_kappa_summary(kappa_results, p_kappa)
    if HAS_MPL:
        plots_generated.append(p_kappa)

    p_phase = str(out / "phase_tracking.png")
    plot_phase_tracking(err_pal, err_ctrl, p_phase)
    if HAS_MPL:
        plots_generated.append(p_phase)

    p_wf = str(out / "waveform_preview.png")
    plot_waveform_preview(
        wf_data["I_pal"], wf_data["Q_pal"],
        wf_data["I_ctrl"], wf_data["Q_ctrl"],
        n_steps_show=min(32, args.n_steps),
        out_path=p_wf,
    )
    if HAS_MPL:
        plots_generated.append(p_wf)

    report_path = str(out / "demo_report.md")
    write_report(
        kappa_results=kappa_results,
        delta_kappa=delta_kappa,
        delta_kappa_se=delta_kappa_se,
        n_steps=args.n_steps,
        a_drive=args.a_drive,
        rms_phase_pal=rms_pal,
        rms_phase_ctrl=rms_ctrl,
        plots=plots_generated,
        out_path=report_path,
    )

    print()
    print("=" * 60)
    print("  Demo complete.  Outputs in:", out)
    print("=" * 60)


if __name__ == "__main__":
    main()
