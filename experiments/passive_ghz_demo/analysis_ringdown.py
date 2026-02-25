#!/usr/bin/env python3
"""
analysis_ringdown.py — Extract κ and phase-tracking metrics.

Loads recorded IQ ringdown traces (NPZ or CSV), computes the intra-cavity
amplitude β(t), fits the energy-decay rate κ from log|β| vs time, and
computes Δκ for palindromic vs control vs baseline.

Expected input file format (NPZ):
  t      — time axis, seconds, shape (N,)
  I_rec  — recorded I channel, shape (N,)
  Q_rec  — recorded Q channel, shape (N,)

Optional keys in NPZ (or supply via --config):
  I_lo_cal, Q_lo_cal  — LO leakage offsets (subtracted from I_rec, Q_rec)
  phi_imbal           — IQ phase imbalance in radians

The script produces:
  ringdown_fits.png       — log|β| vs time with linear fits
  kappa_summary.png       — bar chart of κ values
  phase_tracking_error.png— step-synchronous phase error (drive window)
  amplitude_modulation.png— |β(t)| modulation depth (drive window)

USAGE
-----
  python analysis_ringdown.py \\
      --baseline ringdown_baseline.npz \\
      --pal      ringdown_pal.npz \\
      --ctrl     ringdown_ctrl.npz \\
      --config   config_example.yaml \\
      [--ringdown-start 1e-6] [--ringdown-end 5e-6] \\
      [--n-bootstrap 500]
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ---------------------------------------------------------------------------
# Calibration defaults
# ---------------------------------------------------------------------------
DEFAULT_RINGDOWN_START = 0.0        # seconds after drive-off
DEFAULT_RINGDOWN_END = 5e-6         # seconds after drive-off
DEFAULT_N_BOOTSTRAP = 500


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_trace(path: str) -> dict:
    """Load a trace from NPZ or CSV.  Returns dict with t, I_rec, Q_rec."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        return {"t": data[:, 0], "I_rec": data[:, 1], "Q_rec": data[:, 2]}
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data}


def load_config(config_path: str) -> dict:
    if not HAS_YAML:
        print("[warn] pyyaml not installed — using default calibration values.")
        return {}
    with open(config_path) as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def apply_iq_calibration(
    I_rec: np.ndarray,
    Q_rec: np.ndarray,
    i_offset: float = 0.0,
    q_offset: float = 0.0,
    phi_imbal: float = 0.0,
    amplitude_imbal: float = 1.0,
) -> np.ndarray:
    """
    Convert raw IQ samples to complex β(t) after applying calibration.

    Parameters
    ----------
    I_rec, Q_rec     : raw digitiser samples
    i_offset         : LO leakage on I port
    q_offset         : LO leakage on Q port
    phi_imbal        : IQ phase imbalance (radians)
    amplitude_imbal  : amplitude ratio Q/I (dimensionless)

    Returns
    -------
    beta : complex128 array
    """
    I_cal = I_rec - i_offset
    Q_cal = (Q_rec - q_offset) * amplitude_imbal
    # Correct phase imbalance: rotate Q by -phi_imbal
    I_corr = I_cal - Q_cal * math.sin(phi_imbal)
    Q_corr = Q_cal * math.cos(phi_imbal)
    return I_corr + 1j * Q_corr


# ---------------------------------------------------------------------------
# κ fitting
# ---------------------------------------------------------------------------

def fit_kappa(
    t: np.ndarray,
    beta: np.ndarray,
    t_start: float,
    t_end: float,
) -> tuple[float, float]:
    """
    Fit κ from the post-drive ringdown via linear regression on log|β|.

    The intra-cavity energy decays as |β(t)|² ∝ exp(−κ t), so
    log|β| = −(κ/2) t + const.

    Parameters
    ----------
    t              : time axis (seconds), monotonically increasing from 0
    beta           : complex intra-cavity field
    t_start, t_end : ringdown window relative to t[0]

    Returns
    -------
    kappa    : energy-decay rate (rad/s)
    kappa_se : standard error from linear regression
    """
    mask = (t >= t_start) & (t <= t_end)
    if mask.sum() < 4:
        raise ValueError(
            f"Ringdown window [{t_start:.2e}, {t_end:.2e}] s contains "
            f"only {mask.sum()} points.  Adjust --ringdown-start/end."
        )
    t_win = t[mask]
    log_amp = np.log(np.abs(beta[mask]) + 1e-30)

    # Linear regression: log_amp = slope * t_win + intercept
    A = np.column_stack([t_win, np.ones(len(t_win))])
    result = np.linalg.lstsq(A, log_amp, rcond=None)
    slope, _intercept = result[0]
    kappa = -2.0 * slope   # κ = −2 × d(log|β|)/dt

    # Standard error via residuals
    residuals = log_amp - (A @ result[0])
    n = len(t_win)
    sigma2 = np.sum(residuals**2) / max(n - 2, 1)
    # Var(slope) = sigma2 / sum((t - t_mean)^2)
    t_var = np.sum((t_win - t_win.mean()) ** 2)
    slope_se = math.sqrt(sigma2 / max(t_var, 1e-30))
    kappa_se = 2.0 * slope_se

    return float(kappa), float(kappa_se)


def bootstrap_kappa(
    t: np.ndarray,
    beta: np.ndarray,
    t_start: float,
    t_end: float,
    n_bootstrap: int = 500,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Bootstrap estimate of κ uncertainty.

    Returns
    -------
    kappa_mean : mean κ over bootstrap samples
    kappa_std  : standard deviation (used as error estimate)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    mask = (t >= t_start) & (t <= t_end)
    t_win = t[mask]
    beta_win = beta[mask]
    n = len(t_win)
    kappas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            k, _ = fit_kappa(t_win[idx], beta_win[idx], t_win[idx].min(), t_win[idx].max())
        except Exception:
            k = float("nan")
        kappas[i] = k
    return float(np.nanmean(kappas)), float(np.nanstd(kappas))


# ---------------------------------------------------------------------------
# Phase-tracking and amplitude modulation analysis
# ---------------------------------------------------------------------------

def phase_tracking_analysis(
    beta: np.ndarray,
    fs_rec: float,
    t_step: float = 40e-9,
    mu_increment: float = 3 * math.pi / 4,
) -> tuple[np.ndarray, float]:
    """
    Compute step-synchronous phase tracking error.

    Downsamples β to one sample per AWG step (by averaging within each step
    window), then computes the phase difference between adjacent steps and
    compares to the expected μ increment.

    Returns
    -------
    phase_error : array of phase errors in radians, shape (n_steps-1,)
    rms_error   : RMS phase tracking error
    """
    samples_per_step = round(fs_rec * t_step)
    if samples_per_step < 1:
        raise ValueError("Recording sample rate too low to resolve AWG steps.")
    n_full_steps = len(beta) // samples_per_step
    beta_rs = beta[: n_full_steps * samples_per_step].reshape(n_full_steps, samples_per_step)
    beta_avg = beta_rs.mean(axis=1)

    phase_diff = np.diff(np.unwrap(np.angle(beta_avg)))
    # Expected increment is mu_increment (mod 2π, nearest)
    error = phase_diff - mu_increment
    # Wrap to [-π, π]
    error = (error + math.pi) % (2 * math.pi) - math.pi
    rms_error = float(np.sqrt(np.mean(error**2)))
    return error, rms_error


def amplitude_modulation_depth(beta: np.ndarray) -> float:
    """
    Compute peak-to-peak amplitude modulation depth as (max-min)/(max+min).
    """
    amp = np.abs(beta)
    amp_max = amp.max()
    amp_min = amp.min()
    if amp_max + amp_min < 1e-30:
        return 0.0
    return float((amp_max - amp_min) / (amp_max + amp_min))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_ringdown_fits(runs: dict, t_start: float, t_end: float) -> None:
    """Plot log|β| vs time with fitted lines for each run."""
    if not HAS_MPL:
        print("[info] matplotlib not available — skipping ringdown plot.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"baseline": "#1f77b4", "pal": "#d62728", "ctrl": "#2ca02c"}
    for label, (t, beta, kappa, kappa_se) in runs.items():
        log_amp = np.log(np.abs(beta) + 1e-30)
        ax.plot(t * 1e6, log_amp, alpha=0.4, color=colors.get(label, "grey"), label=f"{label} data")
        mask = (t >= t_start) & (t <= t_end)
        t_win = t[mask]
        slope = -kappa / 2.0
        intercept = np.mean(log_amp[mask]) - slope * np.mean(t_win)
        ax.plot(
            t_win * 1e6,
            slope * t_win + intercept,
            "--",
            color=colors.get(label, "grey"),
            label=f"{label} fit κ={kappa/1e6:.3f} Mrad/s",
        )
    ax.axvspan(t_start * 1e6, t_end * 1e6, alpha=0.08, color="yellow", label="fit window")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("log|β|")
    ax.set_title("Ringdown fits")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("ringdown_fits.png", dpi=150)
    print("Saved ringdown_fits.png")
    plt.close(fig)


def _plot_kappa_summary(kappa_results: dict) -> None:
    if not HAS_MPL:
        return
    labels = list(kappa_results.keys())
    values = [kappa_results[k][0] / 1e6 for k in labels]
    errors = [kappa_results[k][1] / 1e6 for k in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    ax.bar(labels, values, yerr=errors, capsize=6, color=colors[: len(labels)], alpha=0.85)
    ax.set_ylabel("κ (Mrad/s)")
    ax.set_title("κ summary (post-drive ringdown)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("kappa_summary.png", dpi=150)
    print("Saved kappa_summary.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Analyse ringdown traces and extract κ."
    )
    parser.add_argument("--baseline", required=True, help="NPZ/CSV baseline ringdown trace.")
    parser.add_argument("--pal", required=True, help="NPZ/CSV post-palindromic-drive ringdown.")
    parser.add_argument("--ctrl", required=True, help="NPZ/CSV post-control-drive ringdown.")
    parser.add_argument("--config", default="config_example.yaml", help="YAML config file.")
    parser.add_argument(
        "--ringdown-start", type=float, default=DEFAULT_RINGDOWN_START,
        help="Start of fit window (s, relative to trace start; default 0).",
    )
    parser.add_argument(
        "--ringdown-end", type=float, default=DEFAULT_RINGDOWN_END,
        help="End of fit window (s; default 5e-6).",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP,
        help="Bootstrap samples for κ uncertainty (default 500).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config) if Path(args.config).exists() else {}
    cal = cfg.get("calibration", {})
    i_offset = float(cal.get("i_offset", 0.0))
    q_offset = float(cal.get("q_offset", 0.0))
    phi_imbal = float(cal.get("phi_imbal", 0.0))
    amplitude_imbal = float(cal.get("amplitude_imbal", 1.0))
    fs_rec = float(cfg.get("fs_rec", 1e9))

    run_files = {"baseline": args.baseline, "pal": args.pal, "ctrl": args.ctrl}
    kappa_results: dict[str, tuple[float, float]] = {}
    runs_plot: dict = {}

    for run_name, fpath in run_files.items():
        print(f"\nProcessing {run_name}: {fpath}")
        trace = load_trace(fpath)
        t = trace["t"].astype(np.float64)
        beta = apply_iq_calibration(
            trace["I_rec"].astype(np.float64),
            trace["Q_rec"].astype(np.float64),
            i_offset=i_offset,
            q_offset=q_offset,
            phi_imbal=phi_imbal,
            amplitude_imbal=amplitude_imbal,
        )
        kappa, kappa_se = fit_kappa(t, beta, args.ringdown_start, args.ringdown_end)
        kappa_bs, kappa_bs_std = bootstrap_kappa(
            t, beta, args.ringdown_start, args.ringdown_end,
            n_bootstrap=args.n_bootstrap,
        )
        print(f"  κ (linear fit) = {kappa/1e6:.4f} ± {kappa_se/1e6:.4f} Mrad/s")
        print(f"  κ (bootstrap)  = {kappa_bs/1e6:.4f} ± {kappa_bs_std/1e6:.4f} Mrad/s")
        kappa_results[run_name] = (kappa_bs, kappa_bs_std)
        runs_plot[run_name] = (t, beta, kappa, kappa_se)

    # Δκ: palindromic vs control
    k_pal, se_pal = kappa_results["pal"]
    k_ctrl, se_ctrl = kappa_results["ctrl"]
    delta_kappa = k_pal - k_ctrl
    delta_kappa_se = math.sqrt(se_pal**2 + se_ctrl**2)
    print(f"\nΔκ (pal − ctrl) = {delta_kappa/1e6:.4f} ± {delta_kappa_se/1e6:.4f} Mrad/s")

    _plot_ringdown_fits(runs_plot, args.ringdown_start, args.ringdown_end)
    _plot_kappa_summary(kappa_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
