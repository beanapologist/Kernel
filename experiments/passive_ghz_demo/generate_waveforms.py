#!/usr/bin/env python3
"""
generate_waveforms.py — Passive GHz Demo, Drive-only Variant 1.

Generates I/Q baseband waveform arrays for two variants:
  pal   — palindromic precession + μ 8-cycle phase modulation
  ctrl  — μ 8-cycle phase modulation only (control)

Phase schedule (per step n, 0-indexed):
  φ_μ(n)  = (3π/4) · (n mod 8)
  φ_p(n)  = 2π · n / D          (D = 13,717,421)
  φ_pal(n)  = φ_μ(n) + φ_p(n)
  φ_ctrl(n) = φ_μ(n)

Each step is replicated for `samples_per_step` samples (flat-top envelope).

All phase accumulation uses exact rational arithmetic in "turns" to prevent
drift over many steps.

Outputs an NPZ file with:
  I_pal            — I channel, palindromic variant, shape (N_steps * 96,)
  Q_pal            — Q channel, palindromic variant
  I_ctrl           — I channel, control variant
  Q_ctrl           — Q channel, control variant
  phi_per_step_pal — phase per step (radians), palindromic, shape (N_steps,)
  phi_per_step_ctrl— phase per step (radians), control
  turns_per_step_pal  — phase in turns, palindromic
  turns_per_step_ctrl — phase in turns, control
  metadata         — dict stored as object array (use .item() to retrieve)

Usage:
  python generate_waveforms.py [--n-steps N] [--a-drive A] [--output PATH]
"""

import argparse
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Physical / hardware constants
# ---------------------------------------------------------------------------
FS = 2.4e9          # HDAWG sample rate, Hz
T_STEP = 40e-9      # step duration, seconds
SAMPLES_PER_STEP = 96   # = round(FS * T_STEP) = 96
D = 13_717_421      # slow-precession period, steps
MU_CYCLE = 8        # μ phase cycle length, steps
# μ phase increment per step: 3π/4 radians = 3/8 turn
MU_INCREMENT_TURNS = Fraction(3, 8)


def _assert_samples_per_step() -> None:
    """Verify hardware constants are self-consistent."""
    expected = round(FS * T_STEP)
    assert expected == SAMPLES_PER_STEP, (
        f"SAMPLES_PER_STEP mismatch: expected {expected}, got {SAMPLES_PER_STEP}"
    )


def compute_phase_schedule(n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-step phase arrays for palindromic and control variants.

    Uses exact rational arithmetic in turns to avoid floating-point drift.

    Returns
    -------
    phi_pal : ndarray, shape (n_steps,), radians
    phi_ctrl: ndarray, shape (n_steps,), radians
    """
    # Pre-allocate as float64
    turns_pal = np.empty(n_steps, dtype=np.float64)
    turns_ctrl = np.empty(n_steps, dtype=np.float64)

    # Accumulate using Python Fraction for exact rational arithmetic,
    # then convert to float once per step.
    prec_denom = Fraction(1, D)
    for n in range(n_steps):
        mu_turns = Fraction(3 * (n % MU_CYCLE), 8)
        prec_turns = prec_denom * n          # n/D turns
        turns_pal[n] = float(mu_turns + prec_turns)
        turns_ctrl[n] = float(mu_turns)

    phi_pal = turns_pal * (2.0 * math.pi)
    phi_ctrl = turns_ctrl * (2.0 * math.pi)
    return phi_pal, phi_ctrl


def build_waveforms(
    n_steps: int,
    a_drive: float,
) -> dict:
    """
    Build I/Q waveform arrays for both variants.

    Parameters
    ----------
    n_steps  : number of steps
    a_drive  : drive amplitude (0 < a_drive ≤ 1.0, full-scale = 1.0)

    Returns
    -------
    dict with keys: I_pal, Q_pal, I_ctrl, Q_ctrl,
                    phi_per_step_pal, phi_per_step_ctrl,
                    turns_per_step_pal, turns_per_step_ctrl, metadata
    """
    _assert_samples_per_step()

    if not (0.0 < a_drive <= 1.0):
        raise ValueError(f"a_drive must be in (0, 1]; got {a_drive}")

    phi_pal, phi_ctrl = compute_phase_schedule(n_steps)

    # Replicate each step value for SAMPLES_PER_STEP samples (flat-top)
    phi_pal_full = np.repeat(phi_pal, SAMPLES_PER_STEP)
    phi_ctrl_full = np.repeat(phi_ctrl, SAMPLES_PER_STEP)

    I_pal = a_drive * np.cos(phi_pal_full)
    Q_pal = a_drive * np.sin(phi_pal_full)
    I_ctrl = a_drive * np.cos(phi_ctrl_full)
    Q_ctrl = a_drive * np.sin(phi_ctrl_full)

    # Turns (for inspection / unit tests)
    turns_pal = phi_pal / (2.0 * math.pi)
    turns_ctrl = phi_ctrl / (2.0 * math.pi)

    metadata = {
        "fs": FS,
        "t_step": T_STEP,
        "samples_per_step": SAMPLES_PER_STEP,
        "n_steps": n_steps,
        "D": D,
        "mu_cycle": MU_CYCLE,
        "a_drive": a_drive,
        "total_samples": n_steps * SAMPLES_PER_STEP,
        "description": "Passive GHz Demo — Drive-only Variant 1",
    }

    return dict(
        I_pal=I_pal,
        Q_pal=Q_pal,
        I_ctrl=I_ctrl,
        Q_ctrl=Q_ctrl,
        phi_per_step_pal=phi_pal,
        phi_per_step_ctrl=phi_ctrl,
        turns_per_step_pal=turns_pal,
        turns_per_step_ctrl=turns_ctrl,
        metadata=np.array(metadata, dtype=object),
    )


def save_waveforms(data: dict, output_path: str) -> None:
    """Save waveform data to an NPZ file."""
    np.savez(output_path, **data)
    path = Path(output_path)
    if not path.suffix:
        path = path.with_suffix(".npz")
    print(f"Saved waveforms to {path.with_suffix('.npz')}")
    meta = data["metadata"].item()
    print(f"  N_steps          : {meta['n_steps']}")
    print(f"  Samples per step : {meta['samples_per_step']}")
    print(f"  Total samples    : {meta['total_samples']}")
    print(f"  A_drive          : {meta['a_drive']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate I/Q waveforms for Passive GHz Demo Variant 1."
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=200,
        help="Number of 40-ns steps (default: 200).",
    )
    parser.add_argument(
        "--a-drive",
        type=float,
        default=0.5,
        help="Drive amplitude, 0 < A ≤ 1.0 (default: 0.5).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="waveforms",
        help="Output NPZ path (extension added automatically; default: waveforms).",
    )
    args = parser.parse_args(argv)

    data = build_waveforms(n_steps=args.n_steps, a_drive=args.a_drive)
    save_waveforms(data, args.output)


if __name__ == "__main__":
    main()
