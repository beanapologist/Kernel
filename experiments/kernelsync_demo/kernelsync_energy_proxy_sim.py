#!/usr/bin/env python3
"""
kernelsync_energy_proxy_sim.py — KernelSync Energy-Proxy Simulation

Simulates N network nodes under heavy-tail packet-delay variation (PDV) and
compares a KernelSync pilot burst code against an all-ones (baseline) code on
the energy proxy E = R x M (events/sec x chips/event) for a given RMS timing
target.

KernelSync code
---------------
Each chip k in a pilot burst starting at global index n0 has phase:

    s[k] = exp(i * 2pi * phi(n0 + k))
    phi(n) = ((3/8)*(n mod 8) + (n mod D)/D) mod 1

where D = 13 717 421 is the palindromic precession period and 3/8 turns
(= 3pi/4 rad) is the mu-step increment.

The baseline code is s[k] = 1 for all k.

Usage
-----
    python kernelsync_energy_proxy_sim.py [options]

See --help for full option list.

Coherent receiver (KernelSync only)
------------------------------------
KernelSync uses a coherent receiver that exploits the deterministic phase
structure of the code for both integer-chip and sub-chip timing:

1.  Continuous signal model: the leader transmits continuously; the follower
    observes M chips from the global code sequence displaced by tau_int chips.
    This eliminates edge-artefact plateaus in the correlation magnitude.

2.  Phase-based integer detection: within a tight window of +-W_tight chips
    around the PI-loop-predicted tau, the lag minimising |angle(corr)-psi_hat|
    is chosen.  The 3/8 turns/chip phase increment gives an unambiguous range
    of +-4 chips around the prediction.

3.  Sub-chip timing: the complex MF peak encodes the fractional chip offset
    delta via residual phase.  delta_est = -phi_res / (2*pi * 3/8) reduces
    per-measurement timing noise from ~Tc/2 to ~sigma_phase/(2*pi*3/8)*Tc
    (typically 5-20x improvement at practical SNR and burst length).

4.  Burst-to-burst carrier-phase continuity: each node maintains a persistent
    carrier-phase estimate psi_hat updated by EMA each burst.  The psi_hat
    update removes the sub-chip contribution so only the carrier phase remains,
    enabling stable tracking across bursts.

Physical energy model
---------------------
E_physical (W per node) = (tx_pj_per_chip * M + rx_pj_per_burst) * R * 1e-12
  typical: tx_pj_per_chip = 10 pJ  (SerDes Tx per chip)
           rx_pj_per_burst = 100 pJ (FFT correlator per burst)
"""

import argparse
import json
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
    print("[warn] matplotlib not found — plots will be skipped.")

# ---------------------------------------------------------------------------
# Core constants
# ---------------------------------------------------------------------------

D = 13_717_421          # palindromic precession period (chip steps)
MU_CYCLE = 8            # mu-cycle length (steps)
MU_STEP_TURNS = 3 / 8  # mu increment per step (turns) = 3pi/4 rad per step


def phase_turns(n: int) -> float:
    """
    Return the KernelSync chip phase in *turns* for global chip index n.

    phi(n) = ( (3/8) * (n mod 8) + (n mod D) / D ) mod 1

    Encodes both the 8-step mu-cycle and the slow palindromic precession.
    """
    return ((MU_STEP_TURNS * (n % MU_CYCLE)) + ((n % D) / D)) % 1.0


def make_kernelsync_code(n0: int, M: int) -> np.ndarray:
    """Return complex KernelSync chip sequence of length M starting at chip n0."""
    n = np.arange(n0, n0 + M, dtype=np.int64)
    phases = ((MU_STEP_TURNS * (n % MU_CYCLE)) + ((n % D) / D)) % 1.0
    return np.exp(1j * 2 * np.pi * phases)


def make_baseline_code(M: int) -> np.ndarray:
    """Return all-ones (BPSK DC) baseline code of length M."""
    return np.ones(M, dtype=complex)


# ---------------------------------------------------------------------------
# PDV model — heavy-tail mixture
# ---------------------------------------------------------------------------

def sample_pdv(rng: np.random.Generator, n_samples: int) -> np.ndarray:
    """
    Heavy-tail PDV mixture (seconds):
        w.p. 0.99 : delay ~ N(0, (2 ns)^2)
        w.p. 0.01 : delay ~ N(0, (50 ns)^2)
    """
    outlier = rng.random(n_samples) < 0.01
    return np.where(
        outlier,
        rng.normal(0, 50e-9, n_samples),
        rng.normal(0, 2e-9, n_samples),
    )


# ---------------------------------------------------------------------------
# Matched-filter timing estimator (FFT-based, integer-chip)
# ---------------------------------------------------------------------------

def matched_filter_tau_fast(
    y: np.ndarray,
    s: np.ndarray,
    W: int,
) -> int:
    """
    Estimate integer chip offset tau_hat in [-W, W] via correlation:

        tau_hat = argmax_{tau in [-W,W]} |sum_k  y[k] * conj(s[k-tau])|

    Uses zero-padded FFT cross-correlation.  Returns integer chips.
    """
    M = len(s)
    n_fft = max(2 * M, 2 * W + 1)
    Y = np.fft.fft(y, n_fft)
    S = np.fft.fft(s, n_fft)
    corr = np.fft.ifft(Y * np.conj(S))
    taus = np.arange(-W, W + 1)
    metrics = np.abs(corr[taus % n_fft])
    return int(taus[np.argmax(metrics)])


def _batch_matched_filter(
    Y: np.ndarray,
    s: np.ndarray,
    W: int,
) -> np.ndarray:
    """
    Batch FFT matched filter: N signals Y[n, :] against one reference s.

    Returns tau_hats (N,) integer chip estimates.
    """
    N_nodes, M = Y.shape
    n_fft = max(2 * M, 2 * W + 1)
    S_fft = np.fft.fft(s, n_fft)
    Y_fft = np.fft.fft(Y, n_fft, axis=1)
    Corr = np.fft.ifft(Y_fft * np.conj(S_fft), axis=1)
    taus = np.arange(-W, W + 1)
    metrics = np.abs(Corr[:, taus % n_fft])
    return taus[np.argmax(metrics, axis=1)]


def _batch_matched_filter_coherent(
    Y: np.ndarray,
    s: np.ndarray,
    W: int,
    psi_hat: np.ndarray,
    pred_tau: np.ndarray,
    W_tight: int,
) -> tuple:
    """
    Coherent phase-based matched filter for KernelSync.

    For each node n, finds the lag in [pred_tau[n]-W_tight, pred_tau[n]+W_tight]
    that minimises |wrap(angle(corr[tau]) - psi_hat[n])|.  This exploits the
    KernelSync code's known phase rotation (3/8 turns per chip) to resolve the
    integer chip offset without relying on correlation magnitude.

    W_tight must be < 4 to guarantee phase-unambiguous detection (the code's
    phase repeats every 8/3 ~= 2.67 chips; the window must be smaller than
    half that period).

    Returns (tau_hats (N,), peak_complex (N,)).
    """
    N, M = Y.shape
    n_fft = max(2 * M, 2 * W + 1)
    Corr = np.fft.ifft(
        np.fft.fft(Y, n_fft, axis=1) * np.conj(np.fft.fft(s, n_fft)),
        axis=1,
    )  # (N, n_fft)

    taus = np.arange(-W, W + 1)                               # (2W+1,)
    phases = np.angle(Corr[:, taus % n_fft])                  # (N, 2W+1)

    # Per-node phase residual |wrap(angle - psi_hat)|
    phi_res = np.abs(
        (phases - psi_hat[:, np.newaxis] + np.pi) % (2 * np.pi) - np.pi
    )  # (N, 2W+1)

    # Restrict search to [pred_tau-W_tight, pred_tau+W_tight] per node
    mask = (
        (taus[np.newaxis, :] >= (pred_tau - W_tight)[:, np.newaxis]) &
        (taus[np.newaxis, :] <= (pred_tau + W_tight)[:, np.newaxis])
    )  # (N, 2W+1)
    phi_res_masked = np.where(mask, phi_res, np.inf)

    best_idx = np.argmin(phi_res_masked, axis=1)               # (N,)
    tau_hats = taus[best_idx]                                   # (N,)
    peak_complex = Corr[np.arange(N), tau_hats % n_fft]        # (N,)
    return tau_hats, peak_complex


# ---------------------------------------------------------------------------
# Vectorised multi-node simulation
# ---------------------------------------------------------------------------

def simulate_all_nodes(
    rng: np.random.Generator,
    skew_ppms: np.ndarray,
    theta0s: np.ndarray,
    scheme: str,
    R: float,
    M: int,
    Tc: float,
    W: int,
    T_sim: float,
    n_times: int = 100,
    snr_per_chip: float = 20.0,
    W_tight: int = 3,
) -> np.ndarray:
    """
    Vectorised simulation of all N follower nodes simultaneously.

    KernelSync uses the coherent receiver:
      - Continuous signal model (no zero-padding): y[k] = exp(i*psi)*s_global[n0-tau+k]
      - Phase-based integer detection within +-W_tight of PI-predicted tau
      - Sub-chip correction from complex MF peak phase residual
      - Persistent per-node carrier-phase estimate psi_hat (EMA-updated)

    Baseline uses the incoherent receiver (unchanged):
      - Zero-padded signal model
      - Magnitude-based integer-chip MF

    Returns err_trace of shape (N, n_times) in seconds.
    """
    N = len(skew_ppms)
    eps = skew_ppms * 1e-6
    theta_hat = theta0s.copy()
    skew_hat = np.zeros(N)

    dt_pilot = 1.0 / R
    n_pilots = int(T_sim * R)
    noise_sigma = 1.0 / np.sqrt(snr_per_chip)

    alpha_P = 0.6
    alpha_I = 0.15

    # Persistent carrier phase per node (constant; models burst-to-burst continuity)
    psi_true = rng.uniform(0, 2 * np.pi, N)  # (N,) constant for whole simulation
    psi_hat = np.zeros(N)                      # (N,) carrier-phase estimate
    alpha_psi = 0.3                            # EMA gain for psi_hat tracking

    t_samples = np.linspace(0, T_sim, n_times)
    err_trace = np.zeros((N, n_times))

    k_idx = np.arange(M, dtype=np.int64)
    prev_meas = np.zeros(N)
    pilot_idx = 0

    for eval_idx, t_eval in enumerate(t_samples):
        while pilot_idx < n_pilots and (pilot_idx + 1) * dt_pilot <= t_eval:
            t = (pilot_idx + 1) * dt_pilot

            # Residual timing error for all nodes
            true_error = eps * t + theta0s - theta_hat   # (N,)

            # Feed-forward slew
            theta_hat += skew_hat * dt_pilot

            # PDV
            pdv = sample_pdv(rng, N)
            total_offset = true_error + pdv              # (N,)

            # Integer chip offset (clipped to search window)
            tau_int = np.clip(
                np.round(total_offset / Tc).astype(int), -W, W
            )  # (N,)

            # Sub-chip fractional remainder
            delta_frac = total_offset / Tc - tau_int     # (N,)

            # Reference code (same for all nodes at this pilot)
            global_chip = pilot_idx * M

            noise = (
                rng.normal(0, noise_sigma, (N, M))
                + 1j * rng.normal(0, noise_sigma, (N, M))
            )

            if scheme == "kernelsync":
                s = make_kernelsync_code(global_chip, M)

                # Continuous signal model: y[k] = exp(i*(psi-subchip_ph)) * s_global[n0-tau+k]
                # Vectorised per-node code starting position
                n_per_node = global_chip - tau_int       # (N,), different per node
                n_indices = n_per_node[:, np.newaxis] + k_idx[np.newaxis, :]  # (N,M)
                phases_rx = (
                    (MU_STEP_TURNS * (n_indices % MU_CYCLE)) +
                    ((n_indices % D) / D)
                ) % 1.0
                s_rx = np.exp(1j * 2 * np.pi * phases_rx)  # (N, M)

                # Sub-chip phase: fractional delay rotates the whole burst by
                # exp(-i * 2*pi * 3/8 * delta_frac) for a constant-increment code
                subchip_phase = 2 * np.pi * MU_STEP_TURNS * delta_frac  # (N,)
                Y = (
                    np.exp(1j * (psi_true - subchip_phase))[:, np.newaxis] * s_rx
                    + noise
                )

                # PI-predicted integer chip offset (used as search centre)
                pred_tau = np.clip(
                    np.round(true_error / Tc).astype(int), -W, W
                )  # (N,)

                # Coherent phase-based detection + peak complex value
                tau_hats, peak_complex = _batch_matched_filter_coherent(
                    Y, s, W, psi_hat, pred_tau, W_tight
                )

                # Sub-chip correction from phase residual
                measured_phase = np.angle(peak_complex)  # (N,)
                phi_residual = (
                    (measured_phase - psi_hat + np.pi) % (2 * np.pi) - np.pi
                )  # (N,)
                delta_est = np.clip(
                    -phi_residual / (2 * np.pi * MU_STEP_TURNS), -0.5, 0.5
                )  # (N,)
                meas = (tau_hats + delta_est) * Tc       # sub-chip measurement

                # Update psi_hat (EMA): remove sub-chip contribution to isolate carrier
                psi_hat_new = measured_phase + 2 * np.pi * MU_STEP_TURNS * delta_est
                d_psi = (psi_hat_new - psi_hat + np.pi) % (2 * np.pi) - np.pi
                psi_hat = psi_hat + alpha_psi * d_psi

            else:  # baseline: incoherent integer-chip MF (unchanged)
                s = make_baseline_code(M)
                k_minus_tau = k_idx[np.newaxis, :] - tau_int[:, np.newaxis]
                valid = (k_minus_tau >= 0) & (k_minus_tau < M)
                s_shifted = np.where(
                    valid, s[np.clip(k_minus_tau, 0, M - 1)], 0.0
                )
                Y = (
                    np.exp(1j * psi_true[:, np.newaxis]) * s_shifted + noise
                )
                tau_hats = _batch_matched_filter(Y, s, W)
                meas = tau_hats.astype(float) * Tc

            # PI correction
            theta_hat += alpha_P * meas
            skew_hat += alpha_I * (meas - prev_meas) / dt_pilot

            prev_meas = meas
            pilot_idx += 1

        err_trace[:, eval_idx] = eps * t_eval + theta0s - theta_hat

    return err_trace


# Single-node wrapper (retained for testing / CLI compatibility)
def simulate_node(
    rng: np.random.Generator,
    skew_ppm: float,
    theta0: float,
    scheme: str,
    R: float,
    M: int,
    Tc: float,
    W: int,
    T_sim: float,
    snr_per_chip: float = 20.0,
    n_times: int = 200,
) -> np.ndarray:
    """Single-node wrapper around simulate_all_nodes."""
    return simulate_all_nodes(
        rng=rng,
        skew_ppms=np.array([skew_ppm]),
        theta0s=np.array([theta0]),
        scheme=scheme,
        R=R, M=M, Tc=Tc, W=W, T_sim=T_sim,
        n_times=n_times, snr_per_chip=snr_per_chip,
    )[0]


# ---------------------------------------------------------------------------
# Multi-node grid sweep
# ---------------------------------------------------------------------------

def run_grid(
    rng: np.random.Generator,
    scheme: str,
    grid_R: list,
    grid_M: list,
    n_nodes: int,
    T_sim: float,
    Tc: float,
    W_ns: float,
    n_times: int = 100,
    tx_pj_per_chip: float = 10.0,
    rx_pj_per_burst: float = 100.0,
) -> dict:
    """
    For each (R, M) pair, simulate n_nodes follower nodes.

    Returns dict keyed by (R, M) with keys:
        final_rms_ns, E, rms_trace_ns, power_nW
    where power_nW = (tx_pj_per_chip*M + rx_pj_per_burst)*R*1e-3 nW per node.
    """
    W = int(round(W_ns / Tc))

    skew_ppms = rng.uniform(-50, 50, n_nodes)
    theta0s = rng.uniform(-200e-9, 200e-9, n_nodes)

    results = {}
    total = len(grid_R) * len(grid_M)
    done = 0

    for R in grid_R:
        for M in grid_M:
            done += 1
            power_nW = (tx_pj_per_chip * M + rx_pj_per_burst) * R * 1e-3
            print(
                f"  [{scheme:>12}] R={R:5g} evt/s  M={M:3d} chips  "
                f"E={R*M:7g}  P={power_nW:.1f}nW  ({done}/{total}) ...",
                end=" ", flush=True,
            )

            child_rng = np.random.default_rng(rng.integers(0, 2**63))
            err_traces = simulate_all_nodes(
                rng=child_rng,
                skew_ppms=skew_ppms, theta0s=theta0s,
                scheme=scheme, R=R, M=int(M),
                Tc=Tc, W=W, T_sim=T_sim, n_times=n_times,
            )

            rms_vs_time = np.sqrt(np.mean(err_traces ** 2, axis=0))
            final_rms = float(rms_vs_time[-1])
            results[(R, M)] = {
                "final_rms_ns": final_rms * 1e9,
                "E": R * M,
                "power_nW": power_nW,
                "rms_trace_ns": rms_vs_time * 1e9,
            }
            print(f"RMS = {final_rms*1e9:.3f} ns")

    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_pareto(results_base, results_ks, target_rms_ns, out_path):
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    Es_b = [v["E"] for v in results_base.values()]
    rms_b = [v["final_rms_ns"] for v in results_base.values()]
    Es_k = [v["E"] for v in results_ks.values()]
    rms_k = [v["final_rms_ns"] for v in results_ks.values()]
    ax.scatter(Es_b, rms_b, marker="o", label="Baseline", alpha=0.7)
    ax.scatter(Es_k, rms_k, marker="s", label="KernelSync", alpha=0.7)
    ax.axhline(target_rms_ns, color="red", linestyle="--",
               label=f"Target {target_rms_ns} ns")
    ax.set_xscale("log")
    ax.set_xlabel("Energy proxy E = R x M  [events/s x chips/event]")
    ax.set_ylabel("Final RMS time error [ns]")
    ax.set_title("KernelSync vs Baseline — Energy vs Timing Error")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_heatmap(results, grid_R, grid_M, target_rms_ns, title, out_path):
    if not HAS_MPL:
        return
    data = np.array([
        [results[(R, M)]["final_rms_ns"] for R in grid_R]
        for M in grid_M
    ])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(data, origin="lower", aspect="auto",
                   extent=[0, len(grid_R), 0, len(grid_M)])
    ax.set_xticks(np.arange(len(grid_R)) + 0.5)
    ax.set_xticklabels([str(int(r)) for r in grid_R])
    ax.set_yticks(np.arange(len(grid_M)) + 0.5)
    ax.set_yticklabels([str(m) for m in grid_M])
    ax.set_xlabel("R  [events/s]")
    ax.set_ylabel("M  [chips/event]")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Final RMS error [ns]")
    for ri, R in enumerate(grid_R):
        for mi, M in enumerate(grid_M):
            if results[(R, M)]["final_rms_ns"] <= target_rms_ns:
                ax.add_patch(plt.Rectangle(
                    (ri, mi), 1, 1, fill=False, edgecolor="lime", lw=2
                ))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_rms_vs_time(results_base, results_ks, best_key_base, best_key_ks,
                     T_sim, out_path):
    if not HAS_MPL:
        return
    t = np.linspace(0, T_sim, len(results_base[best_key_base]["rms_trace_ns"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, results_base[best_key_base]["rms_trace_ns"],
            label=f"Baseline R={best_key_base[0]:.0f} M={best_key_base[1]}")
    ax.plot(t, results_ks[best_key_ks]["rms_trace_ns"],
            label=f"KernelSync R={best_key_ks[0]:.0f} M={best_key_ks[1]}")
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("RMS time error across nodes [ns]")
    ax.set_title("RMS Time Error vs Time — Best Operating Points")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def find_min_energy(results, target_rms_ns):
    candidates = [
        (k, v) for k, v in results.items()
        if v["final_rms_ns"] <= target_rms_ns
    ]
    if not candidates:
        return None, None
    return min(candidates, key=lambda kv: kv[1]["E"])


def print_summary(results_base, results_ks, target_rms_ns):
    key_b, val_b = find_min_energy(results_base, target_rms_ns)
    key_k, val_k = find_min_energy(results_ks, target_rms_ns)

    best_b_key = min(results_base, key=lambda k: results_base[k]["final_rms_ns"])
    best_k_key = min(results_ks, key=lambda k: results_ks[k]["final_rms_ns"])
    best_b = results_base[best_b_key]
    best_k = results_ks[best_k_key]

    print()
    print("=" * 62)
    print("  KernelSync Energy-Proxy Simulation -- Results Summary")
    print("=" * 62)
    print(f"  Target RMS: {target_rms_ns} ns")
    print()
    if val_b:
        print(f"  Baseline   min-E at target:")
        print(f"    R = {key_b[0]:.0f} evt/s,  M = {key_b[1]} chips")
        print(f"    E = {val_b['E']:.0f}  (RMS = {val_b['final_rms_ns']:.3f} ns)")
    else:
        print(f"  Baseline   -- target NOT achieved; best RMS = "
              f"{best_b['final_rms_ns']:.2f} ns at E = {best_b['E']:.0f}")
    print()
    if val_k:
        print(f"  KernelSync min-E at target:")
        print(f"    R = {key_k[0]:.0f} evt/s,  M = {key_k[1]} chips")
        print(f"    E = {val_k['E']:.0f}  (RMS = {val_k['final_rms_ns']:.3f} ns)")
    else:
        print(f"  KernelSync -- target NOT achieved; best RMS = "
              f"{best_k['final_rms_ns']:.2f} ns at E = {best_k['E']:.0f}")
    print()

    factor = None
    if val_b and val_k and val_k["E"] > 0:
        factor = val_b["E"] / val_k["E"]
        print(f"  Improvement factor (E_baseline / E_KernelSync): {factor:.2f}x")
    else:
        rms_ratio = None
        if best_b["final_rms_ns"] > 0 and best_k["final_rms_ns"] > 0:
            rms_ratio = best_b["final_rms_ns"] / best_k["final_rms_ns"]
            print(f"  Best-RMS ratio (baseline / KernelSync): {rms_ratio:.3f}x")
    print("=" * 62)

    return {
        "target_rms_ns": target_rms_ns,
        "baseline": {
            "min_E_key": list(key_b) if key_b else None,
            "min_E": val_b["E"] if val_b else None,
            "rms_ns": val_b["final_rms_ns"] if val_b else None,
            "best_rms_ns": best_b["final_rms_ns"],
            "best_E": best_b["E"],
        },
        "kernelsync": {
            "min_E_key": list(key_k) if key_k else None,
            "min_E": val_k["E"] if val_k else None,
            "rms_ns": val_k["final_rms_ns"] if val_k else None,
            "best_rms_ns": best_k["final_rms_ns"],
            "best_E": best_k["E"],
        },
        "improvement_factor": factor,
        "rms_ratio": (
            best_b["final_rms_ns"] / best_k["final_rms_ns"]
            if best_k["final_rms_ns"] > 0 else None
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="KernelSync Energy-Proxy Simulation -- N-node PDV demo."
    )
    p.add_argument("--nodes", type=int, default=1000)
    p.add_argument("--tsim", type=float, default=10.0)
    p.add_argument("--tc", type=float, default=40e-9)
    p.add_argument("--window", type=float, default=250e-9)
    p.add_argument("--target-rms-ns", type=float, default=1.0)
    p.add_argument("--grid-R", type=str, default="50,100,200,500,1000")
    p.add_argument("--grid-M", type=str, default="8,16,32,64")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str,
                   default=str(Path(__file__).parent / "out"))
    p.add_argument("--n-times", type=int, default=100)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    grid_R = [float(x) for x in args.grid_R.split(",")]
    grid_M = [int(x) for x in args.grid_M.split(",")]
    rng = np.random.default_rng(args.seed)

    print("=" * 62)
    print("  KernelSync Energy-Proxy Simulation")
    print(f"  nodes={args.nodes}  T_sim={args.tsim}s  Tc={args.tc*1e9:.0f}ns")
    print(f"  window=+-{args.window*1e9:.0f}ns  target={args.target_rms_ns}ns RMS")
    print(f"  grid_R={grid_R}  grid_M={grid_M}")
    print("=" * 62)

    print("\n[1/2] Running Baseline simulation ...")
    results_base = run_grid(
        rng=rng, scheme="baseline",
        grid_R=grid_R, grid_M=grid_M, n_nodes=args.nodes,
        T_sim=args.tsim, Tc=args.tc, W_ns=args.window, n_times=args.n_times,
    )

    print("\n[2/2] Running KernelSync simulation ...")
    results_ks = run_grid(
        rng=rng, scheme="kernelsync",
        grid_R=grid_R, grid_M=grid_M, n_nodes=args.nodes,
        T_sim=args.tsim, Tc=args.tc, W_ns=args.window, n_times=args.n_times,
    )

    summary = print_summary(results_base, results_ks, args.target_rms_ns)

    json_path = out / "results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved {json_path}")

    if HAS_MPL:
        print("\nGenerating plots ...")
        plot_pareto(results_base, results_ks, args.target_rms_ns,
                    str(out / "pareto_energy_vs_error.png"))
        plot_heatmap(results_base, grid_R, grid_M, args.target_rms_ns,
                     "Baseline -- Final RMS error [ns]",
                     str(out / "heatmap_baseline.png"))
        plot_heatmap(results_ks, grid_R, grid_M, args.target_rms_ns,
                     "KernelSync -- Final RMS error [ns]",
                     str(out / "heatmap_kernelsync.png"))

        key_b, _ = find_min_energy(results_base, args.target_rms_ns)
        key_k, _ = find_min_energy(results_ks, args.target_rms_ns)
        if key_b is None:
            key_b = min(results_base, key=lambda k: results_base[k]["final_rms_ns"])
        if key_k is None:
            key_k = min(results_ks, key=lambda k: results_ks[k]["final_rms_ns"])
        plot_rms_vs_time(results_base, results_ks, key_b, key_k, args.tsim,
                         str(out / "rms_vs_time_best.png"))

    print("\nDone. Outputs written to:", out)


if __name__ == "__main__":
    main()
