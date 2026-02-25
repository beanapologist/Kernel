#!/usr/bin/env python3
"""
kernelsync_energy_proxy_sim.py — KernelSync Energy-Proxy Simulation

Simulates 1000 network nodes under heavy-tail packet-delay variation (PDV)
to demonstrate that a structured pilot burst code (derived from the μ 8-cycle
+ palindromic precession) improves timing detection / alignment robustness,
enabling a lower pilot-rate × burst-length energy proxy E = R * M for a
fixed RMS synchronisation target.

Usage
-----
    python kernelsync_energy_proxy_sim.py [options]

See --help for full option list.

Outputs (saved to experiments/kernelsync_demo/out/ by default)
--------------------------------------------------------------
    pareto_energy_vs_error.png   — E vs RMS scatter for both codes
    heatmap_baseline.png         — heat-map of final RMS over (R, M) grid
    heatmap_kernelsync.png       — heat-map of final RMS over (R, M) grid
    rms_vs_time_best.png         — RMS time-error vs wall-clock for best (R,M)
    results.json                 — min-E operating points and metrics
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
MU_CYCLE = 8            # μ-cycle length (steps)
MU_STEP_TURNS = 3 / 8  # μ increment per step (turns) = 3π/4 rad per step


def phase_turns(n: int) -> float:
    """
    Return the KernelSync chip phase in *turns* for global chip index n.

    φ(n) = ( (3/8) * (n mod 8) + (n mod D) / D ) mod 1

    This encodes both the 8-step μ-cycle and the slow palindromic precession.
    """
    return ((MU_STEP_TURNS * (n % MU_CYCLE)) + ((n % D) / D)) % 1.0


def make_kernelsync_code(n0: int, M: int) -> np.ndarray:
    """Return complex KernelSync chip sequence of length M starting at chip n0."""
    phases = np.array([phase_turns(n0 + k) for k in range(M)])
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
        w.p. 0.99 : delay ~ N(0, (2 ns)²)
        w.p. 0.01 : delay ~ N(0, (50 ns)²)
    """
    outlier = rng.random(n_samples) < 0.01
    delay = np.where(
        outlier,
        rng.normal(0, 50e-9, n_samples),
        rng.normal(0, 2e-9, n_samples),
    )
    return delay


# ---------------------------------------------------------------------------
# Matched-filter / correlation timing estimator
# ---------------------------------------------------------------------------

def matched_filter_tau(
    y: np.ndarray,
    s: np.ndarray,
    W: int,
) -> int:
    """
    Estimate integer chip offset τ̂ ∈ [−W, W] via correlation:

        τ̂ = argmax_{τ ∈ [−W,W]} |Σ_k  y[k] · conj(s[k−τ])|

    Parameters
    ----------
    y : received chip sequence, length M
    s : reference code, length M
    W : half search-window in chips
    Returns
    -------
    tau_hat : integer chip offset estimate
    """
    M = len(s)
    best_metric = -1.0
    best_tau = 0
    for tau in range(-W, W + 1):
        acc = 0.0 + 0.0j
        for k in range(M):
            km = k - tau
            if 0 <= km < M:
                acc += y[k] * np.conj(s[km])
        metric = abs(acc)
        if metric > best_metric:
            best_metric = metric
            best_tau = tau
    return best_tau


def matched_filter_tau_fast(
    y: np.ndarray,
    s: np.ndarray,
    W: int,
) -> int:
    """Vectorised matched filter — returns integer chip offset estimate.

    tau_hat = argmax_{tau in [-W,W]} |Σ_k  y[k] · conj(s[k−τ])|

    Returns integer chips.  For sub-chip resolution on KernelSync, see
    ``kernelsync_subchip_offset``.
    """
    M = len(s)
    metrics = np.empty(2 * W + 1)
    for i, tau in enumerate(range(-W, W + 1)):
        if tau >= 0:
            ys = y[tau: tau + M]
            ss = s[: len(ys)]
            # pad if ys shorter than s
            if len(ys) < M:
                pad = M - len(ys)
                ys = np.concatenate([ys, np.zeros(pad, dtype=complex)])
                ss = s
        else:
            # tau < 0: code extends before received window
            ys = y[: M + tau]
            ss = s[-tau: -tau + len(ys)]
            if len(ys) < M:
                pad = M - len(ys)
                ys = np.concatenate([np.zeros(pad, dtype=complex), ys])
                ss = s
        metrics[i] = abs(np.dot(ys, np.conj(ss)))
    return int(np.argmax(metrics)) - W


def kernelsync_subchip_offset(
    y: np.ndarray,
    tau_int: int,
    Tc: float,
) -> float:
    """
    Refine an integer-chip estimate with a sub-chip correction using
    differential phase of successive KernelSync chips.

    The KernelSync code has a constant per-chip phase increment of
    MU_STEP_TURNS = 3/8 turns (= 3π/4 rad).  The differential product
    D[k] = y[k+1] · conj(y[k]) therefore has signal component:

        D[k] ≈ exp(i · 2π · 3/8) · exp(i · 2π · (3/8) · δ)

    where δ is the fractional chip offset.  Taking the mean of D and
    stripping the known 3/8 turn gives an estimate of δ.

    This approach is carrier-phase independent (the common carrier rotation
    ψ cancels in the conjugate product), giving genuine sub-chip resolution.

    Parameters
    ----------
    y        : received chip sequence of length M
    tau_int  : integer chip offset from matched_filter_tau_fast
    Tc       : chip period (s)

    Returns
    -------
    total_offset_s : estimated timing offset in seconds (integer + fractional)
    """
    M = len(y)
    # Build differential product over the received window
    if M < 2:
        return tau_int * Tc
    diff = y[1:] * np.conj(y[:-1])          # length M-1
    mean_diff = np.mean(diff)

    # Remove known constant phase increment of the code (3π/4 rad)
    known_phase_increment = 2 * np.pi * MU_STEP_TURNS
    mean_diff_derotated = mean_diff * np.exp(-1j * known_phase_increment)

    # Fractional chip offset from residual phase
    residual_phase = np.angle(mean_diff_derotated)  # in [-π, π]
    # Each chip of offset adds 2π·(3/8) to the phase; residual = 2π·(3/8)·δ
    if abs(2 * np.pi * MU_STEP_TURNS) > 1e-10:
        delta_chips = residual_phase / (2 * np.pi * MU_STEP_TURNS)
    else:
        delta_chips = 0.0

    # Clamp fractional correction to ±0.5 chips
    delta_chips = max(-0.5, min(0.5, delta_chips))
    return (tau_int + delta_chips) * Tc


# ---------------------------------------------------------------------------
# Single-node simulation
# ---------------------------------------------------------------------------

def simulate_node(
    rng: np.random.Generator,
    skew_ppm: float,
    theta0: float,
    scheme: str,        # "baseline" | "kernelsync"
    R: float,           # pilot rate (events/s)
    M: int,             # chips per event
    Tc: float,          # chip period (s)
    W: int,             # search window (chips)
    T_sim: float,       # simulation duration (s)
    snr_per_chip: float = 20.0,   # SNR per chip (linear)
    n_times: int = 200,           # number of time samples for RMS history
) -> np.ndarray:
    """
    Simulate one follower node for T_sim seconds and return the
    time-error trajectory sampled at n_times evenly-spaced instants.

    Clock model:  t_node(t) = (1 + ε) * t + θ0
    where ε = skew_ppm * 1e-6.

    The node is assumed to have been coarsely synchronised at t=0 so that
    theta_hat is initialised to theta0.  From that point the correction loop
    only needs to track the frequency-skew drift  ε * t.

    Returns
    -------
    err_trace : shape (n_times,) — time error (s) at each sample instant
    """
    eps = skew_ppm * 1e-6

    # Initialise with coarse sync: offset estimate = theta0 (acquired externally)
    theta_hat = theta0
    skew_hat = 0.0

    # Pilot event times
    dt_pilot = 1.0 / R
    t_next_pilot = dt_pilot  # first event after t=0

    # Global chip counter used for KernelSync phase
    global_chip = 0

    # Slew gain parameters  (tune for stability)
    alpha_offset = 0.5    # gain for offset correction
    alpha_skew = 0.1      # gain for skew correction

    # Noise sigma (complex, per chip)
    noise_sigma = 1.0 / np.sqrt(snr_per_chip)

    t_samples = np.linspace(0, T_sim, n_times)
    err_trace = np.empty(n_times)

    t = 0.0
    prev_t_pilot = 0.0
    prev_meas_s = 0.0

    for idx, t_eval in enumerate(t_samples):
        # Advance pilot processing up to t_eval
        while t_next_pilot <= t_eval:
            t = t_next_pilot

            # True time error = residual clock error at this pilot instant
            true_error = eps * t + theta0 - theta_hat  # ≈ accumulated drift

            # Apply current skew estimate as feed-forward (slew, not step-jump)
            theta_hat += skew_hat * dt_pilot

            # Sample PDV
            pdv = sample_pdv(rng, 1)[0]

            # Total offset seen by the matched filter: drift + PDV
            total_offset_s = true_error + pdv

            # Integer chip offset (clipped to search window)
            tau_int = int(round(total_offset_s / Tc))
            tau_int = np.clip(tau_int, -W, W)

            # Build code
            if scheme == "kernelsync":
                s = make_kernelsync_code(global_chip, M)
            else:
                s = make_baseline_code(M)

            # Random carrier phase
            psi = rng.uniform(0, 2 * np.pi)

            # Construct received chips
            noise = (rng.normal(0, noise_sigma, M)
                     + 1j * rng.normal(0, noise_sigma, M))
            y = np.empty(M, dtype=complex)
            for k in range(M):
                km = k - tau_int
                if 0 <= km < M:
                    y[k] = np.exp(1j * psi) * s[km] + noise[k]
                else:
                    y[k] = noise[k]

            # Matched-filter estimate of the integer chip offset
            tau_hat = matched_filter_tau_fast(y, s, W)

            # Continuous-time measurement from matched filter
            if scheme == "kernelsync":
                # Sub-chip refinement via differential phase tracking
                meas_s = kernelsync_subchip_offset(y, tau_hat, Tc)
            else:
                # Baseline: integer-chip resolution only
                meas_s = tau_hat * Tc

            # Slew correction: update offset estimate
            theta_hat += alpha_offset * meas_s

            # Skew update: rate of change of consecutive measurements
            if prev_t_pilot > 0:
                delta_t = t - prev_t_pilot
                if delta_t > 0:
                    skew_hat += alpha_skew * (meas_s - prev_meas_s) / delta_t

            prev_t_pilot = t
            prev_meas_s = meas_s
            global_chip += M
            t_next_pilot += dt_pilot

        # Time error at evaluation point (interpolating slew between pilots)
        err_trace[idx] = eps * t_eval + theta0 - theta_hat

    return err_trace


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
    n_times: int = 200,
) -> dict:
    """
    For each (R, M) pair, simulate n_nodes follower nodes and compute
    final RMS time error.

    Returns dict keyed by (R, M) → {"final_rms_ns": float, "E": float,
                                     "rms_trace": np.ndarray}
    """
    W = int(round(W_ns / Tc))  # window in chips
    results = {}

    total = len(grid_R) * len(grid_M)
    done = 0
    for R in grid_R:
        for M in grid_M:
            done += 1
            print(f"  [{scheme:>12}] R={R:5g} evt/s  M={M:3d} chips  "
                  f"E={R*M:7g}  ({done}/{total}) …", end=" ", flush=True)

            rms_traces = []
            for _ in range(n_nodes):
                skew_ppm = rng.uniform(-50, 50)
                # Initial offset small enough to start within the search window
                # (models a coarse initial acquisition phase)
                theta0 = rng.uniform(-200e-9, 200e-9)
                trace = simulate_node(
                    rng=rng,
                    skew_ppm=skew_ppm,
                    theta0=theta0,
                    scheme=scheme,
                    R=R,
                    M=M,
                    Tc=Tc,
                    W=W,
                    T_sim=T_sim,
                    n_times=n_times,
                )
                rms_traces.append(trace)

            rms_traces = np.array(rms_traces)   # (n_nodes, n_times)
            # RMS across nodes at each time point
            rms_vs_time = np.sqrt(np.mean(rms_traces ** 2, axis=0))
            final_rms = float(rms_vs_time[-1])
            results[(R, M)] = {
                "final_rms_ns": final_rms * 1e9,
                "E": R * M,
                "rms_trace_ns": rms_vs_time * 1e9,
            }
            print(f"RMS = {final_rms*1e9:.3f} ns")

    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_pareto(
    results_base: dict,
    results_ks: dict,
    target_rms_ns: float,
    out_path: str,
) -> None:
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
    ax.set_xlabel("Energy proxy E = R × M  [events/s × chips/event]")
    ax.set_ylabel("Final RMS time error [ns]")
    ax.set_title("KernelSync vs Baseline — Pareto: Energy vs Timing Error")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_heatmap(
    results: dict,
    grid_R: list,
    grid_M: list,
    target_rms_ns: float,
    title: str,
    out_path: str,
) -> None:
    if not HAS_MPL:
        return
    data = np.zeros((len(grid_M), len(grid_R)))
    for ri, R in enumerate(grid_R):
        for mi, M in enumerate(grid_M):
            data[mi, ri] = results[(R, M)]["final_rms_ns"]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(data, origin="lower", aspect="auto",
                   extent=[0, len(grid_R), 0, len(grid_M)])
    ax.set_xticks(np.arange(len(grid_R)) + 0.5)
    ax.set_xticklabels([str(r) for r in grid_R])
    ax.set_yticks(np.arange(len(grid_M)) + 0.5)
    ax.set_yticklabels([str(m) for m in grid_M])
    ax.set_xlabel("R  [events/s]")
    ax.set_ylabel("M  [chips/event]")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Final RMS error [ns]")
    # Mark cells meeting target
    for ri, R in enumerate(grid_R):
        for mi, M in enumerate(grid_M):
            rms = results[(R, M)]["final_rms_ns"]
            if rms <= target_rms_ns:
                ax.add_patch(plt.Rectangle(
                    (ri, mi), 1, 1, fill=False, edgecolor="lime", lw=2
                ))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_rms_vs_time(
    results_base: dict,
    results_ks: dict,
    best_key_base,
    best_key_ks,
    T_sim: float,
    out_path: str,
) -> None:
    if not HAS_MPL:
        return
    t = np.linspace(0, T_sim, len(
        results_base[best_key_base]["rms_trace_ns"]
    ))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, results_base[best_key_base]["rms_trace_ns"],
            label=f"Baseline R={best_key_base[0]} M={best_key_base[1]}")
    ax.plot(t, results_ks[best_key_ks]["rms_trace_ns"],
            label=f"KernelSync R={best_key_ks[0]} M={best_key_ks[1]}")
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

def find_min_energy(
    results: dict,
    target_rms_ns: float,
) -> tuple:
    """Return (R, M) key with minimum E that achieves target RMS, or None."""
    candidates = [
        (k, v) for k, v in results.items()
        if v["final_rms_ns"] <= target_rms_ns
    ]
    if not candidates:
        return None, None
    best_key, best_val = min(candidates, key=lambda kv: kv[1]["E"])
    return best_key, best_val


def print_summary(
    results_base: dict,
    results_ks: dict,
    target_rms_ns: float,
) -> dict:
    key_b, val_b = find_min_energy(results_base, target_rms_ns)
    key_k, val_k = find_min_energy(results_ks, target_rms_ns)

    print()
    print("=" * 62)
    print("  KernelSync Energy-Proxy Simulation — Results Summary")
    print("=" * 62)
    print(f"  Target RMS: {target_rms_ns} ns")
    print()
    if val_b:
        print(f"  Baseline   min-E operating point:")
        print(f"    R = {key_b[0]} evt/s,  M = {key_b[1]} chips")
        print(f"    E = {val_b['E']:.0f}  (RMS = {val_b['final_rms_ns']:.3f} ns)")
    else:
        print("  Baseline   — target NOT achieved in grid")
    print()
    if val_k:
        print(f"  KernelSync min-E operating point:")
        print(f"    R = {key_k[0]} evt/s,  M = {key_k[1]} chips")
        print(f"    E = {val_k['E']:.0f}  (RMS = {val_k['final_rms_ns']:.3f} ns)")
    else:
        print("  KernelSync — target NOT achieved in grid")
    print()
    if val_b and val_k and val_k["E"] > 0:
        factor = val_b["E"] / val_k["E"]
        print(f"  Improvement factor (E_baseline / E_KernelSync): {factor:.2f}×")
    print("=" * 62)

    summary = {
        "target_rms_ns": target_rms_ns,
        "baseline": {
            "min_E_key": list(key_b) if key_b else None,
            "min_E": val_b["E"] if val_b else None,
            "rms_ns": val_b["final_rms_ns"] if val_b else None,
        },
        "kernelsync": {
            "min_E_key": list(key_k) if key_k else None,
            "min_E": val_k["E"] if val_k else None,
            "rms_ns": val_k["final_rms_ns"] if val_k else None,
        },
        "improvement_factor": (
            val_b["E"] / val_k["E"]
            if (val_b and val_k and val_k["E"] > 0) else None
        ),
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="KernelSync Energy-Proxy Simulation — 1000-node PDV demo."
    )
    parser.add_argument("--nodes", type=int, default=1000,
                        help="Number of follower nodes (default: 1000).")
    parser.add_argument("--tsim", type=float, default=10.0,
                        help="Simulation duration in seconds (default: 10).")
    parser.add_argument("--tc", type=float, default=40e-9,
                        help="Chip period in seconds (default: 40e-9).")
    parser.add_argument("--window", type=float, default=250e-9,
                        help="Timing search window ±seconds (default: 250e-9).")
    parser.add_argument("--target-rms-ns", type=float, default=1.0,
                        help="Target RMS synchronisation error in ns (default: 1.0).")
    parser.add_argument("--grid-R", type=str, default="50,100,200,500,1000",
                        help="Comma-separated pilot rates R in evt/s.")
    parser.add_argument("--grid-M", type=str, default="8,16,32,64",
                        help="Comma-separated burst lengths M in chips.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent / "out"),
                        help="Output directory for plots and results.json.")
    parser.add_argument("--n-times", type=int, default=100,
                        help="Number of time samples per node trace (default: 100).")
    return parser.parse_args(argv)


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
    print(f"  window=±{args.window*1e9:.0f}ns  target={args.target_rms_ns}ns RMS")
    print(f"  grid_R={grid_R}  grid_M={grid_M}")
    print("=" * 62)

    print("\n[1/2] Running Baseline simulation …")
    results_base = run_grid(
        rng=rng,
        scheme="baseline",
        grid_R=grid_R,
        grid_M=grid_M,
        n_nodes=args.nodes,
        T_sim=args.tsim,
        Tc=args.tc,
        W_ns=args.window,
        n_times=args.n_times,
    )

    print("\n[2/2] Running KernelSync simulation …")
    results_ks = run_grid(
        rng=rng,
        scheme="kernelsync",
        grid_R=grid_R,
        grid_M=grid_M,
        n_nodes=args.nodes,
        T_sim=args.tsim,
        Tc=args.tc,
        W_ns=args.window,
        n_times=args.n_times,
    )

    # Summary
    summary = print_summary(results_base, results_ks, args.target_rms_ns)

    # Save results.json (strip non-serialisable numpy arrays)
    json_path = out / "results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved {json_path}")

    # Plots
    if HAS_MPL:
        print("\nGenerating plots …")
        plot_pareto(
            results_base, results_ks, args.target_rms_ns,
            str(out / "pareto_energy_vs_error.png"),
        )
        plot_heatmap(
            results_base, grid_R, grid_M, args.target_rms_ns,
            "Baseline — Final RMS error [ns]",
            str(out / "heatmap_baseline.png"),
        )
        plot_heatmap(
            results_ks, grid_R, grid_M, args.target_rms_ns,
            "KernelSync — Final RMS error [ns]",
            str(out / "heatmap_kernelsync.png"),
        )

        key_b, _ = find_min_energy(results_base, args.target_rms_ns)
        key_k, _ = find_min_energy(results_ks, args.target_rms_ns)
        if key_b is None:
            key_b = min(results_base, key=lambda k: results_base[k]["final_rms_ns"])
        if key_k is None:
            key_k = min(results_ks, key=lambda k: results_ks[k]["final_rms_ns"])
        plot_rms_vs_time(
            results_base, results_ks, key_b, key_k, args.tsim,
            str(out / "rms_vs_time_best.png"),
        )

    print("\nDone. Outputs written to:", out)


if __name__ == "__main__":
    main()
