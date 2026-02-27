#!/usr/bin/env python3
"""
plot_torus_topology.py — Geometric Topology Visualiser for the 8+1/Δ Conjecture
=================================================================================

Generates four panels that expose the geometric/topological structure of the
palindrome-quotient conjecture embedded in MasterEigenOracle:

  Panel 1 — µ-Orbit on the Unit Circle
      The 8 eigenspace channels µ^j (j=0…7) as equally-spaced points on S¹.
      µ = e^{i3π/4} rotates 135° per step; the 8-element orbit covers 45°
      increments, closing exactly at µ^8 = 1 (Proposition 2.2).

  Panel 2 — Torus T² Topology (Lissajous winding)
      Two independent angular coordinates (θ_fast, θ_slow) parametrise the
      T² torus.  The fast coordinate winds at 3π/4 rad/step (µ-rotation);
      the slow coordinate winds at ΔΦ = 2π/Δ rad/step (PalindromePrecession).
      Plotting (θ_fast mod 2π, θ_slow mod 2π) for the first 512 steps reveals
      the Lissajous-like winding pattern on the flat-torus [0,2π)².

  Panel 3 — ε-Symmetry-Breaking: Phase Drift vs Step
      Shows the cumulative PalindromePrecession phase (θ_slow = k·ΔΦ) for
      k = 0…7 in red (the 8 fast-cycle steps), and the residual drift from
      a pure 8-cycle baseline (8 equally-spaced steps) in blue.
      The offset ε = 1/Δ ≈ 7.29×10⁻⁸ rad/step is the symmetry-breaking
      perturbation that prevents exact 8-cycle periodicity.

  Panel 4 — CoherenceHarvest: G_eff-weighted Accumulator Trajectories
      Simulates oracle accumulation for a target at θ_t = π/3 over 200 steps,
      plotting each of the 8 channel accumulators.  The channel closest to
      θ_t should grow fastest, demonstrating coherent amplification.

Constants used (from MasterEigenOracle.hpp / PalindromePrecession.hpp):
  DELTA         = 13 717 421       (palindrome denominator Δ)
  EPSILON       = 1/Δ ≈ 7.29e-8   (fine-tuning perturbation ε)
  ORACLE_RATE   = 8 + ε            (palindrome quotient 987654321/123456789)
  SUPER_PERIOD  = 8 × Δ            (torus T² full realignment ≈ 109M steps)
  MU            = e^{i3π/4}        (balanced eigenvalue)

Usage (from the repo root):
    python3 plot_torus_topology.py

Output:
    torus_topology.png  — 2×2 panel figure (saved to current directory)
"""

import math
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Physical constants (mirror MasterEigenOracle.hpp) ────────────────────────
PI           = math.pi
TWO_PI       = 2.0 * PI
DELTA        = 13_717_421                   # palindrome denominator Δ
EPSILON      = 1.0 / DELTA                  # fine-tuning perturbation ε ≈ 7.29e-8
ORACLE_RATE  = 8.0 + EPSILON                # palindrome quotient = 8 + 1/Δ
SUPER_PERIOD = 8 * DELTA                    # torus T² super-period
DELTA_PHI    = TWO_PI / DELTA               # PalindromePrecession phase step
MU_ANGLE     = 3.0 * PI / 4.0              # µ = e^{i3π/4} → 135° per step
ETA          = math.sqrt(0.5)               # 1/√2


def _mu_orbit():
    """Return the 8-element µ-orbit as (real, imag) pairs."""
    pts = []
    angle = 0.0
    for _ in range(8):
        pts.append((math.cos(angle), math.sin(angle)))
        angle += MU_ANGLE
    return pts


# ── Panel 1: µ-Orbit on the Unit Circle ──────────────────────────────────────
def plot_mu_orbit(ax):
    orbit = _mu_orbit()

    # Unit circle
    theta = np.linspace(0, TWO_PI, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="lightgray", lw=1.2, zorder=1)

    colors = plt.cm.hsv(np.linspace(0, 1, 8, endpoint=False))
    for j, (x, y) in enumerate(orbit):
        ax.scatter(x, y, color=colors[j], s=90, zorder=3)
        ax.annotate(
            f"µ^{j}", xy=(x, y),
            xytext=(x * 1.22, y * 1.22),
            fontsize=8, ha="center", va="center", color=colors[j]
        )
        # Draw arrow for each rotation step
        if j < 7:
            nx, ny = orbit[j + 1]
            ax.annotate(
                "", xy=(nx * 0.92, ny * 0.92),
                xytext=(x * 0.92, y * 0.92),
                arrowprops=dict(arrowstyle="->", color=colors[j], lw=1.0),
                zorder=2
            )

    # Close the orbit: µ^7 → µ^0
    x7, y7 = orbit[7]
    x0, y0 = orbit[0]
    ax.annotate(
        "", xy=(x0 * 0.92, y0 * 0.92),
        xytext=(x7 * 0.92, y7 * 0.92),
        arrowprops=dict(arrowstyle="->", color=colors[7], lw=1.0, linestyle="dashed"),
        zorder=2
    )

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axhline(0, color="lightgray", lw=0.5)
    ax.axvline(0, color="lightgray", lw=0.5)
    ax.set_title("µ-Orbit on Unit Circle\n"
                 r"$\mu = e^{i3\pi/4}$, period 8, 45° steps",
                 fontsize=9)
    ax.set_xlabel("Re", fontsize=8)
    ax.set_ylabel("Im", fontsize=8)
    ax.tick_params(labelsize=7)


# ── Panel 2: Torus T² Winding ─────────────────────────────────────────────────
def plot_torus_winding(ax):
    """
    Flat-torus [0,2π)² plot.
      x-axis: θ_fast = (3π/4 × step) mod 2π  (fast µ-rotation)
      y-axis: θ_slow = ΔΦ × step mod 2π       (slow PalindromePrecession)
    Colour encodes time (step index).
    """
    N_STEPS = 512
    fast = np.array([(MU_ANGLE * k) % TWO_PI for k in range(N_STEPS)])
    slow = np.array([(DELTA_PHI  * k) % TWO_PI for k in range(N_STEPS)])

    sc = ax.scatter(fast, slow, c=np.arange(N_STEPS), cmap="viridis",
                    s=6, linewidths=0, alpha=0.85)

    # Mark the 8 fast-cycle positions (k = 0…7) in red
    for k in range(8):
        ax.scatter((MU_ANGLE * k) % TWO_PI, (DELTA_PHI * k) % TWO_PI,
                   c="red", s=40, zorder=5)

    cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Step k", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlim(0, TWO_PI)
    ax.set_ylim(0, TWO_PI)
    ax.set_xlabel(r"$\theta_\mathrm{fast}\ (\mathrm{mod}\ 2\pi)$", fontsize=8)
    ax.set_ylabel(r"$\theta_\mathrm{slow}\ (\mathrm{mod}\ 2\pi)$", fontsize=8)
    ax.set_title(
        r"Torus $T^2$ Winding (flat torus, 512 steps)" "\n"
        r"Fast: $\frac{3\pi}{4}$/step  ·  Slow: $\Delta\Phi$/step",
        fontsize=9
    )
    ax.tick_params(labelsize=7)

    # Mark fast-cycle grid lines at multiples of 45°
    for j in range(9):
        ax.axvline(j * PI / 4, color="gray", lw=0.4, alpha=0.4)

    # Annotate super-period annotation
    ax.text(0.02, 0.97,
            f"Super-period: 8×Δ = {SUPER_PERIOD:,d} steps",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))


# ── Panel 3: ε-Symmetry-Breaking Phase Drift ─────────────────────────────────
def plot_epsilon_drift(ax):
    """
    Shows the PalindromePrecession cumulative phase for steps k=0…63,
    compared to a hypothetical exact-8-cycle baseline (zero slow drift).
    The difference is the ε-drift = k × ΔΦ.
    """
    steps = np.arange(65)
    # Actual slow phase (PalindromePrecession)
    slow_phase = steps * DELTA_PHI

    # Exact-8-cycle baseline: slow phase would be zero (pure fast cycle)
    baseline = np.zeros_like(slow_phase)

    ax.plot(steps, slow_phase * 1e7, color="steelblue", lw=1.5,
            label=r"Cumulative $\varepsilon$-drift $= k \cdot \Delta\Phi$")
    ax.plot(steps, baseline, color="gray", lw=1.0, linestyle="--",
            label="Exact 8-cycle baseline (ΔΦ = 0)")

    # Annotate the 8-step mark
    drift_at_8 = 8 * DELTA_PHI * 1e7
    ax.annotate(
        f"k=8: drift={8*DELTA_PHI:.3e} rad\n"
        r"$\ll 2\pi$ — periodicity broken",
        xy=(8, drift_at_8), xytext=(18, drift_at_8 * 2.5),
        fontsize=7,
        arrowprops=dict(arrowstyle="->", color="steelblue", lw=0.8),
        color="steelblue"
    )

    # Shade fast-cycle groups
    for c in range(8):
        ax.axvspan(c * 8, c * 8 + 8, alpha=0.04,
                   color="orange" if c % 2 == 0 else "blue")

    ax.set_xlabel("Step k", fontsize=8)
    ax.set_ylabel(r"Phase $(\times 10^{-7}$ rad$)$", fontsize=8)
    ax.set_title(
        r"$\varepsilon$-Symmetry-Breaking: Cumulative Phase Drift" "\n"
        r"$\varepsilon = 1/\Delta \approx 7.29 \times 10^{-8}$ rad/step",
        fontsize=9
    )
    ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, 64)

    ax.text(0.98, 0.04,
            fr"$\Delta = {DELTA:,d}$  ·  $\varepsilon = {EPSILON:.3e}$",
            transform=ax.transAxes, fontsize=7, ha="right",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))


# ── Panel 4: CoherenceHarvest Accumulator Trajectories ───────────────────────
def plot_coherence_harvest(ax):
    """
    Simulate 8-channel oracle accumulation for θ_t = π/3 (60°) over n steps.
    Uses G_eff = 1 (canonical coherent state) and ΔΦ = 2π/√n (Dirichlet
    resonance condition, theta_sqrt_n_writeup.tex §2).
    """
    N_STEPS = 200
    THETA_TARGET = PI / 3.0  # 60°

    # Dirichlet-kernel resonance step: ΔΦ = 2π/√n
    dirichlet_delta_phi = TWO_PI / math.sqrt(N_STEPS)
    target_phasor = (math.cos(THETA_TARGET), math.sin(THETA_TARGET))

    # Pre-compute µ-orbit phasors
    mu_orbit = []
    angle = 0.0
    for _ in range(8):
        mu_orbit.append((math.cos(angle), math.sin(angle)))
        angle += MU_ANGLE

    # Accumulate
    accumulators = np.zeros((8, N_STEPS + 1))
    for k in range(N_STEPS):
        slow_angle = k * dirichlet_delta_phi
        sp = (math.cos(slow_angle), math.sin(slow_angle))
        for j in range(8):
            mx, my = mu_orbit[j]
            # probe = slow_phasor × µ^j  (complex multiply)
            px = sp[0] * mx - sp[1] * my
            py = sp[0] * my + sp[1] * mx
            # contrib = Re(probe × conj(target))
            contrib = px * target_phasor[0] + py * target_phasor[1]
            accumulators[j, k + 1] = accumulators[j, k] + contrib

    steps_axis = np.arange(N_STEPS + 1)
    colors = plt.cm.hsv(np.linspace(0, 1, 8, endpoint=False))
    for j in range(8):
        angle_j = j * MU_ANGLE
        lw = 2.0 if abs(angle_j % TWO_PI - THETA_TARGET) < 0.8 else 0.9
        ax.plot(steps_axis, accumulators[j], color=colors[j], lw=lw,
                label=f"ch {j} (µ^{j})")

    # Detection threshold 0.15√n
    threshold = 0.15 * math.sqrt(N_STEPS)
    ax.axhline(threshold, color="red", lw=1.2, linestyle="--",
               label=f"threshold 0.15√n ≈ {threshold:.1f}")
    ax.axhline(-threshold, color="red", lw=1.2, linestyle="--")

    ax.set_xlabel("Step k", fontsize=8)
    ax.set_ylabel("Accumulator A[j]", fontsize=8)
    ax.set_title(
        r"CoherenceHarvest: 8-Channel Accumulation ($\theta_t = \pi/3$)" "\n"
        r"$G_\mathrm{eff}=1$, $\Delta\Phi = 2\pi/\sqrt{n}$, $n=200$",
        fontsize=9
    )
    ax.legend(fontsize=6, ncol=2, loc="lower left")
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, N_STEPS)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  plot_torus_topology.py — 8+1/Δ Geometric Topology           ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    print(f"  Constants:")
    print(f"    Δ  = {DELTA:,d}")
    print(f"    ε  = {EPSILON:.6e}  (1/Δ)")
    print(f"    8+ε = {ORACLE_RATE:.10f}  (palindrome quotient)")
    print(f"    ΔΦ = {DELTA_PHI:.6e} rad/step")
    print(f"    Super-period = 8×Δ = {SUPER_PERIOD:,d} steps")

    if not HAS_MPL:
        print("\n  [error] matplotlib not available — cannot generate PNG.")
        print("          Install with: pip install matplotlib numpy")
        sys.exit(1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        r"Geometric Topology of the $8 + 1/\Delta$ Palindrome-Quotient Conjecture"
        "\n"
        r"$987654321 / 123456789 = 8 + 1/\Delta$,  $\Delta = 13\,717\,421$,"
        r"  $\varepsilon = 1/\Delta \approx 7.29 \times 10^{-8}$",
        fontsize=11, y=1.01
    )

    plot_mu_orbit(axes[0, 0])
    plot_torus_winding(axes[0, 1])
    plot_epsilon_drift(axes[1, 0])
    plot_coherence_harvest(axes[1, 1])

    # Add panel labels
    for idx, (ax, label) in enumerate(zip(axes.flat,
                                          ["(1)", "(2)", "(3)", "(4)"])):
        ax.text(-0.08, 1.02, label, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom")

    fig.tight_layout()
    out = "torus_topology.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ {out}  written.")
    print()


if __name__ == "__main__":
    main()
