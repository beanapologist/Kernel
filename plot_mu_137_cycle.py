#!/usr/bin/env python3
"""
plot_mu_137_cycle.py — Periodic-Cycle, Fourier Analysis and Scale Invariants for µ¹³⁷
=======================================================================================

Visualises three complementary aspects of µ = e^{i3π/4} validated by the C++
tests ``test_mu_137_phase_cycle``, ``test_mu_137_fourier``, and
``test_mu_137_scale_invariants`` in ``test_pipeline_theorems.cpp``:

  Panel 1 — Unit Circle Landmarks
      µ has period 8.  Because 137 ≡ 1 (mod 8), successive multiples of 137
      steps each advance the phase by exactly 135°, cycling through five
      landmark positions on the unit circle:

        µ⁸   (mod 8 = 0)  →  +1            (0°)
        µ¹³⁷ (mod 8 = 1)  →  µ             (135°)
        µ²⁷⁴ (mod 8 = 2)  →  µ² = -i       (270°)
        µ⁴¹¹ (mod 8 = 3)  →  µ³ = (1+i)/√2 (45°)
        µ⁵⁴⁸ (mod 8 = 4)  →  µ⁴ = -1       (180°)

      Directed arrows show the 135° phase advance between consecutive
      landmarks.  The full 8-step µ-orbit is shown as a reference.

  Panel 2 — Fourier Analysis: DFT of {µ⁰, µ¹, …, µ⁷}
      The 8-element sequence x[k] = µ^k = e^{i2π·(3/8)·k} is a pure
      complex sinusoid at normalised frequency 3/8.  Its DFT satisfies:

        |X[3]| = 8      (all energy concentrated at bin k = 3)
        |X[k]| = 0      for k ≠ 3

      Because 137 ≡ 1 (mod 8), the 137-step orbit generates the identical
      DFT (frequency aliasing).  Parseval's theorem,
      Σ|x[k]|² = (1/N)·Σ|X[n]|², is annotated on the plot.

  Panel 3 — Scale Invariants
      Three scale-invariant quantities are simultaneously preserved at every
      orbit point µ^k (k = 0…7):

        (a) |µ^k| = 1                  — magnitude invariant (unit circle)
        (b) C(r) = 2r/(1+r²)|_{r=1}=1 — coherence at its maximum
        (c) R(r) = (1/δ_S)(r-1/r)|_{r=1}=0 — palindrome residual vanishes

      Panel 3 makes these three invariants visible simultaneously:
        - a bar chart of |µ^k| (all bars at 1.0) with a reference line
        - the coherence function C(r) plotted for r ∈ [0.2, 3] with the
          r = 1 operating point highlighted

Usage (from the repo root):
    python3 plot_mu_137_cycle.py

Output:
    mu_137_cycle.png  — 1×3 panel figure (saved to current directory)
"""

import cmath
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

# ── Constants ─────────────────────────────────────────────────────────────────
PI = math.pi
MU_ANGLE = 3.0 * PI / 4.0          # arg(µ) = 135°
MU = cmath.exp(1j * MU_ANGLE)       # µ = e^{i3π/4} = (-1+i)/√2
MU_CYCLE = 8                        # µ has period 8

# Five landmark powers (exponents chosen to illustrate 137-step progression)
LANDMARK_POWERS = [8, 137, 274, 411, 548]
LANDMARK_LABELS = ["µ⁸", "µ¹³⁷", "µ²⁷⁴", "µ⁴¹¹", "µ⁵⁴⁸"]

# Colours for the five landmarks
LANDMARK_COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]

# Radial scaling factor for landmark text labels (places them outside the unit circle)
LABEL_RADIUS_SCALE = 1.28

# Silver ratio δ_S = 1 + √2 (used in the palindrome residual R(r))
DELTA_S = 1.0 + math.sqrt(2.0)


def _mu_power(n: int) -> complex:
    """Return µ^n using the period-8 shortcut: µ^n = µ^(n mod 8)."""
    angle = (n % MU_CYCLE) * MU_ANGLE
    return cmath.exp(1j * angle)


# ── Panel 1: Unit Circle Landmarks ───────────────────────────────────────────
def plot_unit_circle_landmarks(ax: "plt.Axes") -> None:
    """
    Plot the five µ¹³⁷-cycle landmarks on the unit circle.

    Each successive landmark is reached after 137 additional powers of µ.
    Because 137 ≡ 1 (mod 8) the phase advances by exactly arg(µ) = 135°
    per landmark step.  Directed arrows connect consecutive landmarks to
    emphasise the constant angular increment.
    """
    theta = np.linspace(0, 2 * PI, 500)
    ax.plot(np.cos(theta), np.sin(theta),
            color="lightgray", lw=1.0, zorder=1)

    # Full 8-step µ-orbit (faint reference points)
    for j in range(MU_CYCLE):
        ang = j * MU_ANGLE
        ax.scatter(math.cos(ang), math.sin(ang),
                   color="lightgray", s=20, zorder=2)

    # Axis lines
    ax.axhline(0, color="lightgray", lw=0.6, zorder=1)
    ax.axvline(0, color="lightgray", lw=0.6, zorder=1)

    # Compute landmark positions
    pts = [_mu_power(p) for p in LANDMARK_POWERS]

    # Directed arrows between consecutive landmarks
    arrow_kw = dict(arrowstyle="-|>", color="gray", lw=1.0,
                    mutation_scale=12, zorder=3)
    for i in range(len(pts) - 1):
        ax.annotate(
            "",
            xy=(pts[i + 1].real, pts[i + 1].imag),
            xytext=(pts[i].real, pts[i].imag),
            arrowprops=arrow_kw,
        )

    # Plot and label each landmark
    for i, (p, label, color) in enumerate(
        zip(pts, LANDMARK_LABELS, LANDMARK_COLORS)
    ):
        x, y = p.real, p.imag
        phase_deg = math.degrees(cmath.phase(p)) % 360
        ax.scatter(x, y, color=color, s=100, zorder=5, edgecolors="white", lw=0.8)

        # Position label slightly outside the circle
        ax.annotate(
            f"{label}\n{LANDMARK_POWERS[i] % MU_CYCLE} mod 8\n{phase_deg:.0f}°",
            xy=(x, y),
            xytext=(x * LABEL_RADIUS_SCALE, y * LABEL_RADIUS_SCALE),
            fontsize=7.5,
            ha="center", va="center",
            color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color,
                      alpha=0.85, lw=0.7),
        )

    # Modular arithmetic note inside the circle
    ax.text(
        0, -0.42,
        "137 mod 8 = 1\n+135° per landmark",
        ha="center", va="center", fontsize=8.5,
        color="#555555",
    )

    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.65, 1.65)
    ax.set_aspect("equal")
    ax.set_title(
        r"Unit Circle Landmarks: $\mu^{8},\,\mu^{137},\,\mu^{274},\,\mu^{411},\,\mu^{548}$",
        fontsize=10,
    )
    ax.set_xlabel("Re", fontsize=9)
    ax.set_ylabel("Im", fontsize=9)
    ax.tick_params(labelsize=8)


# ── Panel 2: Fourier Analysis ─────────────────────────────────────────────────
def plot_fourier_analysis(ax: "plt.Axes") -> None:
    """
    Stem plot of |DFT({µ⁰, µ¹, …, µ⁷})| showing the dominant frequency bin.

    x[k] = µ^k = e^{i2π·(3/8)·k} is a pure complex sinusoid at normalised
    frequency 3/8, so the DFT concentrates all energy in bin k = 3.

    The Parseval annotation confirms energy equivalence:
        Σ|x[k]|² = (1/N)·Σ|X[n]|²  (both equal N = 8).
    """
    N = MU_CYCLE
    # Time-domain sequence x[k] = µ^k
    x = np.array([_mu_power(k) for k in range(N)])

    # DFT
    X = np.fft.fft(x)
    magnitudes = np.abs(X)
    freq_bins = np.arange(N)

    # Stem plot
    markerline, stemlines, baseline = ax.stem(
        freq_bins, magnitudes, linefmt="C0-", markerfmt="C0o", basefmt="gray"
    )
    plt.setp(markerline, markersize=6)
    plt.setp(stemlines, lw=1.4)

    # Highlight the dominant bin k = 3
    dominant_k = 3
    ax.stem(
        [dominant_k], [magnitudes[dominant_k]],
        linefmt="C1-", markerfmt="C1D", basefmt="none",
        label=f"Dominant bin k={dominant_k}  |X[{dominant_k}]| = {N}",
    )

    # Annotate |X[3]| = 8
    ax.annotate(
        f"|X[{dominant_k}]| = {N}",
        xy=(dominant_k, magnitudes[dominant_k]),
        xytext=(dominant_k + 0.5, magnitudes[dominant_k] - 0.6),
        fontsize=9, color="C1",
        arrowprops=dict(arrowstyle="->", color="C1", lw=1.0),
    )

    # Parseval verification
    energy_time = float(np.sum(np.abs(x) ** 2))     # = N  (each |µ^k| = 1)
    energy_freq = float(np.sum(magnitudes ** 2) / N) # = N
    ax.text(
        0.97, 0.97,
        f"Parseval's theorem\n"
        f"$\\Sigma|x|^2 = {energy_time:.0f}$\n"
        f"$(1/N)\\cdot\\Sigma|X|^2 = {energy_freq:.0f}$",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                  ec="#ccaa00", alpha=0.9),
    )

    # µ frequency label on x-axis
    ax.set_xticks(freq_bins)
    ax.set_xticklabels(
        [f"k={k}  ← µ" if k == dominant_k else f"k={k}"
         for k in freq_bins],
        rotation=35, ha="right", fontsize=8,
    )
    ax.set_ylim(-0.3, N + 1.2)
    ax.set_ylabel("|X[k]|", fontsize=9)
    ax.set_title(
        r"DFT of $\{µ^0, µ^1, \ldots, µ^7\}$: pure tone at $k=3$",
        fontsize=10,
    )
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(axis="y", alpha=0.35)
    ax.tick_params(axis="y", labelsize=8)


# ── Panel 3: Scale Invariants ─────────────────────────────────────────────────
def plot_scale_invariants(ax: "plt.Axes") -> None:
    """
    Panel showing the three scale invariants of the µ orbit.

    Main area — bar chart of |µ^k| for k = 0..7 (all bars exactly 1.0),
    demonstrating that the orbit lies entirely on the unit circle.

    Inset — coherence C(r) = 2r/(1+r²) and palindrome residual
    R(r) = (r − 1/r)/δ_S plotted against r ∈ [0.2, 3.0].  Both curves
    are evaluated directly on the r axis (no offset); a vertical guide at
    r = 1 marks the operating point where C = 1 (maximum) and R = 0.

    At r = 1 all three quantities lock simultaneously:
        |µ^k| = 1,   C(1) = 1  (maximum),   R(1) = 0.
    """
    N = MU_CYCLE
    orbit_mags = np.array([abs(_mu_power(k)) for k in range(N)])

    # ── Main: |µ^k| bar chart ─────────────────────────────────────────────────
    ks = np.arange(N)
    ax.bar(ks, orbit_mags, color="steelblue", alpha=0.7, width=0.5,
           label=r"|µ$^k$| = 1  (unit circle invariant)")
    ax.axhline(1.0, color="steelblue", lw=1.2, ls="--", alpha=0.7)
    ax.set_ylim(0, 1.6)
    ax.set_xlim(-0.6, N - 0.4)
    ax.set_xlabel("Orbit index k", fontsize=9)
    ax.set_ylabel(r"|µ$^k$|", fontsize=9)
    ax.set_xticks(ks)
    ax.tick_params(labelsize=8)
    ax.text(N / 2 - 0.5, 1.09, "|µ^k| = 1  (unit circle)",
            ha="center", va="bottom", fontsize=8, color="steelblue")

    # ── Inset: C(r) and R(r) vs r (own x-axis, no shared-axis offset needed) ──
    axins = ax.inset_axes([0.50, 0.40, 0.48, 0.54])
    r = np.linspace(0.2, 3.0, 400)
    C_r = 2.0 * r / (1.0 + r ** 2)
    R_r = (r - 1.0 / r) / DELTA_S

    axins.plot(r, C_r, color="C1", lw=1.6, label="C(r) = 2r/(1+r²)")
    axins.plot(r, R_r, color="C2", lw=1.2, ls="--",
               label="R(r) = (r − 1/r)/δ_S")

    # r = 1 operating point: C(1)=1, R(1)=0
    axins.scatter([1.0], [1.0], color="C1", s=60, zorder=5)
    axins.scatter([1.0], [0.0], color="C2", s=60, zorder=5)
    axins.axvline(1.0, color="gray", lw=0.8, ls=":", zorder=1)
    axins.axhline(0.0, color="gray", lw=0.5, alpha=0.5)

    axins.annotate("C(1) = 1\nR(1) = 0",
                   xy=(1.0, 0.5), xytext=(1.6, 0.45),
                   fontsize=7, color="#444444",
                   arrowprops=dict(arrowstyle="->", color="#444444", lw=0.8))

    axins.set_xlabel("r  =  |µ^k|", fontsize=7.5)
    axins.set_xlim(0.2, 3.0)
    axins.set_ylim(-0.9, 1.25)
    axins.tick_params(labelsize=7)
    axins.legend(fontsize=6.5, loc="upper right")

    # ── Invariant summary box ──────────────────────────────────────────────────
    ax.text(
        0.02, 0.97,
        "Scale invariants at r = |µ^k| = 1:\n"
        "  |µ^k| = 1       (unit circle)\n"
        "  C(1)  = 1       (max coherence)\n"
        "  R(1)  = 0       (palindrome residual)",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                  ec="#ccaa00", alpha=0.9),
    )

    handles_l, labels_l = ax.get_legend_handles_labels()
    ax.legend(handles_l, labels_l, fontsize=7.5, loc="upper center")
    ax.set_title(
        r"Scale Invariants: $|\mu^k|=1$,  $C(1)=1$,  $R(1)=0$",
        fontsize=10,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main(out_path: str = "mu_137_cycle.png") -> None:
    """Generate and save the three-panel visualisation."""
    if not HAS_MPL:
        print("[error] matplotlib is required — install with: pip install matplotlib")
        sys.exit(1)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fig.suptitle(
        r"Periodic Cycle of $\mu = e^{i3\pi/4}$ under Successive Powers"
        "\n"
        r"$\mu^{137}$ phase shift: 137 × 135° mod 360° = 135°  "
        r"(137 ≡ 1 mod 8)",
        fontsize=11,
    )

    plot_unit_circle_landmarks(axes[0])
    plot_fourier_analysis(axes[1])
    plot_scale_invariants(axes[2])

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
