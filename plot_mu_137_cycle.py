#!/usr/bin/env python3
"""
plot_mu_137_cycle.py — Periodic-Cycle and Fourier Analysis Visualiser for µ¹³⁷
================================================================================

Visualises two complementary aspects of µ = e^{i3π/4} validated by the C++
tests ``test_mu_137_phase_cycle`` and ``test_mu_137_fourier`` in
``test_pipeline_theorems.cpp``:

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

Usage (from the repo root):
    python3 plot_mu_137_cycle.py

Output:
    mu_137_cycle.png  — 1×2 panel figure (saved to current directory)
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main(out_path: str = "mu_137_cycle.png") -> None:
    """Generate and save the two-panel visualisation."""
    if not HAS_MPL:
        print("[error] matplotlib is required — install with: pip install matplotlib")
        sys.exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        r"Periodic Cycle of $\mu = e^{i3\pi/4}$ under Successive Powers"
        "\n"
        r"$\mu^{137}$ phase shift: 137 × 135° mod 360° = 135°  "
        r"(137 ≡ 1 mod 8)",
        fontsize=11,
    )

    plot_unit_circle_landmarks(axes[0])
    plot_fourier_analysis(axes[1])

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
