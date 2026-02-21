/*
 * ChiralNonlinearGate.hpp — Chiral Non-Linear Gate for Quantum Pipeline
 *
 * Implements the chiral non-linear map that selectively shatters linearity
 * on the positive-imaginary domain while preserving classical reversibility
 * on the remaining domain.
 *
 * Mathematical foundation:
 *   Balance primitive µ = e^{i3π/4}: exact 135° rotation R(3π/4)
 *   Im(β) ≤ 0 domain: β' = µ·β              (linear — classical gates, reversible)
 *   Im(β) > 0 domain: β' = µ·β + k·(µ·β)·|µ·β|  (non-linear — reversibility shattered)
 *
 * Integration:
 *   Include this header after the QState definition in quantum_kernel_v2.cpp.
 *   Set coherence_kick_strength = 0.0 on Process for purely linear behaviour.
 */

#pragma once

#include <complex>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace kernel::quantum {

// µ = e^{i3π/4}: exact 135° balance rotation (mirrors MU in quantum_kernel_v2.cpp)
static constexpr double CHIRAL_ETA = 0.70710678118654752440;  // 1/√2
static const std::complex<double> CHIRAL_MU{ -CHIRAL_ETA, CHIRAL_ETA };

// ── chiral_nonlinear ──────────────────────────────────────────────────────────
// Apply the balance primitive µ with selective quadratic kick.
//
//   state         — QState to evolve (must be defined before this header)
//   kick_strength — magnitude of quadratic kick on Im > 0 half (0 = linear everywhere)
//
// Returns the evolved state; does NOT modify cycle_pos (caller's responsibility).
inline QState chiral_nonlinear(QState state, double kick_strength) {
    // Determine domain before applying rotation (Im of β prior to step)
    const bool positive_imag = (state.beta.imag() > 0.0);

    // Apply balance primitive µ = e^{i3π/4} (exact 135° rotation, Section 2)
    state.beta *= CHIRAL_MU;

    // Quadratic coherence kick on Im > 0 domain only — shatters linearity
    if (positive_imag && kick_strength != 0.0) {
        state.beta += kick_strength * state.beta * std::abs(state.beta);
    }

    return state;
}

// ── run_chiral_8cycle_demo ────────────────────────────────────────────────────
// 8-cycle validation: observe linear behaviour on Im ≤ 0 steps and quadratic
// magnitude growth on Im > 0 steps.
inline void run_chiral_8cycle_demo(double kick_strength = 0.1) {
    std::cout << "\n╔═══ Chiral Non-Linear Gate — 8-Cycle Demo ═══╗\n";
    std::cout << "  kick_strength = " << kick_strength << "\n";
    std::cout << std::fixed << std::setprecision(8);

    QState state;  // canonical coherent state: α = 1/√2, β = e^{i3π/4}/√2
    for (int i = 0; i < 8; ++i) {
        const bool pos_imag    = (state.beta.imag() > 0.0);
        const double mag_before = std::abs(state.beta);

        state = chiral_nonlinear(state, kick_strength);

        const double mag_after = std::abs(state.beta);
        std::cout << "  step " << i
                  << " [" << (pos_imag ? "Im>0 nonlin" : "Im≤0 linear") << "]"
                  << "  |β|: " << mag_before << " → " << mag_after
                  << "  Δ|β|=" << (mag_after - mag_before) << "\n";
    }
    std::cout << "╚═════════════════════════════════════════════╝\n";
}

} // namespace kernel::quantum
