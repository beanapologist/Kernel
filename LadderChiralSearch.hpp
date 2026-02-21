/*
 * LadderChiralSearch.hpp — Ladder-based Chiral Search with optional Euler Kick
 *
 * Implements a ladder-step quantum search over n states using the Chiral
 * Non-Linear Gate balance primitive µ = e^{i3π/4}.  An optional Euler kick
 * (e^y − 1) is applied on the positive-imaginary domain when kick_base < 1.0,
 * providing super-linear amplification of the target-state amplitude.
 *
 * Integration:
 *   Include after ChiralNonlinearGate.hpp (QState must be in scope).
 *   Adjust kick_base to switch between linear and exponential regimes.
 */

#pragma once

#include <complex>
#include <cmath>
#include <cstddef>
#include <vector>

#include "ChiralNonlinearGate.hpp"

namespace kernel::quantum {

// ── LadderChiralSearch ────────────────────────────────────────────────────────
// Chiral ladder search over n candidate states.
//
// Members
//   target    — index of the marked state (0-based)
//   kick_base — Euler kick control:
//                 >= 1.0  → no exponential kick (linear / baseline)
//                 <  1.0  → Euler kick active: e^(Im(β)) − 1 applied on Im > 0
//
// Usage
//   LadderChiralSearch s;
//   s.target = 3;
//   s.kick_base = 1.0;   // baseline (no kick)
//   s.ladder_step(8);    // one step over 8 states
//
struct LadderChiralSearch {
    size_t target   = 0;
    double kick_base = 1.0;

    // ── ladder_step ───────────────────────────────────────────────────────────
    // Perform one ladder step over n states.
    //
    // Each call:
    //   1. Initialises n QState objects centred on the canonical coherent state.
    //   2. Applies a phase oracle: flips the sign of β for the target state.
    //   3. Applies the chiral non-linear rotation µ to every state:
    //        - kick_base >= 1.0  → kick_strength = 0 (purely linear)
    //        - kick_base <  1.0  → kick_strength = e^(Im(β)) − 1  (Euler kick)
    //   4. Returns the probability of measuring the target state after the step.
    //
    double ladder_step(size_t n) {
        if (n == 0) return 0.0;

        // Build state register: n copies of the canonical coherent state
        std::vector<QState> states(n);

        // Oracle: mark the target by phase-flipping its β component
        const size_t idx = target % n;
        states[idx].beta = -states[idx].beta;

        // Compute kick strength from kick_base
        //   kick_base >= 1.0 → purely linear (strength 0)
        //   kick_base <  1.0 → Euler kick: e^(Im(β)) − 1 evaluated at the
        //                      pre-rotation imaginary part of the target state
        const double kick_strength = (kick_base < 1.0)
            ? (std::exp(states[idx].beta.imag()) - 1.0)
            : 0.0;

        // Apply chiral rotation to all states
        for (auto& s : states) {
            s = chiral_nonlinear(s, kick_strength);
        }

        // Probability of measuring target ∝ |β|² of the (now-evolved) target
        const double p_target = std::norm(states[idx].beta);

        // Normalisation denominator
        double total = 0.0;
        for (const auto& s : states) {
            total += std::norm(s.beta);
        }

        return (total > 0.0) ? p_target / total : 0.0;
    }
};

} // namespace kernel::quantum
