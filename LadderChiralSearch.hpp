/*
 * LadderChiralSearch.hpp — Ladder-based Chiral Search with optional Euler Kick
 *
 * Implements a ladder-step quantum search over n states using the Chiral
 * Non-Linear Gate balance primitive µ = e^{i3π/4}.  An optional Euler kick
 * (e^y − 1) is applied on the positive-imaginary domain when kick_base < 1.0,
 * providing super-linear amplification of the target-state amplitude.
 *
 * The β-amplitude register is stored as an Eigen::VectorXcd, enabling
 * efficient norm computation and future matrix-operator extensions.
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

#include <Eigen/Dense>

#include "ChiralNonlinearGate.hpp"

namespace kernel::quantum {

// ── StepResult ────────────────────────────────────────────────────────────────
// Return value of LadderChiralSearch::ladder_step.
//
//   p_target  — normalised probability of measuring the target state
//   coherence — average C(r) = 2r/(1+r²) across all n states (Theorem 11)
//               where r_i = |β_i| / |α_i|; higher means more coherent ensemble
//
struct StepResult {
    double p_target  = 0.0;
    double coherence = 0.0;
};

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
    size_t target    = 0;
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
    //   4. Stores β amplitudes in an Eigen::VectorXcd for efficient norm computation.
    //   5. Returns StepResult{p_target, C(r)} for the evolved target state.
    //
    StepResult ladder_step(size_t n) {
        if (n == 0) return {};

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

        // Collect β amplitudes into an Eigen vector for norm operations
        Eigen::VectorXcd betas(static_cast<Eigen::Index>(n));
        for (size_t i = 0; i < n; ++i) {
            betas[static_cast<Eigen::Index>(i)] = states[i].beta;
        }

        // Normalised probability of measuring target: |β_target|² / ‖β‖²
        const double total    = betas.squaredNorm();
        const double p_target = (total > 0.0)
            ? std::norm(betas[static_cast<Eigen::Index>(idx)]) / total
            : 0.0;

        // Average coherence C(r) = 2r/(1+r²) across all n states (Theorem 11).
        // r_i = |β_i| / |α_i|; α is unchanged by chiral_nonlinear, so |α| = CHIRAL_ETA.
        // Averaging across states reveals how the kick redistributes coherence
        // between the marked target and the unmarked background.
        double coh_sum = 0.0;
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(n); ++i) {
            const double r_i     = std::abs(betas[i]) / CHIRAL_ETA;
            const double denom_i = 1.0 + r_i * r_i;
            coh_sum += (2.0 * r_i) / denom_i;
        }
        const double coherence = coh_sum / static_cast<double>(n);

        return {p_target, coherence};
    }
};

} // namespace kernel::quantum
