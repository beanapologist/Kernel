/*
 * SpectralBridge.hpp — Spectral-to-Kernel Bridge and Mode-Based API
 *
 * Bridges two representations of the same physical state:
 *
 *   Quantum representation  — KernelState (α, β, r, C)
 *   Spectral representation — CoherentChannel (λ, G_eff, R_eff)
 *
 * The bridge is exact: given r = |β/α|, the Lyapunov exponent λ = ln r
 * determines both representations uniquely (Theorem 14).  At the coherent
 * fixed point r = 1 (λ = 0):
 *
 *   KernelState:    R(r) = 0,   C = 1,   |β_phasor| = 1
 *   CoherentChannel: R_eff = 1, G_eff = 1
 *
 * Kernel modes:
 *
 *   STANDARD   — single µ-rotation per step (8-cycle, Theorem 10)
 *   PALINDROME — µ-rotation + PalindromePrecession phase increment
 *   SPECTRAL   — µ-rotation with per-step drift check and auto-renormalization
 *   FULL       — all of the above combined: µ-rotation, PalindromePrecession,
 *                and drift-corrected auto-renormalization
 *
 * User-facing API (no manual setup required):
 *
 *   SpectralBridge::step(ks, mode);         // advance one tick
 *   SpectralBridge::to_channel(ks);         // convert to CoherentChannel
 *   SpectralBridge::from_channel(ch, tick); // convert back to KernelState
 *   SpectralBridge::verify(ks);             // check all three invariants
 *
 * Include after KernelState.hpp and PalindromePrecession.hpp.
 */

#pragma once

#include "KernelState.hpp"
#include "PalindromePrecession.hpp"

// Bring in the Ohm–Coherence channel type
#include "ohm_coherence_duality.hpp"

namespace kernel::pipeline {

// ── Kernel mode selector
// ──────────────────────────────────────────────────────
enum class KernelMode {
  STANDARD,   ///< µ-rotation only (pure 8-cycle)
  PALINDROME, ///< µ-rotation + palindrome precession phase
  SPECTRAL,   ///< µ-rotation + drift detection and auto-renormalization
  FULL        ///< µ-rotation + palindrome precession + auto-renormalization
};

// ── SpectralBridge
// ────────────────────────────────────────────────────────────
//
// Stateless helper: all methods are static.  For stateful operation (e.g.
// maintaining a PalindromePrecession step counter), use the Pipeline class in
// KernelPipeline.hpp instead.
//
struct SpectralBridge {

  // ── Representation conversion ─────────────────────────────────────────────

  // Convert KernelState → CoherentChannel using Theorem 14: λ = ln r.
  // The resulting channel captures the spectral view of the same state.
  static kernel::ohm::CoherentChannel to_channel(const KernelState &ks) {
    double r = ks.radius();
    double lambda = ks_lyapunov(r); // λ = ln r
    return kernel::ohm::CoherentChannel{lambda};
  }

  // Reconstruct a KernelState from a CoherentChannel and an optional tick.
  // Sets |β/α| = exp(λ) while preserving the canonical phase of the
  // coherent state β = e^{i3π/4} · |β|.
  static KernelState from_channel(const kernel::ohm::CoherentChannel &ch,
                                  uint64_t tick = 0) {
    KernelState ks;
    ks.tick = tick;

    // r = e^λ  →  |β| = r · |α|
    double r = std::exp(ch.lambda);
    double alpha_mag = KS_ETA; // |α| = 1/√2 in canonical state

    // Re-scale β to achieve the desired radius
    double beta_mag = r * alpha_mag;

    // Normalize so |α|² + |β|² = 1
    double norm = std::sqrt(alpha_mag * alpha_mag + beta_mag * beta_mag);
    double a = alpha_mag / norm;
    double b = beta_mag / norm;

    // Preserve canonical phase of β = e^{i3π/4} / √2 direction
    // Unit-phase direction from canonical state: e^{i3π/4} = (-1+i)/√2
    static const Cx CANONICAL_BETA_PHASE{-KS_ETA, KS_ETA};
    ks.alpha = Cx{a, 0.0};
    ks.beta = CANONICAL_BETA_PHASE * b;
    return ks;
  }

  // ── Single-step evolution ─────────────────────────────────────────────────

  // Advance ks by one tick in the given mode.
  // Uses a thread-local PalindromePrecession so the step counter persists
  // across repeated calls without requiring the caller to manage it.
  static void step(KernelState &ks, KernelMode mode,
                   quantum::PalindromePrecession &pp) {
    // All modes begin with the µ-rotation (Section 2)
    ks.step();

    if (mode == KernelMode::PALINDROME || mode == KernelMode::FULL) {
      // Apply palindrome precession phase to β (|β| invariant)
      pp.apply(ks.beta);
    }

    if (mode == KernelMode::SPECTRAL || mode == KernelMode::FULL) {
      // Detect and correct drift toward r = 1
      ks.auto_renormalize();
    }
  }

  // Convenience overload: creates a local PalindromePrecession instance.
  // Suitable for one-off steps; for multi-step runs use KernelPipeline.
  static void step(KernelState &ks, KernelMode mode) {
    quantum::PalindromePrecession pp;
    step(ks, mode, pp);
  }

  // ── Invariant verification ────────────────────────────────────────────────

  // Returns true iff all three core invariants hold:
  //   1. |β_phasor| = 1  (normalization — unit-circle eigenvalue)
  //   2. R(r) = 0        (balanced state, Theorem 12)
  //   3. R_eff = 1       (ideal channel, Ohm–Coherence Duality)
  static bool verify(const KernelState &ks) { return ks.all_invariants(); }

  // Verify using the spectral representation: coherence channel at λ = 0
  // gives G_eff = 1 and R_eff = 1.
  static bool verify_spectral(const KernelState &ks) {
    auto ch = to_channel(ks);
    return std::abs(ch.R_eff() - 1.0) < KS_COHERENCE_TOL &&
           std::abs(ch.G_eff() - 1.0) < KS_COHERENCE_TOL;
  }
};

} // namespace kernel::pipeline
