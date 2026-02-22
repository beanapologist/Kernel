/*
 * KernelState.hpp — Unified State Container with Invariant Enforcement
 *
 * Wraps QState with coherent-point invariant monitoring and
 * auto-renormalization:
 *
 *   Invariant 1 — Unit-circle eigenvalue (|β_phasor| = 1):
 *     The precession phasor applied to β is always on the unit circle.
 *     Practically: |β| is preserved by every pure-phase operation, so any
 *     deviation of |β| from its initial value indicates numerical drift.
 *
 *   Invariant 2 — R(r) = 0 (Theorem 12):
 *     The palindrome residual R(r) = (1/δ_S)(r − 1/r) = 0 iff r = 1.
 *     A balanced state (r = 1) is the coherent fixed point.
 *
 *   Invariant 3 — R_eff = 1 (Ohm–Coherence Duality, Theorem 14):
 *     R_eff(λ) = cosh(λ) = 1 when λ = ln r = 0, i.e. r = 1.
 *     Equivalently, G_eff = sech(λ) = 1, C = 1 (maximum coherence).
 *
 * All three invariants are simultaneously satisfied at and only at r = 1
 * (Corollary 13: simultaneous break).
 *
 * Drift detection:
 *   has_drift() returns true when |R(r)| > tolerance.
 *   Drift implies r ≠ 1 and all three invariants are broken.
 *
 * Auto-renormalization:
 *   auto_renormalize() projects β toward the balanced amplitude |β| = |α|,
 *   restoring r → 1.  Every renormalization event is appended to renorm_log
 *   so callers can audit corrections.
 *
 * Usage:
 *   KernelState ks;                          // canonical coherent state
 *   ks.step();                               // apply µ-rotation
 *   if (ks.has_drift()) ks.auto_renormalize(); // correct if needed
 *   bool ok = ks.all_invariants();           // verify all three
 */

#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

// ── Forward-declaration of constants used in quantum_kernel_v2.cpp ───────────
// (Duplicated here as constexpr so KernelState.hpp is self-contained.)
namespace kernel::pipeline {

static constexpr double KS_ETA = 0.70710678118654752440;     // 1/√2
static constexpr double KS_DELTA_S = 2.41421356237309504880; // δ_S = 1+√2
static constexpr double KS_RADIUS_TOL = 1e-9;
static constexpr double KS_COHERENCE_TOL = 1e-9;
static constexpr double KS_CONSERVATION_TOL = 1e-12;

using Cx = std::complex<double>;

// ── Coherence helpers (Theorems 11, 12, 14) ──────────────────────────────────
inline double ks_coherence(double r) { return (2.0 * r) / (1.0 + r * r); }
inline double ks_palindrome_residual(double r) {
  return (1.0 / KS_DELTA_S) * (r - 1.0 / r);
}
inline double ks_lyapunov(double r) { return std::log(r > 0.0 ? r : 1e-15); }
inline double ks_r_eff(double r) { return std::cosh(ks_lyapunov(r)); }

// ── RenormEvent: one auto-renormalization log entry
// ───────────────────────────
struct RenormEvent {
  uint64_t tick;   // Kernel tick at which renormalization was applied
  double r_before; // Radius r = |β/α| before correction
  double r_after;  // Radius r = |β/α| after correction
  double R_before; // Palindrome residual R(r) before correction
  double R_after;  // Palindrome residual R(r) after correction
};

// ── KernelState: quantum state with invariant monitoring
// ──────────────────────
//
// Holds the canonical two-component quantum state (α, β) and exposes
// invariant checks and auto-renormalization with logging.
//
// Mathematical state:
//   α  — ground-state coefficient (|α| = 1/√2 at coherent fixed point)
//   β  — excited-state coefficient (|β| = 1/√2 at coherent fixed point)
//   r  = |β/α|         — radius parameter
//   C  = 2|α||β|       — coherence (Theorem 9)
//   R  = (1/δ_S)(r−1/r) — palindrome residual (Theorem 12)
//   λ  = ln r           — Lyapunov exponent (Theorem 14)
//   R_eff = cosh(λ)     — effective resistance (Ohm–Coherence Duality)
//
struct KernelState {
  Cx alpha{KS_ETA, 0.0};               // |0⟩ coefficient (real positive)
  Cx beta{-0.5, 0.5};                  // |1⟩ coefficient  = e^{i3π/4}/√2
  uint64_t tick = 0;                   // Step counter (incremented by step())
  std::vector<RenormEvent> renorm_log; // History of auto-renormalizations

  // ── Derived quantities ────────────────────────────────────────────────────

  // r = |β/α| (radius parameter, Theorem 11)
  double radius() const {
    return std::abs(alpha) > KS_COHERENCE_TOL ? std::abs(beta) / std::abs(alpha)
                                              : 0.0;
  }

  // C = 2|α||β| (coherence, Theorem 9/11)
  double coherence() const { return 2.0 * std::abs(alpha) * std::abs(beta); }

  // R(r) = (1/δ_S)(r − 1/r) (palindrome residual, Theorem 12)
  double palindrome_residual() const {
    return ks_palindrome_residual(radius());
  }

  // λ = ln r (Lyapunov exponent, Theorem 14)
  double lyapunov() const { return ks_lyapunov(radius()); }

  // R_eff = cosh(λ) (effective resistance, Ohm–Coherence Duality)
  double r_eff() const { return ks_r_eff(radius()); }

  // ── Invariant checks ─────────────────────────────────────────────────────

  // Invariant 1: unit-circle eigenvalue — |β| preserved by phase operations.
  // Here we check that |α|² + |β|² = 1 (normalization) to within tolerance,
  // which ensures no amplitude drift has occurred.
  bool beta_unit_invariant() const {
    double norm_sq = std::norm(alpha) + std::norm(beta);
    return std::abs(norm_sq - 1.0) < KS_CONSERVATION_TOL;
  }

  // Invariant 2: R(r) = 0 (balanced coherent fixed point, Theorem 12)
  bool palindrome_zero() const {
    return std::abs(palindrome_residual()) < KS_RADIUS_TOL;
  }

  // Invariant 3: R_eff = 1 (ideal channel in Ohm–Coherence Duality)
  bool r_eff_unity() const {
    return std::abs(r_eff() - 1.0) < KS_COHERENCE_TOL;
  }

  // All three core invariants simultaneously (Corollary 13)
  bool all_invariants() const {
    return beta_unit_invariant() && palindrome_zero() && r_eff_unity();
  }

  // ── Drift detection ───────────────────────────────────────────────────────

  // Returns true when |R(r)| > tol: state has drifted from r = 1.
  // Non-zero palindrome residual implies all three invariants are broken.
  bool has_drift(double tol = KS_RADIUS_TOL) const {
    return std::abs(palindrome_residual()) > tol;
  }

  // ── Normalization ─────────────────────────────────────────────────────────

  // Restore |α|² + |β|² = 1 (quantum normalization).
  // Called internally by auto_renormalize(); may also be called explicitly.
  void normalize() {
    double norm_sq = std::norm(alpha) + std::norm(beta);
    if (std::abs(norm_sq - 1.0) > KS_CONSERVATION_TOL) {
      double scale = 1.0 / std::sqrt(norm_sq);
      alpha *= scale;
      beta *= scale;
    }
  }

  // ── Auto-renormalization ──────────────────────────────────────────────────
  //
  // If |R(r)| > tol, project β toward the balanced amplitude |β| = |α|,
  // restoring r → 1 (and hence R → 0, R_eff → 1).
  //
  // The correction blends the current β magnitude toward |α| at the given
  // rate (0 = no correction, 1 = instant snap to balanced amplitude).
  // Phase of β is preserved exactly.
  //
  // Each renormalization is recorded in renorm_log.
  //
  // Returns true if renormalization was applied, false if state was already
  // within tolerance.
  //
  bool auto_renormalize(double tol = KS_RADIUS_TOL, double rate = 0.5) {
    if (!has_drift(tol))
      return false;

    double r_before = radius();
    double R_before = palindrome_residual();

    // Target magnitude for β: |α| (balanced state, r = 1)
    double target_mag = std::abs(alpha);
    double current_mag = std::abs(beta);

    if (current_mag > KS_COHERENCE_TOL && target_mag > KS_COHERENCE_TOL) {
      // Blend magnitude toward target, preserve phase
      double new_mag = current_mag + rate * (target_mag - current_mag);
      beta = beta * (new_mag / current_mag);
    }

    normalize();

    double r_after = radius();
    double R_after = palindrome_residual();

    renorm_log.push_back({tick, r_before, r_after, R_before, R_after});
    return true;
  }

  // ── State evolution ───────────────────────────────────────────────────────

  // Apply one µ = e^{i3π/4} rotation to β (Section 2, Theorem 10).
  // Increments the tick counter.
  void step() {
    static const Cx MU{-KS_ETA, KS_ETA}; // e^{i3π/4}
    beta *= MU;
    ++tick;
  }

  // Reset to canonical coherent state (Theorem 8)
  void reset() {
    alpha = Cx{KS_ETA, 0.0};
    beta = Cx{-0.5, 0.5};
    tick = 0;
    renorm_log.clear();
  }
};

} // namespace kernel::pipeline
