/*
 * PalindromePrecession.hpp — Unit-Circle Invariant Palindrome Precession
 *
 * Implements uniform angular precession derived from the palindrome quotient
 * 987654321 / 123456789 = 8 + 1/13717421, ensuring that every transformation
 * has its eigenvalue signature on the unit circle (|phasor| = 1 at every step).
 *
 * Mathematical foundation:
 *   Palindrome quotient:   987654321 / 123456789 = 8 + 9/123456789
 *                                                = 8 + 1/13717421
 *   (since 9 × 13717421 = 123456789)
 *
 *   Phase increment:  ΔΦ = 2π / 13717421  ≈ 4.578 × 10⁻⁷ rad/step
 *
 *   Precession phasor at step n:  P(n) = e^{i n ΔΦ} = cos(nΔΦ) + i sin(nΔΦ)
 *   Invariant:  |P(n)| = 1  for all n  (unit circle)
 *
 * Double periodicity (torus structure T²):
 *   Fast 8-cycle:      µ = e^{i3π/4}, period 8 steps
 *   Slow precession:   period 13717421 steps (exact 2π return)
 *   Super-period:      lcm(8, 13717421) × alignment = 109,739,368 steps
 *
 * Zero overhead:
 *   Each step is a single complex multiplication by a unit-norm phasor.
 *   No kick branching, no coherence feedback — T = 0 excess overhead.
 *   |β| is preserved at every step: the precession map is an isometry.
 *
 * Theorem 5.1 (Zero Excess Resistance):
 *   |P(n)| = 1 → r = 1, C = 1, R = 0  (simultaneous coherence invariants)
 *
 * Theorem 6.1 (Optimality of k=1):
 *   Among scaled rates ΔΦ(k) = 2π/(13717421 k), k=1 maximises angular
 *   diversity per step at practical run lengths.
 *
 * Usage:
 *   PalindromePrecession pp;
 *   Cx phasor = pp.current_phasor();  // e^{i·0·ΔΦ} at step 0
 *   pp.apply(beta);                   // beta *= phasor, then advance
 *   Cx p = pp.phasor_at(n);          // e^{i·n·ΔΦ} without advancing
 *
 * Integration:
 *   Self-contained header; no external dependencies beyond <cmath> and
 *   <complex>.  Include before use of PalindromePrecession in any translation
 *   unit.  For use with QState (quantum_kernel_v2.cpp), include this header
 *   after QState is defined and use apply(state.beta).
 */

#pragma once

#include <cmath>
#include <complex>
#include <cstdint>

namespace kernel::quantum {

// ── Palindrome quotient constants
// ───────────────────────────────────────────── 987654321 / 123456789 =
// INTEGER_PART + RESIDUE / DENOMINATOR
//                       = 8 + 9 / 123456789
//                       = 8 + 1 / 13717421   (since 9 × 13717421 = 123456789)
static constexpr uint64_t PALINDROME_NUMERATOR = 987654321ULL;
static constexpr uint64_t PALINDROME_DENOMINATOR = 123456789ULL;
static constexpr uint64_t PALINDROME_INTEGER_PART = 8ULL;
static constexpr uint64_t PALINDROME_RESIDUE = 9ULL;
static constexpr uint64_t PALINDROME_DENOM_FACTOR = 13717421ULL;

// Mathematical constants
static constexpr double PRECESSION_TWO_PI = 2.0 * 3.14159265358979323846;

// ΔΦ = 2π / 13717421: angular increment per step derived from the palindrome
// fractional part 1/13717421.
static constexpr double PRECESSION_DELTA_PHASE =
    PRECESSION_TWO_PI / static_cast<double>(PALINDROME_DENOM_FACTOR);

// ── PalindromePrecession
// ────────────────────────────────────────────────────── Stateful
// uniform-precession operator.
//
// Maintains a step counter n and computes the unit-norm phasor P(n) =
// e^{i n ΔΦ} on demand.  Each call to advance() increments n by 1.
// apply() multiplies a complex amplitude by P(n) and then advances n.
//
// Invariant:  |P(n)| = 1 for all n  (unit circle — eigenvalue signature)
//
struct PalindromePrecession {
  uint64_t step_count = 0; // Current step index n

  // Precomputed single-step phasor e^{iΔΦ}; computed once at program start.
  // Making it a static member avoids repeated trigonometric evaluation and
  // any concerns about thread-safe lazy initialization.
  static const std::complex<double> STEP_PHASOR;

  // Returns the precession phasor P(n) = e^{i n ΔΦ} for the current step.
  // |phasor| = 1 — unit-circle invariant guaranteed by construction.
  std::complex<double> current_phasor() const { return phasor_at(step_count); }

  // Returns the precession phasor P(n) = e^{i n ΔΦ} for an arbitrary step n.
  // Pure computation — does not modify step_count.
  static std::complex<double> phasor_at(uint64_t n) {
    const double angle = static_cast<double>(n) * PRECESSION_DELTA_PHASE;
    return {std::cos(angle), std::sin(angle)};
  }

  // Advance the step counter by one.
  void advance() { ++step_count; }

  // Apply one incremental precession step to a complex amplitude, then advance.
  //   beta' = e^{iΔΦ} · beta   (single ΔΦ increment — preserves |beta|)
  //   n     → n + 1
  //
  // After N calls from step 0, the net phase applied equals N · ΔΦ, which
  // matches phasor_at(N):  (e^{iΔΦ})^N = e^{i N ΔΦ} = phasor_at(N).
  void apply(std::complex<double> &beta) {
    beta *= STEP_PHASOR;
    advance();
  }

  // Reset the step counter to zero (restarts the precession cycle).
  void reset() { step_count = 0; }
};

// Out-of-line definition of the static step phasor e^{iΔΦ}.
inline const std::complex<double> PalindromePrecession::STEP_PHASOR = {
    std::cos(PRECESSION_DELTA_PHASE), std::sin(PRECESSION_DELTA_PHASE)};

// ── Scaled precession (generalisation to rate k)
// ────────────────────────────── Returns the phase increment for a scaled
// precession rate:
//   delta_phase(k) = 2π / (PALINDROME_DENOM_FACTOR × k)   rad/step
//
// k=1 → palindrome baseline (maximum angular diversity per step).
// k>1 → slower precession, longer super-period.
inline double precession_delta_phase(unsigned k) {
  return PRECESSION_TWO_PI / (static_cast<double>(PALINDROME_DENOM_FACTOR) *
                              static_cast<double>(k));
}

} // namespace kernel::quantum
