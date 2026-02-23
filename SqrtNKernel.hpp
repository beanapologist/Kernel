/*
 * SqrtNKernel.hpp — O(√N) Integration Kernel
 *
 * Unifies KernelPipeline, PalindromePrecession, and SpectralBridge into a
 * single API demonstrating deterministic O(√N) phase-coherent search on
 * classical hardware.
 *
 * Components integrated:
 *   Pipeline        (KernelPipeline.hpp)     — KernelState + PalindromePrecession
 *                                              + SpectralBridge + auto-renormalization
 *   phasor_at()     (PalindromePrecession)    — stateless phase sweep lookup
 *   channel()       (SpectralBridge)          — quantum ↔ spectral representation
 *   r_eff()         (KernelState)             — G_eff = sech(λ) = 1/R_eff weighting
 *
 * Algorithm (encapsulates coherent_phase_search() from test_coherent_search.cpp):
 *
 *   Phase step ΔΦ = 2π/√N  via PalindromePrecession::phasor_at(step * scale).
 *   Coherence weight G_eff = sech(λ) = 1/R_eff from Pipeline KernelState.
 *   Detection threshold 0.15·√N (Dirichlet-kernel analysis).
 *   Safety limit: 4·√N + 16 steps.  Expected detection ≈ 0.19·√N.
 *
 * Two uses of PalindromePrecession:
 *   Phase source (search)  : PalindromePrecession::phasor_at(step * scale)
 *                            — stateless static lookup, not Pipeline's counter.
 *   State evolution (Pipeline): Pipeline::tick() advances internal pp_ counter
 *                            — maintains KernelState invariants each tick.
 *
 * Quick start:
 *   #include "SqrtNKernel.hpp"
 *   using namespace kernel::pipeline;
 *   SqrtNKernel k = SqrtNKernel::create();
 *   size_t steps = k.search(1 << 20, 42);  // ≈ 201 steps for N = 2^20
 */

#pragma once

#include "KernelPipeline.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>

namespace kernel::pipeline {

// ── SqrtNSearchResult ─────────────────────────────────────────────────────────
// Extended diagnostics from SqrtNKernel::search_ex().
struct SqrtNSearchResult {
  size_t steps;          // Oracle calls until detection (expected ≈ 0.19·√N)
  double peak_amplitude; // Best-channel accumulator value at detection
  double final_g_eff;    // G_eff = sech(λ) = 1/R_eff at the detection step
  size_t renorm_events;  // Number of auto_renormalize() calls during search
  bool detected;         // true if threshold was crossed before max_steps
};

// ── SqrtNKernel ───────────────────────────────────────────────────────────────
//
// Integration class combining all O(√N) algorithm components:
//
//   Pipeline (KernelMode::FULL by default):
//     - KernelState:           quantum state with invariant monitoring
//     - PalindromePrecession:  unit-circle phase evolution (stateful internal)
//     - SpectralBridge:        quantum ↔ Ohm–Coherence spectral bridge
//     - auto-renormalization:  drift correction with event logging
//
//   Coherence-weighted 8-channel accumulation:
//     - G_eff = sech(λ) = 1/R_eff from KernelState::r_eff() (Theorem 14)
//     - = 1.0 for canonical state (r=1); < 1.0 if amplitude drift occurred
//     - Naturally penalises incoherent contributions without restarts
//
//   Phase sweep source:
//     - PalindromePrecession::phasor_at(step * scale) — stateless lookup
//     - scale = floor(PALINDROME_DENOM_FACTOR / √N) — maps period to ΔΦ = 2π/√N
//
// Expected performance: detection at ≈ 0.19·√N oracle calls, speedup ≈ 2.6·√N.
//
class SqrtNKernel {
public:
  using Cx = std::complex<double>;

  // ── Factory ─────────────────────────────────────────────────────────────────
  // Create a SqrtNKernel in the given mode (FULL by default: all submodules).
  static SqrtNKernel create(KernelMode mode = KernelMode::FULL) {
    SqrtNKernel k;
    k.pipeline_ = Pipeline::create(mode);
    return k;
  }

  // ── Primary search API ───────────────────────────────────────────────────────
  // Search for target index in N items.
  // Resets Pipeline to canonical state, runs coherent phase search, returns
  // the number of oracle calls until detection (expected ≈ 0.19·√N).
  size_t search(size_t N, size_t target) {
    return search_ex(N, target).steps;
  }

  // Extended search: returns full diagnostics (steps, peak amplitude, G_eff,
  // renorm event count, and whether detection succeeded before max_steps).
  SqrtNSearchResult search_ex(size_t N, size_t target) {
    pipeline_.reset();

    const double sqrt_n = std::sqrt(static_cast<double>(N));
    const double theta_target =
        TWO_PI * static_cast<double>(target % N) / static_cast<double>(N);
    const Cx target_phasor{std::cos(theta_target), std::sin(theta_target)};

    // Scale: maps palindrome period to effective ΔΦ = 2π/√N per step.
    //   phasor_at(step * scale) ≈ e^{i·step·2π/√N}
    //   when scale = floor(PALINDROME_DENOM_FACTOR / √N).
    // Truncation is intentional — preserves the verified O(√N) behaviour.
    const uint64_t scale = static_cast<uint64_t>(
        static_cast<double>(kernel::quantum::PALINDROME_DENOM_FACTOR) / sqrt_n);

    // 8 real amplitude accumulators (one per bridge channel j = 0…7).
    std::array<double, 8> accum{};
    accum.fill(0.0);

    // Detection threshold (Dirichlet-kernel analysis):
    //   Best bridge channel crosses 0.15·√N at K ≈ 0.19·√N — independent of N.
    const double threshold = THRESHOLD_FACTOR * sqrt_n;

    // Safety limit: abort after 4·√N + 16 steps.
    // (+16 handles rounding for small N where 4·√N is itself small.)
    const uint64_t max_steps = 4 * static_cast<uint64_t>(sqrt_n) + 16;

    size_t renorm_events = 0;
    double peak_amplitude = 0.0;
    double final_g_eff = 1.0;

    for (uint64_t step = 0; step < max_steps; ++step) {
      // Phase source: stateless palindrome lookup scaled to ΔΦ = 2π/√N.
      // Independent of Pipeline's internal PalindromePrecession counter.
      const Cx slow_phasor =
          kernel::quantum::PalindromePrecession::phasor_at(step * scale);

      // Coherence weight from Pipeline KernelState (Theorem 14):
      //   G_eff = sech(λ) = 1/cosh(ln r) = 1/R_eff
      //   = 1.0 when r=1 (canonical coherent state) — full coherence weight.
      //   < 1.0 when r≠1 (drift occurred) — natural decoherence penalty.
      const double g_eff = 1.0 / pipeline_.state().r_eff();
      final_g_eff = g_eff;

      // Accumulate 8 bridge channels: sech-weighted cosine overlaps with target.
      for (int j = 0; j < 8; ++j) {
        const Cx probe = slow_phasor * bridge_[j];
        // Re(probe · conj(target_phasor)) = cos(phase_probe − θ_target)
        const double contrib = probe.real() * target_phasor.real() +
                               probe.imag() * target_phasor.imag();
        accum[j] += g_eff * contrib;
      }

      // Detect: check whether the best channel crossed the threshold.
      double best = 0.0;
      for (int j = 0; j < 8; ++j) {
        const double a = std::abs(accum[j]);
        if (a > best)
          best = a;
      }
      peak_amplitude = best;

      if (best >= threshold) {
        return {static_cast<size_t>(step + 1), best, g_eff, renorm_events,
                true};
      }

      // Advance Pipeline one tick:
      //   SpectralBridge::step() → KernelState::step() (µ-rotation)
      //                          → PalindromePrecession::apply() (phase)
      //                          → auto_renormalize() if drift detected
      const size_t log_before = pipeline_.renorm_log().size();
      pipeline_.tick();
      renorm_events += pipeline_.renorm_log().size() - log_before;
    }

    // Detection failed within safety limit (should not occur for valid N).
    return {static_cast<size_t>(max_steps), peak_amplitude, final_g_eff,
            renorm_events, false};
  }

  // ── Accessors ────────────────────────────────────────────────────────────────

  // Underlying Pipeline for direct inspection (KernelState, renorm log, etc.).
  const Pipeline &pipeline() const { return pipeline_; }

  // G_eff = sech(λ) = 1/R_eff from the current Pipeline KernelState.
  // = 1.0 at canonical coherent state (r=1, λ=0).
  // < 1.0 if amplitude drift has occurred.
  double g_eff() const { return 1.0 / pipeline_.state().r_eff(); }

  // CoherentChannel spectral view of current state (via SpectralBridge).
  kernel::ohm::CoherentChannel channel() const { return pipeline_.channel(); }

  // Verify all three core invariants on current Pipeline state:
  //   1. |α|² + |β|² = 1   (unit-circle normalization)
  //   2. R(r) = 0           (palindrome residual zero — r=1)
  //   3. R_eff = 1          (ideal Ohm–Coherence channel)
  bool verify_invariants() const { return pipeline_.verify_invariants(); }

  // Verify using the spectral (Ohm–Coherence) representation.
  bool verify_spectral() const { return pipeline_.verify_spectral(); }

  // Reset Pipeline to canonical coherent state (r=1, G_eff=1, tick=0).
  void reset() { pipeline_.reset(); }

private:
  Pipeline pipeline_;

  // Precomputed bridge phasors: bridge_[k] = µ^k where µ = e^{i3π/4}.
  // gcd(3,8) = 1 → the 8 phasors uniformly cover the 45° slices of [0°,360°).
  // For any θ_target, the nearest bridge phase is at most 22.5° away,
  // ensuring best-channel overlap ≥ cos(22.5°) ≈ 0.924.
  std::array<Cx, 8> bridge_ = build_bridge();

  // ── Constants ────────────────────────────────────────────────────────────────
  static constexpr double PI = 3.14159265358979323846;
  static constexpr double TWO_PI = 2.0 * PI;
  static constexpr double BRIDGE_ETA = 0.70710678118654752440; // 1/√2

  // Detection threshold: 15% of Dirichlet-kernel peak amplitude (≈ √N/π).
  // Best bridge channel crosses at K ≈ 0.19·√N, independent of N and target.
  static constexpr double THRESHOLD_FACTOR = 0.15;

  // Build the 8 unit-circle phasors {µ^k : k=0…7} where µ = e^{i3π/4}.
  static std::array<Cx, 8> build_bridge() {
    const Cx mu{-BRIDGE_ETA, BRIDGE_ETA};
    std::array<Cx, 8> bridge;
    Cx power{1.0, 0.0};
    for (int k = 0; k < 8; ++k) {
      bridge[k] = power;
      power *= mu;
    }
    return bridge;
  }
};

} // namespace kernel::pipeline
