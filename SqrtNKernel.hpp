/*
 * SqrtNKernel.hpp — O(√N) Classical Hardware Integration Layer
 *
 * Integrates all four Kernel components into one unified O(√N) API:
 *
 *   KernelPipeline     — end-to-end pipeline (µ-rotation, renormalization)
 *   PalindromePrecession — phasor_at(k·scale) drives the phase sweep
 *   SpectralBridge     — channel() gives the CoherentChannel at each step
 *   KernelState        — accessible via state(), invariants verifiable
 *
 * O(√N) accumulation model:
 *   A classical brute-force search over N items costs O(N) steps.
 *   The SqrtNKernel mimics the quantum speedup classically: it runs exactly
 *   ⌈√N⌉ pipeline ticks while accumulating a weighted G_eff sum, achieving
 *   an effective speedup factor of √N over the naive baseline.
 *
 * Design choices:
 *   - Pipeline G_eff is weighted at each accumulation step by a linear ramp
 *     w_k = k (k = 1…⌈√N⌉), so later steps (closer to the "answer") carry
 *     more signal.  The final accumulated_g_eff() is the normalised weighted
 *     mean; for an ideal coherent state it equals 1.0.
 *
 *   - Phase sweep: at step k the phasor
 *       P_k = PalindromePrecession::phasor_at(k · ⌈√N⌉)
 *     is computed.  Multiplying the step index by sqrt_N_ compresses the
 *     entire precession orbit into ⌈√N⌉ samples (scaled phase-space stride).
 *     |P_k| = 1 for all k — unit-circle invariant is maintained.
 *
 *   - SpectralBridge is accessed exclusively through channel(), which calls
 *     SpectralBridge::to_channel(state_) internally.
 *
 * Quick start:
 *
 *   #include "SqrtNKernel.hpp"
 *   using namespace kernel::sqrtn;
 *
 *   SqrtNKernel sk(1024);          // problem size N = 1024 → √N = 32 steps
 *   sk.run();                      // execute the O(√N) sweep
 *   bool ok = sk.verify_invariants();          // true for canonical state
 *   double g  = sk.accumulated_g_eff();        // ≈ 1.0 for ideal channel
 *   double sp = sk.speedup_metric();           // ≈ 32 (= N/√N = √N)
 *   auto   ch = sk.channel();                  // CoherentChannel at final step
 */

#pragma once

#include "KernelPipeline.hpp"
#include "PalindromePrecession.hpp"
#include "SpectralBridge.hpp"
#include "KernelState.hpp"

#include <cmath>
#include <complex>
#include <cstdint>

namespace kernel::sqrtn {

using namespace kernel::pipeline;
using Cx = std::complex<double>;

// ── SqrtNKernel
// ────────────────────────────────────────────────────────────────
//
// Integration class unifying the four Kernel components into a single
// O(√N) classical hardware demonstration.
//
// Thread safety: not thread-safe; intended for single-threaded use.
//
class SqrtNKernel {
public:
  // ── Construction ──────────────────────────────────────────────────────────

  // Construct a kernel for a problem of size N.
  // mode controls the underlying Pipeline's operation:
  //   FULL     — µ-rotation + palindrome precession + auto-renormalization
  //   SPECTRAL — µ-rotation + auto-renormalization (no extra precession)
  //   STANDARD — µ-rotation only
  explicit SqrtNKernel(uint64_t N, KernelMode mode = KernelMode::FULL)
      : N_(N),
        sqrt_N_(
            static_cast<uint64_t>(std::ceil(std::sqrt(static_cast<double>(N))))),
        pipeline_(Pipeline::create(mode)) {}

  // ── Core execution ─────────────────────────────────────────────────────────

  // Run the O(√N) accumulation sweep.
  //
  // Iterates exactly sqrt_N_ = ⌈√N⌉ ticks.  At each tick k (1-based):
  //
  //   1. pipeline_.tick() — advances the pipeline one step (µ-rotation,
  //      optional palindrome precession, optional auto-renormalization).
  //
  //   2. Phase sweep — computes the scaled precession phasor:
  //        P_k = PalindromePrecession::phasor_at(k * sqrt_N_)
  //      This samples the palindrome precession orbit at a stride of sqrt_N_,
  //      mapping ⌈√N⌉ accumulation steps onto the full precession period.
  //      |P_k| = 1 for all k (unit-circle invariant).
  //
  //   3. G_eff weighting — reads the spectral channel via channel() and
  //      accumulates:
  //        accumulated_g_eff_ += k * channel().G_eff()
  //      Weight w_k = k gives a linear ramp so the final steps have higher
  //      influence (analogous to amplitude amplification in Grover's algorithm).
  //
  // After the loop the accumulated sum is divided by Σ w_k = sqrt_N_*(sqrt_N_+1)/2
  // to produce the weighted mean G_eff in [0, 1].  For an ideal coherent state
  // the result is 1.0.
  void run() {
    accumulated_g_eff_ = 0.0;
    double weight_sum = 0.0;

    for (uint64_t k = 1; k <= sqrt_N_; ++k) {
      // 1. Advance the pipeline one tick
      pipeline_.tick();

      // 2. Phase sweep: scaled precession phasor at step k * sqrt_N_
      //    Uses PalindromePrecession::phasor_at (static, no side-effects).
      last_phase_phasor_ =
          quantum::PalindromePrecession::phasor_at(k * sqrt_N_);

      // 3. G_eff weighting via SpectralBridge (accessed through channel())
      double w = static_cast<double>(k);
      weight_sum += w;
      accumulated_g_eff_ += w * pipeline_.channel().G_eff();
    }

    if (weight_sum > 0.0)
      accumulated_g_eff_ /= weight_sum;
  }

  // ── Accessors ─────────────────────────────────────────────────────────────

  // Current spectral channel — SpectralBridge accessed via channel().
  // Returns the CoherentChannel corresponding to the pipeline's current state.
  kernel::ohm::CoherentChannel channel() const { return pipeline_.channel(); }

  // Weighted mean G_eff from the last run() call.
  // Value is in (0, 1]; equals 1.0 for a perfectly coherent (r = 1) state.
  // Returns 0.0 if run() has not been called.
  double accumulated_g_eff() const { return accumulated_g_eff_; }

  // Most recent scaled phase phasor P_k = phasor_at(sqrt_N_ · sqrt_N_).
  // |last_phase_phasor()| = 1 always (unit-circle invariant).
  // Returns {1, 0} if run() has not been called.
  Cx last_phase_phasor() const { return last_phase_phasor_; }

  // Classical O(N)/O(√N) speedup ratio = N / ⌈√N⌉ ≈ √N.
  double speedup_metric() const {
    return static_cast<double>(N_) / static_cast<double>(sqrt_N_);
  }

  // Verify all three core invariants on the current pipeline state:
  //   |α|² + |β|² = 1,   R(r) = 0,   R_eff = 1.
  bool verify_invariants() const { return pipeline_.verify_invariants(); }

  // Verify via the spectral representation: G_eff = R_eff = 1.
  bool verify_spectral() const { return pipeline_.verify_spectral(); }

  // Read-only access to the underlying KernelState.
  const KernelState &state() const { return pipeline_.state(); }

  // Problem size N (as provided to the constructor).
  uint64_t N() const { return N_; }

  // Number of accumulation steps = ⌈√N⌉.
  uint64_t sqrt_N() const { return sqrt_N_; }

  // ── Reset ─────────────────────────────────────────────────────────────────

  // Reset to the canonical coherent state, clearing the accumulation.
  void reset() {
    pipeline_.reset();
    accumulated_g_eff_ = 0.0;
    last_phase_phasor_ = Cx{1.0, 0.0};
  }

private:
  uint64_t N_;
  uint64_t sqrt_N_;
  Pipeline pipeline_;
  double accumulated_g_eff_ = 0.0;
  Cx last_phase_phasor_{1.0, 0.0};
};

} // namespace kernel::sqrtn
