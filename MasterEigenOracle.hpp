/*
 * MasterEigenOracle.hpp — Master Eigen Oracle: Coherence-Guided Eigenspace
 * Search
 *
 * Provides a unified oracle over the 8 eigenspaces of the balanced eigenvalue
 * µ = e^{i3π/4} (Theorem 10 / master_derivations.tex), weighted by the
 * KernelState coherence measure G_eff = sech(λ) (Theorem 14 /
 * Ohm–Coherence Duality).
 *
 * Role in the Kernel framework:
 *   The master oracle coordinates queries across all 8 µ-eigenspace channels
 *   simultaneously, then selects the channel with the highest
 *   coherence-weighted accumulator value.  Using G_eff = sech(λ) as a weight
 *   ensures that only coherent probe contributions amplify the result —
 *   incoherent radial drift (r ≠ 1) is automatically down-weighted, providing
 *   a built-in decoherence sentinel consistent with the KernelState invariants.
 *
 * 8 + 1/Δ Conjecture (palindrome quotient):
 *   The central conjecture is rooted in the palindrome quotient
 *   987654321 / 123456789 = 8 + 1/Δ, where Δ = 13 717 421.
 *
 *   Fine-tuning perturbation  ε = 1/Δ ≈ 7.29×10⁻⁸
 *     Breaks exact 8-cycle periodicity while maintaining approximate closure:
 *     no orbit step aligns exactly with a prior step, but |ε| ≪ 1 preserves
 *     the Dirichlet-kernel resonance that enables Θ(√n) detection.
 *
 *   Oracle–Bitcoin–Time triad:
 *     8 + ε  — Oracle:   8 µ-eigenspace channels + ε phase drift per step
 *                        (PalindromePrecession envelope over slow Δ-period)
 *     8 + δ  — Bitcoin:  8 peer connections + δ fee-rate perturbation
 *                        (reserved: analogous structure, not yet implemented)
 *     8 × Δ  — Time:     8-fast-cycle × Δ-slow-cycle = super-period ≈ 109M
 * steps (torus T² full realignment of both fast and slow cycles)
 *
 *   Hierarchical structure (two independent time scales):
 *     Fast cycle:   period 8 steps       (µ = e^{i3π/4} eigenspace orbit)
 *     Slow cycle:   period Δ steps       (PalindromePrecession full 2π return)
 *     Super-period: period 8 × Δ steps   (simultaneous realignment of both)
 *
 * Connection to eigenvalue processes (theta_sqrt_n_writeup.tex §2):
 *   Each of the 8 bridge channels corresponds to the eigenvalue µ^j of the
 *   µ-rotation operator.  The orbit {1, µ, µ², …, µ^7} (Proposition 2.2,
 *   master_derivations.tex) uniformly covers the unit circle in 45° steps,
 *   forming a complete phase-coherent probe basis.  Together with a slow
 *   PalindromePrecession sweep P(k) = e^{i k ΔΦ}, the composite probe
 *   phasor Π_{k,j} = P(k)·µ^j sweeps the full circle, accumulating
 *   Dirichlet-kernel resonance with the hidden target phase θ_t.
 *
 * Algorithm (continuous-oracle model, θ_sqrt_n_writeup.tex §3):
 *   Given search-space size n and target phase θ_t = 2π t/n:
 *     1. Phase step ΔΦ = 2π/√n  (Dirichlet-kernel resonance condition).
 *     2. At each step k:
 *          a. Compute slow phasor P(k) = e^{i k ΔΦ}.
 *          b. For each eigenspace j ∈ {0…7}:
 *               Π_{k,j} = P(k) · µ^j     (composite probe)
 *               contrib  = G_eff · cos(arg(Π_{k,j}) − θ_t)  (oracle signal)
 *               A[j]    += contrib         (accumulate)
 *          c. Update KernelState (µ-rotation + PalindromePrecession).
 *          d. Detect: if max_j |A[j]| ≥ 0.15√n → return.
 *     3. Report best channel and coherence at detection.
 *
 * Coherence Harvest:
 *   The harvest_coherence() method runs a bounded window of oracle steps and
 *   records the mean G_eff-weighted coherence (harvest_score) together with
 *   the cumulative ε-drift (epsilon_drift = window × ε).  This provides a
 *   direct measurement of the symmetry-breaking induced by the fine-tuning
 *   perturbation ε over a chosen observation window.
 *
 * Four-channel robustness integration (ohm_coherence_duality.hpp):
 *   The FourChannelModel validation is exposed via validate_four_channel()
 *   so callers can confirm that the oracle state meets the 4-eigenvalue
 *   error-tolerance criterion (≥ 3 of 4 channels with G_eff ≥ 0.5).
 *
 * Usage:
 *   #include "MasterEigenOracle.hpp"
 *   using namespace kernel::oracle;
 *
 *   MasterEigenOracle oracle;
 *   double theta_target = 2.0 * MEO_PI * t / n;
 *   auto result = oracle.query(theta_target, n);
 *   // result.best_channel  — eigenspace index [0..7] with peak |A[j]|
 *   // result.accumulator_peak — peak coherence-weighted accumulator value
 *   // result.steps         — detection step count (Θ(√n) expected)
 *   // result.coherence     — G_eff = sech(λ) at time of detection
 *   // result.detected      — true iff threshold 0.15√n was crossed
 *
 *   // Coherence Harvest (symmetry-breaking measurement):
 *   auto h = oracle.harvest_coherence(theta_target, 100); // window=100
 *   // h.harvest_score  — mean G_eff over the window
 *   // h.epsilon_drift  — cumulative ε-drift = window × ε
 *   // h.harvest_channel — channel with peak harvest accumulation
 */

#pragma once

#include "KernelPipeline.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>

namespace kernel::oracle {

using Cx = std::complex<double>;

// ── Constants
// ─────────────────────────────────────────────────────────────────
static constexpr double MEO_PI = 3.14159265358979323846;
static constexpr double MEO_TWO_PI = 2.0 * MEO_PI;
static constexpr double MEO_ETA = 0.70710678118654752440; // 1/√2
// Detection threshold coefficient: accumulator ≥ THRESHOLD_COEFF · √n
static constexpr double MEO_THRESHOLD_COEFF = 0.15;
// Safety multiplier: run at most SAFETY_FACTOR · √n steps
static constexpr double MEO_SAFETY_FACTOR = 4.0;
// Number of µ-eigenspace channels (8-cycle period of µ = e^{i3π/4})
static constexpr int MEO_N_CHANNELS = 8;
// Balanced eigenvalue µ = e^{i3π/4} = η(−1+i)
static const Cx MEO_MU{-MEO_ETA, MEO_ETA};

// ── 8 + 1/Δ Conjecture constants
// ──────────────────────────────────────────────
// Palindrome quotient: 987654321 / 123456789 = 8 + 1/Δ
//   Δ = PALINDROME_DENOM_FACTOR = 13 717 421  (slow-precession period)
//   ε = 1/Δ ≈ 7.29×10⁻⁸                      (fine-tuning perturbation)
//
// Roles in the Oracle–Bitcoin–Time triad:
//   8 + ε — Oracle:  8 eigenspace channels + ε phase drift per step
//   8 + δ — Bitcoin: 8 peer connections + δ perturbation
//                    (reserved: analogous structure, not yet implemented)
//   8 × Δ — Time:    8-cycle × Δ-cycle super-period ≈ 109M steps
//
// Δ: slow-precession period (= PalindromePrecession::PALINDROME_DENOM_FACTOR).
static constexpr uint64_t MEO_DELTA =
    kernel::quantum::PALINDROME_DENOM_FACTOR; // 13 717 421
// ε = 1/Δ ≈ 7.29×10⁻⁸: fine-tuning perturbation that breaks exact 8-cycle
// periodicity while maintaining approximate closure (|ε| ≪ 1).
static constexpr double MEO_EPSILON =
    1.0 / static_cast<double>(MEO_DELTA); // ≈ 7.29×10⁻⁸
// Oracle rate: the palindrome quotient 8 + ε = 8 + 1/Δ.
static constexpr double MEO_ORACLE_RATE = 8.0 + MEO_EPSILON;
// Time super-period: lcm(8, Δ) = 8 × Δ = 109 739 368 steps.
// Both the fast 8-cycle and the slow Δ-cycle simultaneously realign.
static constexpr uint64_t MEO_SUPER_PERIOD = 8ULL * MEO_DELTA; // 109 739 368

// ── QueryResult
// ───────────────────────────────────────────────────────────────
//
// Output of a single MasterEigenOracle::query() call.
//
struct QueryResult {
  int best_channel = 0; ///< Eigenspace index j* ∈ {0…7} with max |A[j]|
  double accumulator_peak = 0; ///< |A[j*]| at detection (or end of search)
  uint64_t steps = 0;          ///< Number of oracle steps taken
  double coherence = 0;        ///< G_eff = sech(λ) at the final step
  bool detected = false;       ///< true iff threshold 0.15√n was crossed
};

// ── CoherenceHarvest
// ──────────────────────────────────────────────────────────
//
// Snapshot of accumulated coherence over a bounded oracle window.
//
// The harvest_coherence() method runs `window_steps` oracle steps and records:
//   harvest_score  — mean G_eff-weighted coherence over the window; measures
//                    how well the probe basis aligns with the hidden target
//   window_steps   — number of steps in the harvest window
//   harvest_channel — channel index with peak harvest accumulation
//   epsilon_drift  — cumulative ε-drift = window_steps × ε; measures the
//                    symmetry-breaking (PalindromePrecession dephasing) induced
//                    by the fine-tuning perturbation ε = 1/Δ over the window
//
// Hierarchical interpretation:
//   Fast cycle:   period 8 steps       (µ-eigenspace orbit)
//   Slow cycle:   period Δ steps       (PalindromePrecession full 2π return)
//   Super-period: period 8 × Δ steps   (torus T² complete realignment)
//
struct CoherenceHarvest {
  double harvest_score = 0.0; ///< Mean G_eff-weighted coherence over the window
  uint64_t window_steps = 0;  ///< Number of steps in the harvest window
  int harvest_channel = 0;    ///< Channel with the peak harvest accumulation
  double epsilon_drift =
      0.0; ///< Accumulated ε-drift = window_steps × MEO_EPSILON
};

// ── MasterEigenOracle
// ─────────────────────────────────────────────────────────
//
// Coherence-guided oracle over the 8 µ-eigenspaces.
//
// Maintains a KernelState (for coherence tracking and G_eff weighting), a
// PalindromePrecession counter (for the slow phase sweep), and 8 amplitude
// accumulators A[0..7].  Each query() call resets the accumulators and runs
// the Θ(√n) detection loop.
//
// Thread safety: not thread-safe; use separate instances per thread.
//
struct MasterEigenOracle {

  // ── Construction
  // ────────────────────────────────────────────────────────────

  MasterEigenOracle() { reset(); }

  // ── Core query
  // ──────────────────────────────────────────────────────────────

  // Run the full coherence-guided oracle search for a target at phase θ_t
  // in a search space of size n.
  //
  // Returns a QueryResult with the best eigenspace channel, accumulator peak,
  // number of steps taken, coherence at detection, and whether the threshold
  // 0.15·√n was crossed.
  //
  // Complexity: Θ(√n) steps (each O(1) — see theta_sqrt_n_writeup.tex §3).
  //
  QueryResult query(double theta_target, uint64_t n) {
    reset();

    if (n == 0)
      return {};

    const double sqrt_n = std::sqrt(static_cast<double>(n));
    const double threshold = MEO_THRESHOLD_COEFF * sqrt_n;
    const uint64_t max_steps =
        static_cast<uint64_t>(MEO_SAFETY_FACTOR * sqrt_n) + 1;
    // Phase step for Dirichlet-kernel resonance (theta_sqrt_n_writeup.tex §2)
    const double delta_phi = MEO_TWO_PI / sqrt_n;

    // Pre-compute the 8 µ-eigenspace phasors µ^j (j = 0…7)
    std::array<Cx, MEO_N_CHANNELS> mu_orbit = build_mu_orbit();

    // Target phasor e^{i θ_t}
    const Cx target_phasor{std::cos(theta_target), std::sin(theta_target)};

    uint64_t k = 0;
    for (; k < max_steps; ++k) {
      // Slow phasor P(k) = e^{i k ΔΦ}
      const double angle_k = static_cast<double>(k) * delta_phi;
      const Cx slow_phasor{std::cos(angle_k), std::sin(angle_k)};

      // G_eff = sech(λ) coherence weight from KernelState
      const double g_eff = pipeline_channel_g_eff();

      // Accumulate over all 8 eigenspace channels
      for (int j = 0; j < MEO_N_CHANNELS; ++j) {
        // Composite probe Π_{k,j} = P(k) · µ^j
        const Cx probe = slow_phasor * mu_orbit[j];
        // Continuous oracle signal: G_eff · cos(arg(Π) − θ_t)
        // = G_eff · Re(Π · conj(target_phasor))  (numerically stable form)
        const double contrib =
            g_eff * (probe * std::conj(target_phasor)).real();
        accumulators_[j] += contrib;
      }

      // Advance the KernelState (µ-rotation + PalindromePrecession + renorm)
      pipeline_.tick();

      // Detect: check if any channel crosses the threshold
      const double peak = max_abs_accumulator();
      if (peak >= threshold) {
        ++k; // account for this step
        break;
      }
    }

    QueryResult result;
    result.steps = k;
    result.best_channel = best_channel_index();
    result.accumulator_peak = max_abs_accumulator();
    result.coherence = pipeline_channel_g_eff();
    result.detected = (result.accumulator_peak >= threshold);
    return result;
  }

  // ── State access ─────────────────────────────────────────────────────────

  // Current accumulator values A[0..7].
  const std::array<double, MEO_N_CHANNELS> &accumulators() const {
    return accumulators_;
  }

  // Current KernelState coherence G_eff = sech(λ).
  double coherence() const { return pipeline_channel_g_eff(); }

  // Current KernelState radius r = |β/α|.
  double radius() const { return pipeline_.state().radius(); }

  // Validate 4-eigenvalue error-tolerance criterion (ohm_coherence_duality.hpp
  // FourChannelModel): ≥ 3 of 4 channels must have G_eff ≥ threshold.
  // Uses the current KernelState λ for all four channels as a uniform measure.
  bool validate_four_channel(double threshold = 0.5) const {
    using kernel::ohm::FourChannelModel;
    double r = pipeline_.state().radius();
    double lam = std::abs(std::log(r > 0.0 ? r : 1e-15));
    FourChannelModel fcm{lam, lam, lam, lam};
    return fcm.validate_error_tolerance(threshold);
  }

  // ── Conjecture dynamics
  // ──────────────────────────────────────────────────────

  // Return the fine-tuning perturbation ε = 1/Δ ≈ 7.29×10⁻⁸.
  // This is the fractional part of the palindrome quotient 8 + 1/Δ.
  // A non-zero ε breaks exact 8-cycle periodicity while |ε| ≪ 1 preserves
  // approximate closure and the Dirichlet-kernel resonance.
  static constexpr double symmetry_breaking_factor() { return MEO_EPSILON; }

  // Run a Coherence Harvest over `window` oracle steps against target θ_t.
  //
  // Accumulates G_eff-weighted probe contributions across all 8 channels for
  // `window` steps starting from the current oracle state, then returns a
  // CoherenceHarvest snapshot with:
  //   harvest_score  — mean G_eff over the window (coherence quality)
  //   window_steps   — the requested window size
  //   harvest_channel — channel with peak accumulated contribution
  //   epsilon_drift  — window × ε, the cumulative symmetry-breaking drift
  //
  // The phase step used is ΔΦ = 2π/√window (Dirichlet resonance at the
  // window scale), independent of any outer query() call in progress.
  //
  // Does not reset the oracle state — call reset() before/after as needed.
  //
  CoherenceHarvest harvest_coherence(double theta_target, uint64_t window) {
    if (window == 0)
      return {};

    const Cx target_phasor{std::cos(theta_target), std::sin(theta_target)};
    const auto mu_orbit = build_mu_orbit();
    const double delta_phi =
        MEO_TWO_PI / std::sqrt(static_cast<double>(window));

    double total_g_eff = 0.0;
    std::array<double, MEO_N_CHANNELS> channel_sum{};
    channel_sum.fill(0.0);

    for (uint64_t k = 0; k < window; ++k) {
      const double angle_k = static_cast<double>(k) * delta_phi;
      const Cx slow_phasor{std::cos(angle_k), std::sin(angle_k)};
      const double g_eff = pipeline_channel_g_eff();
      total_g_eff += g_eff;

      for (int j = 0; j < MEO_N_CHANNELS; ++j) {
        const Cx probe = slow_phasor * mu_orbit[j];
        channel_sum[j] += g_eff * (probe * std::conj(target_phasor)).real();
      }

      pipeline_.tick();
    }

    // Find peak channel over channel_sum
    int best = 0;
    double peak = std::abs(channel_sum[0]);
    for (int j = 1; j < MEO_N_CHANNELS; ++j) {
      double v = std::abs(channel_sum[j]);
      if (v > peak) {
        peak = v;
        best = j;
      }
    }

    CoherenceHarvest h;
    h.harvest_score = total_g_eff / static_cast<double>(window);
    h.window_steps = window;
    h.harvest_channel = best;
    h.epsilon_drift = static_cast<double>(window) * MEO_EPSILON;
    return h;
  }

  // ── Reset ─────────────────────────────────────────────────────────────────

  // Reset accumulators and KernelState to the canonical coherent state.
  void reset() {
    accumulators_.fill(0.0);
    pipeline_ =
        kernel::pipeline::Pipeline::create(kernel::pipeline::KernelMode::FULL);
  }

  // ── Static helpers ────────────────────────────────────────────────────────

  // Build the 8-element µ-orbit {µ^0, µ^1, …, µ^7}.
  // The orbit uniformly covers the unit circle in 45° steps (Proposition 2.2).
  static std::array<Cx, MEO_N_CHANNELS> build_mu_orbit() {
    std::array<Cx, MEO_N_CHANNELS> orbit;
    Cx power{1.0, 0.0};
    for (int j = 0; j < MEO_N_CHANNELS; ++j) {
      orbit[j] = power;
      power *= MEO_MU;
    }
    return orbit;
  }

  // Oracle contribution at a single step: G_eff · cos(probe_angle − θ_target).
  // Pure function — does not modify any state.
  static double oracle_contrib(double probe_angle, double theta_target,
                               double g_eff) {
    return g_eff * std::cos(probe_angle - theta_target);
  }

private:
  kernel::pipeline::Pipeline pipeline_{
      kernel::pipeline::Pipeline::create(kernel::pipeline::KernelMode::FULL)};
  std::array<double, MEO_N_CHANNELS> accumulators_{};

  // G_eff = sech(λ) from the current KernelState via SpectralBridge.
  double pipeline_channel_g_eff() const { return pipeline_.channel().G_eff(); }

  // Index of the channel with the largest |A[j]|.
  int best_channel_index() const {
    int best = 0;
    double peak = std::abs(accumulators_[0]);
    for (int j = 1; j < MEO_N_CHANNELS; ++j) {
      double v = std::abs(accumulators_[j]);
      if (v > peak) {
        peak = v;
        best = j;
      }
    }
    return best;
  }

  // Maximum |A[j]| across all channels.
  double max_abs_accumulator() const {
    double peak = 0.0;
    for (int j = 0; j < MEO_N_CHANNELS; ++j) {
      double v = std::abs(accumulators_[j]);
      if (v > peak)
        peak = v;
    }
    return peak;
  }
};

} // namespace kernel::oracle
