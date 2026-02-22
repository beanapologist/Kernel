/*
 * ohm_coherence_duality.hpp — Ohm–Coherence Duality Framework
 *
 * Implements the duality:  C = sech(λ) = G_eff = 1 / R_eff
 * where λ is the Lyapunov exponent (Theorem 14 in Pipeline of Coherence).
 *
 * Key models:
 *   CoherentChannel    : single channel characterised by its Lyapunov exponent
 *   MultiChannelSystem : N parallel channels, G_tot = Σ G_i
 *   PipelineSystem     : series stage composition, R_tot = Σ R_stage
 *   FourChannelModel   : 4-eigenvalue structure for error tolerance validation
 *   OUProcess          : Ornstein–Uhlenbeck noise on λ, coherence degradation
 *   QuTritDegradation  : 3-level qutrit coherence degradation patterns
 *
 * Mathematical foundation (Theorem 14):
 *   G_eff(λ) = sech(λ) = 1 / cosh(λ)     (effective conductance)
 *   R_eff(λ) = cosh(λ) = 1 / sech(λ)     (effective resistance)
 *   C = G_eff   (coherence = conductance by the duality)
 *
 * Jensen's inequality under noise:
 *   ⟨G⟩ = ⟨sech(λ)⟩ ≤ sech(⟨λ⟩) when sech is locally concave (|λ| small)
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace kernel::ohm {

// ── Mathematical constant π
// ───────────────────────────────────────────────────
static constexpr double OHM_PI = 3.14159265358979323846;

// ── Numerical tolerance
// ───────────────────────────────────────────────────────
static constexpr double OHM_TOL = 1e-12;

// ── Core Ohm–Coherence Duality functions (Theorem 14) ────────────────────────
// C = sech(λ) = G_eff = 1 / R_eff,   λ = Lyapunov exponent ≥ 0

// Effective conductance: G_eff(λ) = sech(λ) = 1 / cosh(λ)
inline double conductance(double lambda) { return 1.0 / std::cosh(lambda); }

// Effective resistance: R_eff(λ) = cosh(λ) = 1 / G_eff(λ)
inline double resistance(double lambda) { return std::cosh(lambda); }

// Lyapunov exponent from coherence C ∈ (0, 1]: λ = arccosh(1/C)
inline double lyapunov_from_coherence(double C) {
  if (C <= 0.0 || C > 1.0 + OHM_TOL)
    throw std::domain_error("coherence must be in (0, 1]");
  // Clamp for floating-point safety before acosh
  double inv_C = 1.0 / C;
  if (inv_C < 1.0)
    inv_C = 1.0;
  return std::acosh(inv_C);
}

// ── CoherentChannel
// ─────────────────────────────────────────────────────────── Single coherent
// channel characterised by its Lyapunov exponent λ. λ = 0   → ideal, C = G_eff
// = 1, R_eff = 1 λ > 0   → degraded, C < 1
struct CoherentChannel {
  double lambda; // Lyapunov exponent (λ ≥ 0)

  explicit CoherentChannel(double lam) : lambda(lam) {}

  // Effective conductance G_eff = sech(λ)
  double G_eff() const { return conductance(lambda); }

  // Effective resistance R_eff = cosh(λ)
  double R_eff() const { return resistance(lambda); }

  // Coherence C = sech(λ)  (equal to G_eff by the duality)
  double coherence() const { return conductance(lambda); }
};

// ── MultiChannelSystem (N parallel channels)
// ────────────────────────────────── Parallel conductances add: G_tot = Σ G_i =
// Σ sech(λ_i) Total resistance:          R_tot = 1 / G_tot
struct MultiChannelSystem {
  std::vector<CoherentChannel> channels;

  // Heterogeneous channels: each with its own λ
  explicit MultiChannelSystem(const std::vector<double> &lambdas) {
    for (double lam : lambdas)
      channels.emplace_back(lam);
  }

  // Homogeneous N-channel model: all channels share the same λ
  MultiChannelSystem(int N, double lambda) {
    for (int i = 0; i < N; ++i)
      channels.emplace_back(lambda);
  }

  // G_tot = Σ G_i
  double G_total() const {
    double g = 0.0;
    for (const auto &ch : channels)
      g += ch.G_eff();
    return g;
  }

  // R_tot = 1 / G_tot
  double R_total() const { return 1.0 / G_total(); }

  // Index of the weakest (lowest G_eff) channel
  int weakest_channel() const {
    int idx = 0;
    double min_g = channels[0].G_eff();
    for (int i = 1; i < static_cast<int>(channels.size()); ++i) {
      double g = channels[i].G_eff();
      if (g < min_g) {
        min_g = g;
        idx = i;
      }
    }
    return idx;
  }
};

// ── PipelineSystem (series stages)
// ──────────────────────────────────────────── Series resistances add: R_tot =
// Σ R_stage = Σ cosh(λ_i) Total conductance:      G_tot = 1 / R_tot
struct PipelineSystem {
  std::vector<CoherentChannel> stages;

  explicit PipelineSystem(const std::vector<double> &lambdas) {
    for (double lam : lambdas)
      stages.emplace_back(lam);
  }

  // R_tot = Σ cosh(λ_i)
  double R_total() const {
    double r = 0.0;
    for (const auto &s : stages)
      r += s.R_eff();
    return r;
  }

  // G_tot = 1 / R_tot
  double G_total() const { return 1.0 / R_total(); }

  // Index of the bottleneck stage (largest R_eff — limiting channel)
  int bottleneck_stage() const {
    int idx = 0;
    double max_r = stages[0].R_eff();
    for (int i = 1; i < static_cast<int>(stages.size()); ++i) {
      double r = stages[i].R_eff();
      if (r > max_r) {
        max_r = r;
        idx = i;
      }
    }
    return idx;
  }
};

// ── FourChannelModel (4-eigenvalue structure)
// ───────────────────────────────── Four-channel redundancy system for quantum
// error tolerance. The 4-eigenvalue structure supports detection of one faulty
// channel and correction when ≥ 3 of 4 channels remain coherent (G_eff ≥
// threshold).
struct FourChannelModel {
  static constexpr int N_CHANNELS = 4;
  double lambdas[N_CHANNELS]; // per-channel Lyapunov exponents

  FourChannelModel(double lam0, double lam1, double lam2, double lam3) {
    lambdas[0] = lam0;
    lambdas[1] = lam1;
    lambdas[2] = lam2;
    lambdas[3] = lam3;
  }

  // Fill out[i] = G_eff(λ_i) for all 4 channels
  void eigenvalues(double out[N_CHANNELS]) const {
    for (int i = 0; i < N_CHANNELS; ++i)
      out[i] = conductance(lambdas[i]);
  }

  // Validate error tolerance: true iff ≥ 3 channels have G_eff ≥ threshold
  bool validate_error_tolerance(double threshold = 0.5) const {
    int coherent = 0;
    for (int i = 0; i < N_CHANNELS; ++i)
      if (conductance(lambdas[i]) >= threshold)
        ++coherent;
    return coherent >= 3;
  }

  // Index of the weakest channel (lowest G_eff)
  int weakest_channel() const {
    int idx = 0;
    double min_g = conductance(lambdas[0]);
    for (int i = 1; i < N_CHANNELS; ++i) {
      double g = conductance(lambdas[i]);
      if (g < min_g) {
        min_g = g;
        idx = i;
      }
    }
    return idx;
  }
};

// ── OUProcess (Ornstein–Uhlenbeck noise on λ)
// ───────────────────────────────── Discrete-time Euler–Maruyama approximation
// of  dλ = −θ(λ − μ)dt + σ dW. Simulates coherence degradation as λ fluctuates
// around a mean value μ. Uses a portable LCG + Box–Muller transform (no
// external dependencies).
struct OUProcess {
  double theta; // Mean-reversion rate (θ > 0)
  double mu;    // Long-run mean of λ
  double sigma; // Noise amplitude (σ ≥ 0)

  OUProcess(double theta_, double mu_, double sigma_)
      : theta(theta_), mu(mu_), sigma(sigma_) {}

  // Simulate `steps` OU steps starting from lambda0 with time-step dt.
  // Returns a path of length (steps + 1) including the initial value.
  std::vector<double> simulate(double lambda0, int steps, double dt,
                               uint64_t seed = 42) const {
    std::vector<double> path;
    path.reserve(steps + 1);
    path.push_back(lambda0);

    uint64_t state = seed;
    double lam = lambda0;
    for (int i = 0; i < steps; ++i) {
      // Box–Muller: two uniform draws → one standard-normal sample
      double u1 = lcg_uniform(state);
      double u2 = lcg_uniform(state);
      if (u1 < 1e-15)
        u1 = 1e-15; // guard against log(0)
      double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * OHM_PI * u2);
      // Euler–Maruyama step
      lam += -theta * (lam - mu) * dt + sigma * std::sqrt(dt) * z;
      path.push_back(lam);
    }
    return path;
  }

  // Compute average conductance ⟨G⟩ = (1/N) Σ sech(λ_i) over a λ-path.
  static double average_conductance(const std::vector<double> &path) {
    double sum = 0.0;
    for (double lam : path)
      sum += conductance(lam);
    return sum / static_cast<double>(path.size());
  }

private:
  // Linear congruential generator returning a uniform sample in [0, 1)
  static double lcg_uniform(uint64_t &s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(s >> 11) / static_cast<double>(1ULL << 53);
  }
};

// ── QuTritDegradation
// ───────────────────────────────────────────────────────── Coherence
// degradation for a 3-level (qutrit) system. Each pair of levels has its own
// Lyapunov exponent (λ_01, λ_02, λ_12). Effective coherence is the mean
// sech-conductance across the three channels.
struct QuTritDegradation {
  double lambda_01; // λ for 0 ↔ 1 transition
  double lambda_02; // λ for 0 ↔ 2 transition
  double lambda_12; // λ for 1 ↔ 2 transition

  QuTritDegradation(double l01, double l02, double l12)
      : lambda_01(l01), lambda_02(l02), lambda_12(l12) {}

  // Average sech-coherence across the three transition channels
  double coherence_avg() const {
    return (conductance(lambda_01) + conductance(lambda_02) +
            conductance(lambda_12)) /
           3.0;
  }

  // Minimum (worst-case) coherence — most degraded transition
  double coherence_min() const {
    double c01 = conductance(lambda_01);
    double c02 = conductance(lambda_02);
    double c12 = conductance(lambda_12);
    return std::min(c01, std::min(c02, c12));
  }
};

} // namespace kernel::ohm
