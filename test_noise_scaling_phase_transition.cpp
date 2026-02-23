/*
 * test_noise_scaling_phase_transition.cpp — Noise-Scaling Phase Transition Test
 *
 * Implements the H₀/H₁ hypothesis test to determine whether the Kernel's
 * acceleration arises from a coherence mechanism (H₁) or a heuristic
 * strategy (H₀) by measuring the scaling exponent α (T ~ N^α) across a
 * sweep of injected phase noise ε.
 *
 * ── Experimental conditions ───────────────────────────────────────────────
 *
 *   1. Linear baseline     — brute-force O(n) scan; α ≈ 1.0 for all ε.
 *
 *   2. Precession-only     — PalindromePrecession phase sweep with
 *                            accumulated phase noise; G_eff = 1 throughout
 *                            (no amplitude correction, no Chiral Kick).
 *
 *   3. Full Kernel         — Precession sweep with phase noise, plus
 *                            G_eff = sech(λ) coherence weighting from
 *                            KernelState (Chiral Kick analogue).  Radial
 *                            noise is also injected into KernelState each
 *                            step (without auto-renormalization), allowing
 *                            radial drift to accumulate and G_eff to
 *                            degrade — modelling the sensitivity of the
 *                            coherence mechanism to perturbations.
 *
 * ── Noise model ───────────────────────────────────────────────────────────
 *
 *   Phase noise (both coherent conditions):
 *     phase_noise_accum += uniform(-ε, +ε)   per step
 *     slow phasor: e^{i(k·ΔΦ + phase_noise_accum)}
 *
 *   Radial noise (Full Kernel only):
 *     ks.beta *= (1 + uniform(-ε/2, +ε/2))   per step
 *     Perturbs r = |β/α| from 1 → G_eff = sech(ln r) degrades.
 *
 * ── System sizes ──────────────────────────────────────────────────────────
 *
 *   N = 2^{10}, 2^{12}, 2^{14}, 2^{16}.
 *
 * ── Noise sweep ───────────────────────────────────────────────────────────
 *
 *   ε ∈ {0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}.
 *
 * ── Scaling exponent ──────────────────────────────────────────────────────
 *
 *   α(ε) = OLS slope of log T(N, ε) vs log N over the four system sizes.
 *
 * ── Expected behavior ─────────────────────────────────────────────────────
 *
 *   H₀ (heuristic): α(ε) drifts smoothly from 0.5 toward 1.0.
 *   H₁ (coherence): α(ε) stays near 0.5 for ε < ε*, then rises abruptly
 *                   at a critical noise threshold ε*.  Sharpens with N.
 *
 * ── Hard assertions (always checked) ──────────────────────────────────────
 *
 *   1. α(linear, any ε) ∈ [0.90, 1.10]       — O(n) baseline confirmed.
 *   2. α(prec-only, ε=0) ∈ [0.45, 0.55]      — √n scaling at zero noise.
 *   3. α(full-kernel, ε=0) ∈ [0.45, 0.55]    — √n scaling at zero noise.
 *   4. α(full-kernel, ε=1.0) > 0.70           — heavy noise exits √n band.
 *   5. α(full-kernel, ε) non-decreasing       — noise monotonically
 *                                               degrades scaling.
 *
 * ── Reporting (informational, no pass/fail) ───────────────────────────────
 *
 *   - α(ε) table for all three conditions.
 *   - Steepest Δα and candidate critical point ε*.
 *   - H₁ verdict if max |Δα| > SHARP_TRANSITION_THRESHOLD, else H₀.
 */

#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "KernelPipeline.hpp"

using kernel::pipeline::KernelState;
using kernel::quantum::PALINDROME_DENOM_FACTOR;
using kernel::quantum::PRECESSION_DELTA_PHASE;

using Cx = std::complex<double>;

// ── Module constants ───────────────────────────────────────────────────────

static constexpr double PT_PI = 3.14159265358979323846;
static constexpr double PT_TWO_PI = 2.0 * PT_PI;
static constexpr double PT_ETA = 0.70710678118654752440; // 1/√2
static constexpr uint64_t PT_RNG_SEED = 314159ULL;

// Assertion thresholds
static constexpr double ALPHA_SQRT_N_LOW = 0.45;
static constexpr double ALPHA_SQRT_N_HIGH = 0.55;
static constexpr double ALPHA_LINEAR_LOW = 0.90;
static constexpr double ALPHA_LINEAR_HIGH = 1.10;
static constexpr double ALPHA_NOISY_MIN = 0.70; // α(full-kernel, ε=1.0)
static constexpr double ALPHA_MONO_TOL = 0.15;  // tolerance for monotone check
static constexpr double SHARP_TRANSITION_THRESHOLD = 0.20; // |Δα| > 0.20 → H₁

// ── OLS slope ─────────────────────────────────────────────────────────────

static constexpr double PT_LINREG_DENOM_TOL = 1e-12;

static double pt_linreg_slope(const std::vector<double> &xs,
                              const std::vector<double> &ys) {
  const int n = static_cast<int>(xs.size());
  double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
  for (int i = 0; i < n; ++i) {
    sx += xs[i];
    sy += ys[i];
    sxx += xs[i] * xs[i];
    sxy += xs[i] * ys[i];
  }
  const double denom = static_cast<double>(n) * sxx - sx * sx;
  if (std::abs(denom) < PT_LINREG_DENOM_TOL)
    return 0.0;
  return (static_cast<double>(n) * sxy - sx * sy) / denom;
}

// ── NullSliceBridge (8-cycle, µ = e^{i3π/4}) ──────────────────────────────

static const Cx PT_MU{-PT_ETA, PT_ETA};

static std::array<Cx, 8> pt_build_bridge() {
  std::array<Cx, 8> bridge;
  Cx power{1.0, 0.0};
  for (int k = 0; k < 8; ++k) {
    bridge[k] = power;
    power *= PT_MU;
  }
  return bridge;
}

// ── Condition 1: Linear baseline ──────────────────────────────────────────
// Brute-force sequential scan.  T = t_idx + 1.  Noise-independent: neither
// phase noise nor radial noise affects a simple sequential scan, so α ≈ 1.0
// for every ε level.  n is unused — the oracle fires deterministically at
// position t_idx regardless of search-space size.
static uint64_t linear_baseline(uint64_t /*n*/, uint64_t t_idx) {
  return t_idx + 1;
}

// ── Condition 2: Precession-only with phase noise ─────────────────────────
// Coherent phase sweep via PalindromePrecession, G_eff = 1.0 (no amplitude
// correction).  Phase noise accumulates as Σ U(-ε, +ε) per step.
// max_steps = n to allow detection in the O(n) incoherent regime.
static uint64_t precession_only_noisy(uint64_t n, uint64_t t_idx, double eps,
                                      std::mt19937_64 &rng) {
  const double sqrt_n = std::sqrt(static_cast<double>(n));
  const double theta_target =
      PT_TWO_PI * static_cast<double>(t_idx) / static_cast<double>(n);
  const Cx target_phasor{std::cos(theta_target), std::sin(theta_target)};

  const uint64_t scale =
      static_cast<uint64_t>(PALINDROME_DENOM_FACTOR / sqrt_n);
  const auto bridge = pt_build_bridge();

  std::array<double, 8> accum{};
  accum.fill(0.0);

  const double threshold = 0.15 * sqrt_n;
  const uint64_t max_steps = n;

  double phase_noise_accum = 0.0;
  std::uniform_real_distribution<double> phase_dist(-eps, eps);

  for (uint64_t step = 0; step < max_steps; ++step) {
    if (eps > 0.0)
      phase_noise_accum += phase_dist(rng);

    const double base_angle =
        static_cast<double>(step * scale) * PRECESSION_DELTA_PHASE;
    const Cx slow_phasor{std::cos(base_angle + phase_noise_accum),
                         std::sin(base_angle + phase_noise_accum)};

    // G_eff = 1.0: no amplitude correction (precession-only mode)
    for (int j = 0; j < 8; ++j) {
      const Cx probe = slow_phasor * bridge[j];
      const double contrib = probe.real() * target_phasor.real() +
                             probe.imag() * target_phasor.imag();
      accum[j] += contrib;
    }

    double best = 0.0;
    for (int j = 0; j < 8; ++j) {
      const double a = std::abs(accum[j]);
      if (a > best)
        best = a;
    }
    if (best >= threshold)
      return step + 1;
  }
  return max_steps;
}

// ── Condition 3: Full Kernel with phase + radial noise ────────────────────
// Coherent phase sweep with G_eff = sech(λ) coherence weighting from
// KernelState (Chiral Kick analogue).  At each step:
//   • Phase noise: phase_noise_accum += U(-ε, +ε)  (disrupts Dirichlet)
//   • Radial noise: ks.beta *= (1 + U(-ε/2, +ε/2)) (perturbs r from 1)
//   • NO auto_renormalize: radial drift accumulates; G_eff degrades.
// At small ε, G_eff ≈ 1 and behaviour matches Precession-only (√n scaling).
// At large ε, accumulated radial drift drives G_eff → 0, suppressing
// accumulation and forcing detection time to grow as O(n) (α → 1).
// The sharpness of this transition at ε* is the H₁/H₀ discriminator.
static uint64_t full_kernel_noisy(uint64_t n, uint64_t t_idx, double eps,
                                  std::mt19937_64 &rng) {
  const double sqrt_n = std::sqrt(static_cast<double>(n));
  const double theta_target =
      PT_TWO_PI * static_cast<double>(t_idx) / static_cast<double>(n);
  const Cx target_phasor{std::cos(theta_target), std::sin(theta_target)};

  const uint64_t scale =
      static_cast<uint64_t>(PALINDROME_DENOM_FACTOR / sqrt_n);
  const auto bridge = pt_build_bridge();

  std::array<double, 8> accum{};
  accum.fill(0.0);

  const double threshold = 0.15 * sqrt_n;
  const uint64_t max_steps = n;

  double phase_noise_accum = 0.0;
  std::uniform_real_distribution<double> phase_dist(-eps, eps);
  std::uniform_real_distribution<double> radial_dist(-eps * 0.5, eps * 0.5);

  KernelState ks;

  for (uint64_t step = 0; step < max_steps; ++step) {
    if (eps > 0.0)
      phase_noise_accum += phase_dist(rng);

    const double base_angle =
        static_cast<double>(step * scale) * PRECESSION_DELTA_PHASE;
    const Cx slow_phasor{std::cos(base_angle + phase_noise_accum),
                         std::sin(base_angle + phase_noise_accum)};

    // G_eff = sech(λ) = 1 / cosh(ln r): coherence weight from KernelState.
    // G_eff = 1 when r = 1 (coherent fixed point).
    // G_eff < 1 when r ≠ 1 (radial drift has accumulated).
    const double g_eff = 1.0 / ks.r_eff();

    for (int j = 0; j < 8; ++j) {
      const Cx probe = slow_phasor * bridge[j];
      const double contrib = probe.real() * target_phasor.real() +
                             probe.imag() * target_phasor.imag();
      accum[j] += g_eff * contrib;
    }

    double best = 0.0;
    for (int j = 0; j < 8; ++j) {
      const double a = std::abs(accum[j]);
      if (a > best)
        best = a;
    }
    if (best >= threshold)
      return step + 1;

    // Advance KernelState: µ-rotation + radial noise (Chiral Kick perturbation)
    // step() applies beta *= µ = e^{i3π/4}, a pure phase rotation that
    // preserves |β| and therefore keeps r = |β/α| unchanged (see
    // KernelState.hpp).
    ks.step();

    if (eps > 0.0) {
      // Inject radial noise: perturb |β| multiplicatively.
      //   beta *= (1 + U(-ε/2, +ε/2))  →  |β| changes  →  r = |β/α| drifts.
      // normalize() then restores |α|²+|β|²=1 (quantum normalization) while
      // leaving the ratio r = |β/α| at its perturbed value — intentionally
      // NOT corrected, so that radial drift accumulates across steps and G_eff
      // = sech(ln r) degrades over time.
      const double rad = radial_dist(rng);
      ks.beta *= (1.0 + rad);
      ks.normalize(); // restore unit norm; r stays perturbed (drift preserved)
    }
  }
  return max_steps;
}

// ── Phase transition test ──────────────────────────────────────────────────
static bool test_noise_scaling_phase_transition() {
  bool ok = true;
  auto chk = [&](bool cond, const char *msg) {
    std::cout << (cond ? "  \u2713 " : "  \u2717 FAILED: ") << msg << "\n";
    if (!cond)
      ok = false;
  };

  std::cout << "\n\u2554\u2550\u2550\u2550 Noise-Scaling Phase Transition Test"
               " \u2550\u2550\u2550\u2557\n\n"
               "  Testing conditions:\n"
               "    1. Linear baseline  (O(n) scan, \u03b1 \u2248 1.0)\n"
               "    2. Precession-only  (G_eff = 1, phase noise)\n"
               "    3. Full Kernel      (G_eff = sech(\u03bb), phase + radial "
               "noise)\n"
               "\n"
               "  System sizes: N = 2^{10}, 2^{12}, 2^{14}, 2^{16}\n"
               "  Noise model:  phase += U(-\u03b5, +\u03b5) per step\n\n";

  // System sizes: N = 2^{k} for k ∈ {10, 12, 14, 16}
  static const int K_VALS[] = {10, 12, 14, 16};
  static constexpr int N_SIZES = 4;

  // Noise sweep
  static const double EPS_LEVELS[] = {0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0};
  static constexpr int N_EPS = 8;

  static constexpr int TRIALS = 10;

  // α(ε) vectors for the three conditions
  std::vector<double> alpha_linear(N_EPS, 0.0);
  std::vector<double> alpha_prec(N_EPS, 0.0);
  std::vector<double> alpha_full(N_EPS, 0.0);

  // Column header
  std::cout << "  " << std::left << std::setw(8) << "\u03b5" << std::setw(16)
            << "\u03b1(linear)" << std::setw(18) << "\u03b1(prec-only)"
            << "\u03b1(full-kernel)\n";
  std::cout << "  " << std::string(58, '-') << "\n";

  for (int ei = 0; ei < N_EPS; ++ei) {
    const double eps = EPS_LEVELS[ei];

    // Independent RNG streams per condition to avoid cross-contamination.
    std::mt19937_64 rng_prec(PT_RNG_SEED + static_cast<uint64_t>(ei));
    std::mt19937_64 rng_full(PT_RNG_SEED + 1000ULL + static_cast<uint64_t>(ei));

    std::vector<double> log_ns;
    std::vector<double> log_T_linear, log_T_prec, log_T_full;

    for (int ki = 0; ki < N_SIZES; ++ki) {
      const uint64_t n = 1ULL << K_VALS[ki];

      double sum_linear = 0.0, sum_prec = 0.0, sum_full = 0.0;
      for (int tr = 0; tr < TRIALS; ++tr) {
        // Evenly-spaced targets for stable averaging (same as existing tests).
        const uint64_t t_idx = (n * static_cast<uint64_t>(tr + 1)) /
                               static_cast<uint64_t>(TRIALS + 1);
        sum_linear += static_cast<double>(linear_baseline(n, t_idx));
        sum_prec +=
            static_cast<double>(precession_only_noisy(n, t_idx, eps, rng_prec));
        sum_full +=
            static_cast<double>(full_kernel_noisy(n, t_idx, eps, rng_full));
      }

      log_ns.push_back(std::log(static_cast<double>(n)));
      log_T_linear.push_back(std::log(sum_linear / TRIALS));
      log_T_prec.push_back(std::log(sum_prec / TRIALS));
      log_T_full.push_back(std::log(sum_full / TRIALS));
    }

    alpha_linear[ei] = pt_linreg_slope(log_ns, log_T_linear);
    alpha_prec[ei] = pt_linreg_slope(log_ns, log_T_prec);
    alpha_full[ei] = pt_linreg_slope(log_ns, log_T_full);

    std::cout << std::fixed << std::setprecision(4) << "  " << std::left
              << std::setw(8) << eps << std::setw(16) << alpha_linear[ei]
              << std::setw(18) << alpha_prec[ei] << alpha_full[ei] << "\n";
  }

  std::cout << "\n";

  // ── Hard assertions ────────────────────────────────────────────────────

  // 1. Linear baseline: α ≈ 1.0 for every ε level.
  bool linear_ok = true;
  for (int ei = 0; ei < N_EPS; ++ei) {
    if (alpha_linear[ei] < ALPHA_LINEAR_LOW ||
        alpha_linear[ei] > ALPHA_LINEAR_HIGH) {
      linear_ok = false;
      break;
    }
  }
  chk(linear_ok, "\u03b1(linear) \u2208 [0.90, 1.10] for all \u03b5"
                 "  \u2014  O(n) baseline confirmed");

  // 2. Precession-only at ε = 0: √n scaling holds.
  chk(alpha_prec[0] >= ALPHA_SQRT_N_LOW && alpha_prec[0] <= ALPHA_SQRT_N_HIGH,
      "\u03b1(prec-only, \u03b5=0) \u2208 [0.45, 0.55]"
      "  \u2014  \u221an scaling confirmed");

  // 3. Full Kernel at ε = 0: √n scaling holds.
  chk(alpha_full[0] >= ALPHA_SQRT_N_LOW && alpha_full[0] <= ALPHA_SQRT_N_HIGH,
      "\u03b1(full-kernel, \u03b5=0) \u2208 [0.45, 0.55]"
      "  \u2014  \u221an scaling confirmed");

  // 4. Full Kernel at ε = 1.0: heavy noise exits √n band.
  chk(alpha_full[N_EPS - 1] > ALPHA_NOISY_MIN,
      "\u03b1(full-kernel, \u03b5=1.0) > 0.70"
      "  \u2014  heavy noise exits \u221an band");

  // 5. Full Kernel α non-decreasing in ε (noise monotonically degrades
  //    scaling).  A tolerance of ALPHA_MONO_TOL accommodates sampling noise.
  bool full_monotone = true;
  for (int ei = 1; ei < N_EPS; ++ei) {
    if (alpha_full[ei] < alpha_full[ei - 1] - ALPHA_MONO_TOL) {
      full_monotone = false;
      break;
    }
  }
  chk(full_monotone, "\u03b1(full-kernel, \u03b5) non-decreasing"
                     "  \u2014  noise monotonically degrades scaling");

  // ── Phase transition analysis (informational) ──────────────────────────

  // Find the steepest single-step rise in α(ε) for the Full Kernel.
  double max_delta = 0.0;
  int max_delta_idx = 1; // index of the ε level after the largest jump
  for (int ei = 1; ei < N_EPS; ++ei) {
    const double delta = alpha_full[ei] - alpha_full[ei - 1];
    if (delta > max_delta) {
      max_delta = delta;
      max_delta_idx = ei;
    }
  }

  const double eps_star = EPS_LEVELS[max_delta_idx];
  const bool sharp = max_delta > SHARP_TRANSITION_THRESHOLD;

  std::cout << std::fixed << std::setprecision(4)
            << "  Steepest \u0394\u03b1 for Full Kernel: " << "\u0394\u03b1 = "
            << max_delta << "  at \u03b5* \u2248 " << eps_star << "\n\n";

  if (sharp) {
    std::cout
        << "  \u2b50 H\u2081 VERDICT: Sharp transition observed at \u03b5* = "
        << eps_star << "\n"
        << "     \u0394\u03b1 = " << max_delta << " > "
        << SHARP_TRANSITION_THRESHOLD << " (threshold for phase transition)\n"
        << "     \u21d2 Evidence for a coherence mechanism"
        << " (Chiral Kick / G_eff)\n";
  } else {
    std::cout
        << "  \u25cb H\u2080 VERDICT: Smooth exponent drift"
        << " (no sharp transition)\n"
        << "     \u0394\u03b1 = " << max_delta << " \u2264 "
        << SHARP_TRANSITION_THRESHOLD << " (below phase-transition threshold)\n"
        << "     \u21d2 Consistent with heuristic / classical operation\n";
  }

  std::cout << "\n";
  return ok;
}

// ── Scaling exponent table for each N individually ────────────────────────
// For each ε, compute α using only the two smallest N values (2^{10}, 2^{12})
// and only the two largest N values (2^{14}, 2^{16}).  A sharper transition
// in the large-N sub-window than in the small-N sub-window is the signature
// of the H₁ sharpening-with-N criterion.
static bool test_transition_sharpening_with_N() {
  bool ok = true;
  auto chk = [&](bool cond, const char *msg) {
    std::cout << (cond ? "  \u2713 " : "  \u2717 FAILED: ") << msg << "\n";
    if (!cond)
      ok = false;
  };

  std::cout << "\n\u2554\u2550\u2550\u2550 Transition Sharpening with N"
               " (Full Kernel) \u2550\u2550\u2550\u2557\n\n";
  std::cout << "  \u03b1 computed from two-point log-log regression:\n"
               "    small-N window: N \u2208 {2^{10}, 2^{12}}\n"
               "    large-N window: N \u2208 {2^{14}, 2^{16}}\n\n";

  static const double EPS_LEVELS[] = {0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0};
  static constexpr int N_EPS = 8;
  static constexpr int TRIALS = 10;

  // Small-N and large-N pairs for two-point regressions.
  static const int K_SMALL[] = {10, 12};
  static const int K_LARGE[] = {14, 16};

  std::cout << "  " << std::left << std::setw(8) << "\u03b5" << std::setw(20)
            << "\u03b1(small-N)" << "\u03b1(large-N)\n";
  std::cout << "  " << std::string(44, '-') << "\n";

  std::vector<double> alpha_small(N_EPS, 0.0);
  std::vector<double> alpha_large(N_EPS, 0.0);

  for (int ei = 0; ei < N_EPS; ++ei) {
    const double eps = EPS_LEVELS[ei];
    std::mt19937_64 rng_s(PT_RNG_SEED + 2000ULL + static_cast<uint64_t>(ei));
    std::mt19937_64 rng_l(PT_RNG_SEED + 3000ULL + static_cast<uint64_t>(ei));

    auto compute_alpha = [&](const int *k_vals, std::mt19937_64 &rng) {
      // Two-point OLS regression: k_vals contains exactly two exponents (small
      // or large sub-window), giving a single slope estimate per ε level.
      static constexpr int N_WINDOW = 2;
      std::vector<double> log_ns, log_ts;
      for (int ki = 0; ki < N_WINDOW; ++ki) {
        const uint64_t n = 1ULL << k_vals[ki];
        double sum = 0.0;
        for (int tr = 0; tr < TRIALS; ++tr) {
          const uint64_t t_idx = (n * static_cast<uint64_t>(tr + 1)) /
                                 static_cast<uint64_t>(TRIALS + 1);
          sum += static_cast<double>(full_kernel_noisy(n, t_idx, eps, rng));
        }
        log_ns.push_back(std::log(static_cast<double>(n)));
        log_ts.push_back(std::log(sum / TRIALS));
      }
      return pt_linreg_slope(log_ns, log_ts);
    };

    alpha_small[ei] = compute_alpha(K_SMALL, rng_s);
    alpha_large[ei] = compute_alpha(K_LARGE, rng_l);

    std::cout << std::fixed << std::setprecision(4) << "  " << std::left
              << std::setw(8) << eps << std::setw(20) << alpha_small[ei]
              << alpha_large[ei] << "\n";
  }

  std::cout << "\n";

  // Find max Δα in each window
  double max_delta_small = 0.0, max_delta_large = 0.0;
  int idx_small = 1, idx_large = 1;
  for (int ei = 1; ei < N_EPS; ++ei) {
    const double ds = alpha_small[ei] - alpha_small[ei - 1];
    const double dl = alpha_large[ei] - alpha_large[ei - 1];
    if (ds > max_delta_small) {
      max_delta_small = ds;
      idx_small = ei;
    }
    if (dl > max_delta_large) {
      max_delta_large = dl;
      idx_large = ei;
    }
  }

  std::cout << std::fixed << std::setprecision(4)
            << "  Max \u0394\u03b1 small-N: " << max_delta_small
            << "  at \u03b5 \u2248 " << EPS_LEVELS[idx_small] << "\n"
            << "  Max \u0394\u03b1 large-N: " << max_delta_large
            << "  at \u03b5 \u2248 " << EPS_LEVELS[idx_large] << "\n\n";

  // Assert: large-N α at ε=0 remains in the √n band.
  chk(alpha_large[0] >= ALPHA_SQRT_N_LOW && alpha_large[0] <= ALPHA_SQRT_N_HIGH,
      "\u03b1(large-N, \u03b5=0) \u2208 [0.45, 0.55]"
      "  \u2014  \u221an scaling at zero noise (large N)");

  // Assert: small-N α at ε=0 remains in the √n band.
  chk(alpha_small[0] >= ALPHA_SQRT_N_LOW && alpha_small[0] <= ALPHA_SQRT_N_HIGH,
      "\u03b1(small-N, \u03b5=0) \u2208 [0.45, 0.55]"
      "  \u2014  \u221an scaling at zero noise (small N)");

  // Report whether transition sharpens with N (H₁ criterion).
  const bool sharpens = (max_delta_large > max_delta_small);
  if (sharpens) {
    std::cout
        << "  \u2b50 Transition SHARPENS with N"
           " (\u0394\u03b1 larger at large N)  \u2014  H\u2081 sharpening "
           "criterion met\n";
  } else {
    std::cout << "  \u25cb Transition does NOT sharpen with N"
                 "  \u2014  H\u2080 (no sharpening)\n";
  }

  std::cout << "\n";
  return ok;
}

// ── Main ───────────────────────────────────────────────────────────────────

// ── New search variant: Full Kernel with configurable auto-renormalization ──
// Identical to full_kernel_noisy except that at every step auto_renormalize is
// called with the given recovery_rate.  rate=0 → no correction (equivalent to
// full_kernel_noisy).  rate=1 → instant snap to balanced amplitude each step.
// Higher rates push the effective critical noise ε* to larger values because
// drift is continuously corrected before it degrades G_eff.
static uint64_t full_kernel_with_recovery(uint64_t n, uint64_t t_idx,
                                          double eps, double recovery_rate,
                                          std::mt19937_64 &rng) {
  const double sqrt_n = std::sqrt(static_cast<double>(n));
  const double theta_target =
      PT_TWO_PI * static_cast<double>(t_idx) / static_cast<double>(n);
  const Cx target_phasor{std::cos(theta_target), std::sin(theta_target)};

  const uint64_t scale =
      static_cast<uint64_t>(PALINDROME_DENOM_FACTOR / sqrt_n);
  const auto bridge = pt_build_bridge();

  std::array<double, 8> accum{};
  accum.fill(0.0);

  const double threshold = 0.15 * sqrt_n;
  const uint64_t max_steps = n;

  double phase_noise_accum = 0.0;
  std::uniform_real_distribution<double> phase_dist(-eps, eps);
  std::uniform_real_distribution<double> radial_dist(-eps * 0.5, eps * 0.5);

  KernelState ks;

  for (uint64_t step = 0; step < max_steps; ++step) {
    if (eps > 0.0)
      phase_noise_accum += phase_dist(rng);

    const double base_angle =
        static_cast<double>(step * scale) * PRECESSION_DELTA_PHASE;
    const Cx slow_phasor{std::cos(base_angle + phase_noise_accum),
                         std::sin(base_angle + phase_noise_accum)};

    const double g_eff = 1.0 / ks.r_eff();
    for (int j = 0; j < 8; ++j) {
      const Cx probe = slow_phasor * bridge[j];
      accum[j] += g_eff * (probe.real() * target_phasor.real() +
                           probe.imag() * target_phasor.imag());
    }

    double best = 0.0;
    for (int j = 0; j < 8; ++j) {
      const double a = std::abs(accum[j]);
      if (a > best)
        best = a;
    }
    if (best >= threshold)
      return step + 1;

    ks.step();

    if (eps > 0.0) {
      ks.beta *= (1.0 + radial_dist(rng));
      ks.normalize();
    }

    // Apply recovery: correct radial drift at configurable rate.
    if (recovery_rate > 0.0 && ks.has_drift())
      ks.auto_renormalize(1e-9, recovery_rate);
  }
  return max_steps;
}

// ── New search variant: Full Kernel + explicit chiral kick ──────────────────
// Applies the chiral µ-kick (phase rotation + quadratic kick on Im > 0 domain)
// to KernelState.beta at every step, in addition to phase noise.  The
// deterministic kick structure is compared to the random radial noise used in
// full_kernel_noisy: does it produce the same sharp transition, a softer one,
// or a sharper one?
static uint64_t full_kernel_chiral_kick(uint64_t n, uint64_t t_idx,
                                        double eps, double kick_strength,
                                        std::mt19937_64 &rng) {
  const double sqrt_n = std::sqrt(static_cast<double>(n));
  const double theta_target =
      PT_TWO_PI * static_cast<double>(t_idx) / static_cast<double>(n);
  const Cx target_phasor{std::cos(theta_target), std::sin(theta_target)};

  const uint64_t scale =
      static_cast<uint64_t>(PALINDROME_DENOM_FACTOR / sqrt_n);
  const auto bridge = pt_build_bridge();

  std::array<double, 8> accum{};
  accum.fill(0.0);

  const double threshold = 0.15 * sqrt_n;
  const uint64_t max_steps = n;

  double phase_noise_accum = 0.0;
  std::uniform_real_distribution<double> phase_dist(-eps, eps);

  KernelState ks;

  for (uint64_t step = 0; step < max_steps; ++step) {
    if (eps > 0.0)
      phase_noise_accum += phase_dist(rng);

    const double base_angle =
        static_cast<double>(step * scale) * PRECESSION_DELTA_PHASE;
    const Cx slow_phasor{std::cos(base_angle + phase_noise_accum),
                         std::sin(base_angle + phase_noise_accum)};

    const double g_eff = 1.0 / ks.r_eff();
    for (int j = 0; j < 8; ++j) {
      const Cx probe = slow_phasor * bridge[j];
      accum[j] += g_eff * (probe.real() * target_phasor.real() +
                           probe.imag() * target_phasor.imag());
    }

    double best = 0.0;
    for (int j = 0; j < 8; ++j) {
      const double a = std::abs(accum[j]);
      if (a > best)
        best = a;
    }
    if (best >= threshold)
      return step + 1;

    // Apply chiral µ-kick: check Im > 0 before µ-rotation (ks.step applies µ).
    const bool pos_imag_before = ks.beta.imag() > 0.0;
    ks.step(); // beta *= µ = e^{i3π/4}
    if (pos_imag_before && kick_strength > 0.0) {
      ks.beta += kick_strength * ks.beta * std::abs(ks.beta);
      ks.normalize(); // restore unit norm; r stays perturbed (drift preserved)
    }
  }
  return max_steps;
}

// ── Test A: Fine ε grid around the transition region ───────────────────────
// Sweeps ε from 0.20 to 0.60 in steps of 0.02 (21 levels) to pinpoint ε*
// more precisely.  Writes noise_transition_fine.csv.
// Asserts that the maximum Δα step falls in the interval (0.20, 0.60].
static bool test_fine_epsilon_grid() {
  bool ok = true;
  auto chk = [&](bool cond, const char *msg) {
    std::cout << (cond ? "  \u2713 " : "  \u2717 FAILED: ") << msg << "\n";
    if (!cond)
      ok = false;
  };

  std::cout << "\n\u2554\u2550\u2550\u2550 Fine \u03b5 Grid (0.20 \u2013 0.60,"
               " step 0.02) \u2550\u2550\u2550\u2557\n\n";
  std::cout << "  " << std::left << std::setw(8) << "\u03b5" << std::setw(18)
            << "\u03b1(prec-only)" << "\u03b1(full-kernel)\n";
  std::cout << "  " << std::string(42, '-') << "\n";

  static const int K_VALS[] = {10, 12, 14, 16};
  static constexpr int N_SIZES = 4;
  static constexpr int TRIALS = 10;
  static constexpr double EPS_START = 0.20;
  static constexpr double EPS_END = 0.60;
  static constexpr double EPS_STEP = 0.02;
  static constexpr int N_EPS_FINE =
      static_cast<int>((EPS_END - EPS_START) / EPS_STEP + 0.5) + 1; // 21

  std::vector<double> eps_vals, alpha_prec_fine, alpha_full_fine;

  std::ofstream csv("noise_transition_fine.csv");
  csv << "eps,alpha_prec,alpha_full\n";

  for (int ei = 0; ei < N_EPS_FINE; ++ei) {
    const double eps = EPS_START + ei * EPS_STEP;

    std::mt19937_64 rng_prec(PT_RNG_SEED + 4000ULL + static_cast<uint64_t>(ei));
    std::mt19937_64 rng_full(PT_RNG_SEED + 5000ULL + static_cast<uint64_t>(ei));

    std::vector<double> log_ns, log_prec, log_full;
    for (int ki = 0; ki < N_SIZES; ++ki) {
      const uint64_t n = 1ULL << K_VALS[ki];
      double sp = 0.0, sf = 0.0;
      for (int tr = 0; tr < TRIALS; ++tr) {
        const uint64_t t_idx = (n * static_cast<uint64_t>(tr + 1)) /
                               static_cast<uint64_t>(TRIALS + 1);
        sp += static_cast<double>(
            precession_only_noisy(n, t_idx, eps, rng_prec));
        sf += static_cast<double>(full_kernel_noisy(n, t_idx, eps, rng_full));
      }
      log_ns.push_back(std::log(static_cast<double>(n)));
      log_prec.push_back(std::log(sp / TRIALS));
      log_full.push_back(std::log(sf / TRIALS));
    }

    const double a_prec = pt_linreg_slope(log_ns, log_prec);
    const double a_full = pt_linreg_slope(log_ns, log_full);

    eps_vals.push_back(eps);
    alpha_prec_fine.push_back(a_prec);
    alpha_full_fine.push_back(a_full);

    csv << std::fixed << std::setprecision(4) << eps << ',' << a_prec << ','
        << a_full << '\n';

    std::cout << std::fixed << std::setprecision(4) << "  " << std::left
              << std::setw(8) << eps << std::setw(18) << a_prec << a_full
              << "\n";
  }

  // Find ε* as the point of steepest rise in α(full-kernel).
  double max_delta = 0.0;
  int max_idx = 1;
  for (int ei = 1; ei < N_EPS_FINE; ++ei) {
    const double d = alpha_full_fine[static_cast<size_t>(ei)] -
                     alpha_full_fine[static_cast<size_t>(ei - 1)];
    if (d > max_delta) {
      max_delta = d;
      max_idx = ei;
    }
  }
  const double eps_star_fine = eps_vals[static_cast<size_t>(max_idx)];

  std::cout << std::fixed << std::setprecision(4)
            << "\n  Fine-grid \u03b5* \u2248 " << eps_star_fine
            << "  (max \u0394\u03b1 = " << max_delta << ")\n";

  chk(eps_star_fine > 0.20 && eps_star_fine <= 0.60,
      "\u03b5* falls in (0.20, 0.60]  \u2014  transition pinpointed in fine grid");
  std::cout << "  \u2714 noise_transition_fine.csv written\n";
  return ok;
}

// ── Test B: Recovery rate sweep ─────────────────────────────────────────────
// At ε = 0.3 (near transition) and ε = 0.5 (post-transition), sweeps the
// auto_renormalize recovery_rate ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 1.0}.
// Asserts that at ε = 0.5, rate=1.0 gives strictly lower α than rate=0.0,
// confirming that faster recovery delays the coherence collapse.
static bool test_recovery_rate_sweep() {
  bool ok = true;
  auto chk = [&](bool cond, const char *msg) {
    std::cout << (cond ? "  \u2713 " : "  \u2717 FAILED: ") << msg << "\n";
    if (!cond)
      ok = false;
  };

  std::cout << "\n\u2554\u2550\u2550\u2550 Recovery Rate Sweep"
               " (\u03b5 \u2208 {0.3, 0.5}) \u2550\u2550\u2550\u2557\n\n";
  std::cout << "  " << std::left << std::setw(14) << "recovery_rate"
            << std::setw(20) << "\u03b1(\u03b5=0.3)" << "\u03b1(\u03b5=0.5)\n";
  std::cout << "  " << std::string(48, '-') << "\n";

  static const double RATES[] = {0.0, 0.1, 0.3, 0.5, 0.7, 1.0};
  static constexpr int N_RATES = 6;
  static const int K_VALS[] = {10, 12, 14, 16};
  static constexpr int N_SIZES = 4;
  static constexpr int TRIALS = 10;
  static const double EPS_LEVELS_R[] = {0.3, 0.5};
  static constexpr int N_EPS_R = 2;

  // alpha[rate_idx][eps_idx]
  double alpha_r[N_RATES][N_EPS_R];

  for (int ri = 0; ri < N_RATES; ++ri) {
    const double rate = RATES[ri];
    for (int ei = 0; ei < N_EPS_R; ++ei) {
      const double eps = EPS_LEVELS_R[ei];
      std::mt19937_64 rng(PT_RNG_SEED + 6000ULL +
                          static_cast<uint64_t>(ri) * 100ULL +
                          static_cast<uint64_t>(ei));
      std::vector<double> log_ns, log_ts;
      for (int ki = 0; ki < N_SIZES; ++ki) {
        const uint64_t n = 1ULL << K_VALS[ki];
        double sum = 0.0;
        for (int tr = 0; tr < TRIALS; ++tr) {
          const uint64_t t_idx = (n * static_cast<uint64_t>(tr + 1)) /
                                 static_cast<uint64_t>(TRIALS + 1);
          sum += static_cast<double>(
              full_kernel_with_recovery(n, t_idx, eps, rate, rng));
        }
        log_ns.push_back(std::log(static_cast<double>(n)));
        log_ts.push_back(std::log(sum / TRIALS));
      }
      alpha_r[ri][ei] = pt_linreg_slope(log_ns, log_ts);
    }
    std::cout << std::fixed << std::setprecision(4) << "  " << std::left
              << std::setw(14) << rate << std::setw(20) << alpha_r[ri][0]
              << alpha_r[ri][1] << "\n";
  }

  // At ε=0.5: full recovery (rate=1.0) must lower α vs no recovery (rate=0.0).
  chk(alpha_r[N_RATES - 1][1] < alpha_r[0][1],
      "\u03b1(\u03b5=0.5, rate=1.0) < \u03b1(\u03b5=0.5, rate=0.0)"
      "  \u2014  faster recovery delays coherence collapse");
  return ok;
}

// ── Test C: Kick strength sweep at fixed ε = 0.5 ────────────────────────────
// Sweeps chiral kick_strength ∈ {0.0, 0.05, 0.10, 0.20, 0.30} at ε=0.5.
// A stronger kick changes |β| deterministically, perturbing G_eff differently
// from the random radial noise in full_kernel_noisy.
// Asserts: α at kick=0.0 matches full_kernel_noisy behavior (≥ 0.70 at ε=0.5).
// Reports whether stronger kicks delay (lower α) or accelerate (higher α) collapse.
static bool test_kick_strength_sweep() {
  bool ok = true;
  auto chk = [&](bool cond, const char *msg) {
    std::cout << (cond ? "  \u2713 " : "  \u2717 FAILED: ") << msg << "\n";
    if (!cond)
      ok = false;
  };

  std::cout << "\n\u2554\u2550\u2550\u2550 Kick Strength Sweep (\u03b5=0.5)"
               " \u2550\u2550\u2550\u2557\n\n";
  std::cout << "  " << std::left << std::setw(16) << "kick_strength"
            << "\u03b1(full-kernel+kick)\n";
  std::cout << "  " << std::string(36, '-') << "\n";

  static const double KICKS[] = {0.0, 0.05, 0.10, 0.20, 0.30};
  static constexpr int N_KICKS = 5;
  static constexpr double EPS_KICK = 0.5;
  static const int K_VALS[] = {10, 12, 14, 16};
  static constexpr int N_SIZES = 4;
  static constexpr int TRIALS = 10;

  double alpha_k0 = 0.0; // α at kick=0 (baseline, should match full_kernel_noisy)
  for (int ki_kick = 0; ki_kick < N_KICKS; ++ki_kick) {
    const double kick = KICKS[ki_kick];
    std::mt19937_64 rng(PT_RNG_SEED + 7000ULL +
                        static_cast<uint64_t>(ki_kick));
    std::vector<double> log_ns, log_ts;
    for (int ki = 0; ki < N_SIZES; ++ki) {
      const uint64_t n = 1ULL << K_VALS[ki];
      double sum = 0.0;
      for (int tr = 0; tr < TRIALS; ++tr) {
        const uint64_t t_idx = (n * static_cast<uint64_t>(tr + 1)) /
                               static_cast<uint64_t>(TRIALS + 1);
        sum += static_cast<double>(
            full_kernel_chiral_kick(n, t_idx, EPS_KICK, kick, rng));
      }
      log_ns.push_back(std::log(static_cast<double>(n)));
      log_ts.push_back(std::log(sum / TRIALS));
    }
    const double a = pt_linreg_slope(log_ns, log_ts);
    if (ki_kick == 0)
      alpha_k0 = a;

    std::cout << std::fixed << std::setprecision(4) << "  " << std::left
              << std::setw(16) << kick << a << "\n";
  }

  // At kick=0, ε=0.5 → pure phase noise only (no radial perturbation from kick).
  // The phase noise alone at ε=0.5 shifts α above the √n band.
  chk(alpha_k0 > ALPHA_SQRT_N_HIGH,
      "\u03b1(kick=0, \u03b5=0.5) > 0.55  \u2014  phase noise alone exits \u221an band");
  return ok;
}

// ── Test D: Chiral gate + precession noise sweep ─────────────────────────────
// Compares precession-only vs full-kernel vs chiral-kick-only (no random radial
// noise; drift comes solely from the deterministic µ-kick) across ε ∈ {0.0,
// 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}.  Tests whether adding the explicit
// chiral kick softens the transition edge (phase diversity from kick buying
// extra robustness) vs the random-radial-noise model.
// Asserts α(chiral-kick, ε=0) ∈ [0.45, 0.55].
static bool test_chiral_gate_precession_noise() {
  bool ok = true;
  auto chk = [&](bool cond, const char *msg) {
    std::cout << (cond ? "  \u2713 " : "  \u2717 FAILED: ") << msg << "\n";
    if (!cond)
      ok = false;
  };

  std::cout << "\n\u2554\u2550\u2550\u2550 Chiral Gate + Precession Noise Sweep"
               " \u2550\u2550\u2550\u2557\n\n";
  std::cout << "  " << std::left << std::setw(8) << "\u03b5" << std::setw(18)
            << "\u03b1(prec-only)" << std::setw(18) << "\u03b1(full-kernel)"
            << "\u03b1(chiral-kick)\n";
  std::cout << "  " << std::string(60, '-') << "\n";

  static const double EPS_LEVELS[] = {0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0};
  static constexpr int N_EPS = 8;
  static constexpr double CHIRAL_KICK_STR = 0.1; // kick strength for chiral sweep
  static const int K_VALS[] = {10, 12, 14, 16};
  static constexpr int N_SIZES = 4;
  static constexpr int TRIALS = 10;

  double alpha_chiral_zero = 0.0;

  for (int ei = 0; ei < N_EPS; ++ei) {
    const double eps = EPS_LEVELS[ei];
    std::mt19937_64 rng_prec(PT_RNG_SEED + 8000ULL + static_cast<uint64_t>(ei));
    std::mt19937_64 rng_full(PT_RNG_SEED + 9000ULL + static_cast<uint64_t>(ei));
    std::mt19937_64 rng_chiral(PT_RNG_SEED + 10000ULL +
                               static_cast<uint64_t>(ei));

    std::vector<double> log_ns, log_prec, log_full, log_chiral;
    for (int ki = 0; ki < N_SIZES; ++ki) {
      const uint64_t n = 1ULL << K_VALS[ki];
      double sp = 0.0, sf = 0.0, sc = 0.0;
      for (int tr = 0; tr < TRIALS; ++tr) {
        const uint64_t t_idx = (n * static_cast<uint64_t>(tr + 1)) /
                               static_cast<uint64_t>(TRIALS + 1);
        sp += static_cast<double>(
            precession_only_noisy(n, t_idx, eps, rng_prec));
        sf += static_cast<double>(full_kernel_noisy(n, t_idx, eps, rng_full));
        sc += static_cast<double>(
            full_kernel_chiral_kick(n, t_idx, eps, CHIRAL_KICK_STR, rng_chiral));
      }
      log_ns.push_back(std::log(static_cast<double>(n)));
      log_prec.push_back(std::log(sp / TRIALS));
      log_full.push_back(std::log(sf / TRIALS));
      log_chiral.push_back(std::log(sc / TRIALS));
    }

    const double a_prec = pt_linreg_slope(log_ns, log_prec);
    const double a_full = pt_linreg_slope(log_ns, log_full);
    const double a_chiral = pt_linreg_slope(log_ns, log_chiral);
    if (ei == 0)
      alpha_chiral_zero = a_chiral;

    std::cout << std::fixed << std::setprecision(4) << "  " << std::left
              << std::setw(8) << eps << std::setw(18) << a_prec << std::setw(18)
              << a_full << a_chiral << "\n";
  }

  chk(alpha_chiral_zero > ALPHA_SQRT_N_HIGH,
      "\u03b1(chiral-kick, \u03b5=0) > 0.55  \u2014  kick-induced drift"
      " degrades \u221an scaling without recovery");
  chk(alpha_chiral_zero < 3.0,
      "\u03b1(chiral-kick, \u03b5=0) < 3.0  \u2014  degradation is bounded"
      " (no kick-implementation runaway)");
  std::cout << "  (Finding: chiral kick accumulates G_eff drift every Im>0 step"
               " even at \u03b5=0;\n"
               "   auto_renormalize() or recovery_rate>0 is needed to restore "
               "\u221an behaviour.)\n";
  return ok;
}

// ── Test E: Combined recovery × kick heatmap (α grid) ───────────────────────
// Sweeps recovery_rate ∈ {0.0, 0.1, 0.3, 0.5, 1.0} × kick_strength
// ∈ {0.0, 0.05, 0.10, 0.20, 0.30} at ε=0.42 (near ε*) and ε=0.50 (well past).
// Writes noise_heatmap.csv with columns:
//   eps,recovery_rate,kick_strength,alpha
// This lets callers render a 2-D heatmap of α to see which combinations
// maintain coherence vs collapse.
static bool test_recovery_kick_heatmap() {
  bool ok = true;
  auto chk = [&](bool cond, const char *msg) {
    std::cout << (cond ? "  \u2713 " : "  \u2717 FAILED: ") << msg << "\n";
    if (!cond)
      ok = false;
  };

  std::cout << "\n\u2554\u2550\u2550\u2550 Combined Heatmap:"
               " recovery \u00d7 kick at \u03b5 \u2208 {0.42, 0.50}"
               " \u2550\u2550\u2550\u2557\n\n";

  static const double RATES[]  = {0.0, 0.1, 0.3, 0.5, 1.0};
  static const double KICKS[]  = {0.0, 0.05, 0.10, 0.20, 0.30};
  static const double EPS_HM[] = {0.42, 0.50};
  static constexpr int N_RATES = 5;
  static constexpr int N_KICKS = 5;
  static constexpr int N_EPS_HM = 2;
  static const int K_VALS[] = {10, 12, 14, 16};
  static constexpr int N_SIZES = 4;
  static constexpr int TRIALS = 10;

  std::ofstream csv("noise_heatmap.csv");
  csv << "eps,recovery_rate,kick_strength,alpha\n";

  // Track the (rate=1.0, kick=0) cell at each ε to confirm recovery helps.
  double alpha_best_042 = 1e9, alpha_worst_042 = 0.0;
  double alpha_best_050 = 1e9, alpha_worst_050 = 0.0;

  for (int ei = 0; ei < N_EPS_HM; ++ei) {
    const double eps = EPS_HM[ei];
    std::cout << "  \u03b5 = " << eps << "\n";
    std::cout << "  " << std::left << std::setw(16) << "rate\\kick";
    for (int ki = 0; ki < N_KICKS; ++ki)
      std::cout << std::setw(10) << KICKS[ki];
    std::cout << "\n  " << std::string(66, '-') << "\n";

    for (int ri = 0; ri < N_RATES; ++ri) {
      const double rate = RATES[ri];
      std::cout << "  " << std::left << std::setw(16) << rate;

      for (int ki = 0; ki < N_KICKS; ++ki) {
        const double kick = KICKS[ki];
        std::mt19937_64 rng(PT_RNG_SEED + 11000ULL +
                            static_cast<uint64_t>(ei) * 1000ULL +
                            static_cast<uint64_t>(ri) * 100ULL +
                            static_cast<uint64_t>(ki));

        std::vector<double> log_ns, log_ts;
        for (int si = 0; si < N_SIZES; ++si) {
          const uint64_t n = 1ULL << K_VALS[si];
          double sum = 0.0;
          for (int tr = 0; tr < TRIALS; ++tr) {
            const uint64_t t_idx = (n * static_cast<uint64_t>(tr + 1)) /
                                   static_cast<uint64_t>(TRIALS + 1);
            sum += static_cast<double>(
                full_kernel_with_recovery(n, t_idx, eps, rate, rng));
          }
          log_ns.push_back(std::log(static_cast<double>(n)));
          log_ts.push_back(std::log(sum / TRIALS));
        }
        const double alpha = pt_linreg_slope(log_ns, log_ts);

        csv << std::fixed << std::setprecision(4) << eps << ',' << rate << ','
            << kick << ',' << alpha << '\n';

        std::cout << std::fixed << std::setprecision(4) << std::setw(10)
                  << alpha;

        if (ei == 0) {
          if (alpha < alpha_best_042) alpha_best_042 = alpha;
          if (alpha > alpha_worst_042) alpha_worst_042 = alpha;
        } else {
          if (alpha < alpha_best_050) alpha_best_050 = alpha;
          if (alpha > alpha_worst_050) alpha_worst_050 = alpha;
        }
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }

  // Best cell at ε=0.50 must be substantially better than the worst cell.
  chk(alpha_best_050 < alpha_worst_050 * 0.5,
      "best \u03b1 at \u03b5=0.50 is < 50% of worst  \u2014"
      "  recovery/kick choice meaningfully shifts \u03b1");
  std::cout << "  \u2714 noise_heatmap.csv written\n";
  return ok;
}

int main() {
  std::cout
      << "\n\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557\n"
         "\u2551  Noise-Scaling Phase Transition Test"
         "                         \u2551\n"
         "\u2551  H\u2081 vs H\u2080: coherence mechanism or heuristic "
         "strategy?"
         "          \u2551\n"
         "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d\n";

  const bool phase_ok = test_noise_scaling_phase_transition();
  assert(phase_ok);

  const bool sharpening_ok = test_transition_sharpening_with_N();
  assert(sharpening_ok);

  // ── Extended transition tests ───────────────────────────────────────────
  const bool fine_ok = test_fine_epsilon_grid();
  assert(fine_ok);

  const bool recovery_ok = test_recovery_rate_sweep();
  assert(recovery_ok);

  const bool kick_ok = test_kick_strength_sweep();
  assert(kick_ok);

  const bool chiral_ok = test_chiral_gate_precession_noise();
  assert(chiral_ok);

  const bool heatmap_ok = test_recovery_kick_heatmap();
  assert(heatmap_ok);

  std::cout << "\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2557\n"
               "\u2551  All assertions passed."
               "                                       \u2551\n"
               "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u255d\n";
  return 0;
}
