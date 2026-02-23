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
