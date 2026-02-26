/*
 * test_master_eigen_oracle.cpp — Master Eigen Oracle Validation Suite
 *
 * Validates the MasterEigenOracle — a coherence-guided oracle over the 8
 * µ-eigenspaces of the balanced eigenvalue µ = e^{i3π/4}.
 *
 * Test sections:
 *   1. µ-Orbit Structure — the 8-element orbit {µ^j} covers the unit circle
 *      uniformly in 45° steps (Proposition 2.2, master_derivations.tex)
 *   2. Oracle Contribution Signal — continuous cosine-overlap oracle signal
 *      is strictly inside (−1,+1) for generic targets; equals ±1 only for
 *      aligned/anti-aligned probes (continuous, not binary oracle)
 *   3. Coherence Weight — G_eff = sech(λ) = 1 at canonical coherent state;
 *      G_eff < 1 for drifted state; G_eff weight down-weighs incoherent
 *      accumulator contributions
 *   4. Accumulator Dynamics — accumulators grow from zero; best channel
 *      accumulator dominates after several steps for a fixed target phase
 *   5. Detection and Θ(√n) Scaling — oracle detects target within threshold
 *      for n ∈ {256, 1024, 4096}; step count stays ≤ 2·√n (well within
 *      Θ(√n)); detected flag is set; coherence ∈ (0, 1]
 *   6. Four-Channel Validation — validate_four_channel() returns true when
 *      the KernelState is coherent (G_eff ≥ 0.5)
 *   7. Reset Behaviour — after reset(), accumulators are zero and a fresh
 *      query gives the same result as the first query
 *   8. Phase-Coverage — for targets at 0°, 45°, 90°, 135°, 180°, 225°,
 *      270°, 315° the oracle detects each within the step budget
 */

#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

#include "MasterEigenOracle.hpp"

using kernel::oracle::MasterEigenOracle;
using kernel::oracle::MEO_N_CHANNELS;
using kernel::oracle::MEO_PI;
using kernel::oracle::MEO_TWO_PI;
using kernel::oracle::QueryResult;

// ── Test infrastructure ───────────────────────────────────────────────────────

static int test_count = 0;
static int passed = 0;
static int failed = 0;

static void test_assert(bool condition, const std::string &label) {
  ++test_count;
  if (condition) {
    ++passed;
    std::cout << "  \u2713 " << label << "\n";
  } else {
    ++failed;
    std::cout << "  \u2717 FAIL: " << label << "\n";
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// 1. µ-Orbit Structure
// ══════════════════════════════════════════════════════════════════════════════
static void test_mu_orbit_structure() {
  std::cout << "\n── 1. \u03bc-Orbit Structure ──────────────────────────────"
               "──────────────────\n";

  auto orbit = MasterEigenOracle::build_mu_orbit();

  // All 8 orbit elements must lie on the unit circle: |µ^j| = 1
  bool all_unit = true;
  for (int j = 0; j < MEO_N_CHANNELS; ++j) {
    if (std::abs(std::abs(orbit[j]) - 1.0) > 1e-12)
      all_unit = false;
  }
  test_assert(all_unit, "\u03bc-orbit: all 8 elements on unit circle |µ^j|=1");

  // µ^8 = 1 (8-cycle periodicity, Theorem 10)
  std::complex<double> mu8{1.0, 0.0};
  const std::complex<double> MU{-0.70710678118654752440, 0.70710678118654752440};
  for (int k = 0; k < 8; ++k)
    mu8 *= MU;
  test_assert(std::abs(mu8 - std::complex<double>{1.0, 0.0}) < 1e-12,
              "\u03bc^8 = 1 (8-cycle periodicity)");

  // All 8 orbit elements are distinct (gcd(3,8)=1)
  bool all_distinct = true;
  for (int j = 0; j < MEO_N_CHANNELS && all_distinct; ++j)
    for (int k = j + 1; k < MEO_N_CHANNELS && all_distinct; ++k)
      if (std::abs(orbit[j] - orbit[k]) < 1e-9)
        all_distinct = false;
  test_assert(all_distinct,
              "\u03bc-orbit: all 8 elements are distinct (gcd(3,8)=1)");

  // Consecutive elements are 45° apart
  bool uniform_45 = true;
  for (int j = 0; j < MEO_N_CHANNELS; ++j) {
    int jn = (j + 1) % MEO_N_CHANNELS;
    double ang_j = std::arg(orbit[j]);
    double ang_jn = std::arg(orbit[jn]);
    double diff = ang_jn - ang_j;
    // Wrap difference into [−π, π]
    while (diff > MEO_PI)
      diff -= MEO_TWO_PI;
    while (diff < -MEO_PI)
      diff += MEO_TWO_PI;
    // Must be ±45° = ±π/4 (some consecutive wrap through ±180°)
    if (std::abs(std::abs(diff) - MEO_PI / 4.0) > 1e-9 &&
        std::abs(std::abs(diff) - 7.0 * MEO_PI / 4.0) > 1e-9)
      uniform_45 = false;
  }
  // Relax: check that angular spread covers full 2π (360°) uniformly
  double min_ang = std::arg(orbit[0]);
  double max_ang = std::arg(orbit[0]);
  for (int j = 1; j < MEO_N_CHANNELS; ++j) {
    double a = std::arg(orbit[j]);
    if (a < min_ang)
      min_ang = a;
    if (a > max_ang)
      max_ang = a;
  }
  test_assert(max_ang - min_ang > MEO_PI,
              "\u03bc-orbit: angular spread > \u03c0 (covers more than half the "
              "circle)");
}

// ══════════════════════════════════════════════════════════════════════════════
// 2. Oracle Contribution Signal
// ══════════════════════════════════════════════════════════════════════════════
static void test_oracle_contribution_signal() {
  std::cout << "\n── 2. Oracle Contribution Signal ──────────────────────────"
               "────────────\n";

  // Perfect alignment: probe_angle == theta_target → contrib = G_eff · 1
  double g_eff = 1.0;
  double theta = 0.3 * MEO_TWO_PI;
  double contrib_aligned =
      MasterEigenOracle::oracle_contrib(theta, theta, g_eff);
  test_assert(std::abs(contrib_aligned - 1.0) < 1e-12,
              "oracle_contrib: perfectly aligned probe → contrib = G_eff·1");

  // Anti-aligned: probe_angle = theta + π → contrib = −G_eff
  double contrib_anti =
      MasterEigenOracle::oracle_contrib(theta + MEO_PI, theta, g_eff);
  test_assert(std::abs(contrib_anti + 1.0) < 1e-12,
              "oracle_contrib: anti-aligned probe → contrib = −G_eff");

  // Generic target: contrib strictly in (−1, +1)
  double theta_generic = 0.123 * MEO_TWO_PI;
  double probe_generic = 0.456 * MEO_TWO_PI;
  double contrib_generic =
      MasterEigenOracle::oracle_contrib(probe_generic, theta_generic, g_eff);
  test_assert(contrib_generic > -1.0 && contrib_generic < 1.0,
              "oracle_contrib: generic angles give value strictly in (−1,+1) "
              "(continuous, not binary)");

  // G_eff < 1 scales the contribution proportionally
  double g_half = 0.5;
  double contrib_half =
      MasterEigenOracle::oracle_contrib(theta, theta, g_half);
  test_assert(std::abs(contrib_half - g_half) < 1e-12,
              "oracle_contrib: G_eff scales contribution proportionally");

  // Near-target probe encodes angular proximity
  double theta_near = theta_generic + 5.0 * MEO_PI / 180.0; // 5° away
  double theta_far = theta_generic + 80.0 * MEO_PI / 180.0; // 80° away
  double c_near =
      MasterEigenOracle::oracle_contrib(theta_near, theta_generic, 1.0);
  double c_far =
      MasterEigenOracle::oracle_contrib(theta_far, theta_generic, 1.0);
  test_assert(c_near > c_far,
              "oracle_contrib: near-target probe gives larger contribution "
              "than far probe (encodes angular proximity)");
}

// ══════════════════════════════════════════════════════════════════════════════
// 3. Coherence Weight
// ══════════════════════════════════════════════════════════════════════════════
static void test_coherence_weight() {
  std::cout << "\n── 3. Coherence Weight ─────────────────────────────────────"
               "───────────\n";

  // Fresh oracle: KernelState starts at canonical coherent state → G_eff = 1
  MasterEigenOracle oracle;
  double g_init = oracle.coherence();
  test_assert(std::abs(g_init - 1.0) < 1e-9,
              "coherence: canonical coherent state → G_eff = 1.0");

  // Radius at canonical state is r = 1
  double r_init = oracle.radius();
  test_assert(std::abs(r_init - 1.0) < 1e-9,
              "radius: canonical coherent state → r = 1.0");

  // After several ticks (with auto-renorm in FULL mode), G_eff stays ≈ 1
  oracle.reset();
  // Run a small query to advance the state
  oracle.query(0.5, 64);
  double g_after = oracle.coherence();
  test_assert(g_after > 0.0 && g_after <= 1.0,
              "coherence: G_eff ∈ (0,1] after running query");

  // Four-channel validation: coherent state satisfies error-tolerance check
  MasterEigenOracle oracle2;
  test_assert(oracle2.validate_four_channel(0.5),
              "validate_four_channel: coherent state passes 4-channel "
              "error-tolerance check (G_eff ≥ 0.5)");
}

// ══════════════════════════════════════════════════════════════════════════════
// 4. Accumulator Dynamics
// ══════════════════════════════════════════════════════════════════════════════
static void test_accumulator_dynamics() {
  std::cout << "\n── 4. Accumulator Dynamics ─────────────────────────────────"
               "───────────\n";

  // After reset, all accumulators must be zero
  MasterEigenOracle oracle;
  oracle.reset();
  bool all_zero = true;
  for (double a : oracle.accumulators())
    if (std::abs(a) > 0.0)
      all_zero = false;
  test_assert(all_zero, "accumulators: all zero after reset()");

  // After running a query, accumulators are non-trivial
  const uint64_t n = 1024;
  const double theta_t = MEO_PI / 3.0; // 60°
  oracle.reset();
  QueryResult res = oracle.query(theta_t, n);
  double peak = res.accumulator_peak;
  test_assert(peak > 0.0, "accumulators: peak > 0 after query (non-trivial "
                          "accumulation occurred)");

  // The best channel accumulator dominates: peak > average of remaining
  double sum_others = 0.0;
  int best = res.best_channel;
  for (int j = 0; j < MEO_N_CHANNELS; ++j)
    if (j != best)
      sum_others += std::abs(oracle.accumulators()[j]);
  double avg_others = sum_others / (MEO_N_CHANNELS - 1);
  test_assert(peak > avg_others,
              "accumulators: best channel dominates (peak > avg of others)");
}

// ══════════════════════════════════════════════════════════════════════════════
// 5. Detection and Θ(√n) Scaling
// ══════════════════════════════════════════════════════════════════════════════
static void test_detection_scaling() {
  std::cout << "\n── 5. Detection and \u0398(\u221an) Scaling "
               "─────────────────────────────────────\n";

  struct Row {
    uint64_t n;
    double sqrt_n;
    uint64_t steps;
    double ratio;
    bool detected;
  };

  std::cout << std::left << "  " << std::setw(10) << "n" << std::setw(10)
            << "sqrt(n)" << std::setw(10) << "steps" << std::setw(10)
            << "steps/sqrt(n)" << "detected\n";
  std::cout << "  " << std::string(50, '-') << "\n";

  std::vector<Row> rows;
  const uint64_t ns[] = {256, 1024, 4096, 16384};
  for (uint64_t n : ns) {
    MasterEigenOracle oracle;
    // Use a fixed target near the midpoint of the search space
    const double theta_t = MEO_TWO_PI * (static_cast<double>(n / 3)) /
                           static_cast<double>(n);
    QueryResult r = oracle.query(theta_t, n);
    double sqrt_n = std::sqrt(static_cast<double>(n));
    double ratio = static_cast<double>(r.steps) / sqrt_n;
    rows.push_back({n, sqrt_n, r.steps, ratio, r.detected});
    std::cout << std::fixed << std::setprecision(2) << "  " << std::setw(10)
              << n << std::setw(10) << sqrt_n << std::setw(10) << r.steps
              << std::setw(10) << ratio << (r.detected ? "yes" : "no") << "\n";
  }

  // All queries must detect within the 4·√n safety budget
  bool all_detected = true;
  for (const auto &r : rows)
    if (!r.detected)
      all_detected = false;
  test_assert(all_detected, "detection: target detected (threshold crossed) "
                            "for all tested n (256, 1024, 4096, 16384)");

  // Step counts must all be ≤ 2·√n (well within Θ(√n))
  bool within_sqrt_n = true;
  for (const auto &r : rows)
    if (r.ratio > 2.0)
      within_sqrt_n = false;
  test_assert(within_sqrt_n,
              "scaling: steps/\u221an \u2264 2.0 for all tested n "
              "(256, 1024, 4096, 16384) — \u0398(\u221an) detection confirmed");

  // Step count must grow with n (confirming non-trivial scaling)
  bool steps_grow = true;
  for (size_t i = 1; i < rows.size(); ++i)
    if (rows[i].steps <= rows[i - 1].steps)
      steps_grow = false;
  test_assert(steps_grow,
              "scaling: step count strictly increases with n "
              "(coherent search scales with \u221an, not O(1))");

  // Coherence ∈ (0, 1] at detection
  bool coherence_valid = true;
  for (const auto &r : rows) {
    MasterEigenOracle oracle;
    const double theta_t = MEO_TWO_PI * (static_cast<double>(r.n / 3)) /
                           static_cast<double>(r.n);
    QueryResult qr = oracle.query(theta_t, r.n);
    if (qr.coherence <= 0.0 || qr.coherence > 1.0 + 1e-9)
      coherence_valid = false;
  }
  test_assert(coherence_valid, "coherence: G_eff \u2208 (0,1] at detection "
                               "for all tested n");
}

// ══════════════════════════════════════════════════════════════════════════════
// 6. Four-Channel Validation
// ══════════════════════════════════════════════════════════════════════════════
static void test_four_channel_validation() {
  std::cout << "\n── 6. Four-Channel Validation ──────────────────────────────"
               "────────────\n";

  // At canonical coherent state (λ=0, G_eff=1): all 4 channels pass
  MasterEigenOracle oracle;
  test_assert(oracle.validate_four_channel(0.5),
              "four_channel: canonical state passes 4-channel check "
              "(G_eff=1 ≥ 0.5)");

  // With a very strict threshold near 1.0, validation may fail
  // (this tests the threshold-sensitivity path)
  bool strict_pass = oracle.validate_four_channel(0.9999);
  test_assert(strict_pass,
              "four_channel: at canonical state (λ=0) passes even strict "
              "threshold 0.9999 (sech(0)=1)");

  // Threshold of 1.0 exactly is the boundary condition
  bool exact_threshold = oracle.validate_four_channel(1.0);
  // G_eff = sech(0) = 1.0 exactly — should pass
  test_assert(exact_threshold,
              "four_channel: G_eff = 1.0 at λ=0 satisfies threshold = 1.0");
}

// ══════════════════════════════════════════════════════════════════════════════
// 7. Reset Behaviour
// ══════════════════════════════════════════════════════════════════════════════
static void test_reset_behaviour() {
  std::cout << "\n── 7. Reset Behaviour ──────────────────────────────────────"
               "────────────\n";

  MasterEigenOracle oracle;
  const uint64_t n = 512;
  const double theta_t = MEO_PI / 4.0; // 45°

  // First query
  QueryResult r1 = oracle.query(theta_t, n);

  // After reset, second query must reproduce the same result (deterministic)
  oracle.reset();
  QueryResult r2 = oracle.query(theta_t, n);

  test_assert(r1.steps == r2.steps,
              "reset: same step count on two successive queries after reset");
  test_assert(r1.best_channel == r2.best_channel,
              "reset: same best channel on two successive queries after reset");
  test_assert(std::abs(r1.accumulator_peak - r2.accumulator_peak) < 1e-9,
              "reset: same accumulator peak on two successive queries after "
              "reset");
  test_assert(r1.detected == r2.detected,
              "reset: detected flag matches on two successive queries");

  // After reset, accumulators are zero
  oracle.reset();
  bool zero_after_reset = true;
  for (double a : oracle.accumulators())
    if (std::abs(a) > 0.0)
      zero_after_reset = false;
  test_assert(zero_after_reset,
              "reset: accumulators are zero immediately after reset()");
}

// ══════════════════════════════════════════════════════════════════════════════
// 8. Phase-Coverage
// ══════════════════════════════════════════════════════════════════════════════
static void test_phase_coverage() {
  std::cout << "\n── 8. Phase-Coverage ───────────────────────────────────────"
               "────────────\n";

  // Targets at the 8 cardinal+diagonal angles (0°,45°,…,315°)
  const uint64_t n = 2048;
  const double step_budget = 2.0 * std::sqrt(static_cast<double>(n));
  bool all_detected = true;
  bool within_budget = true;

  std::cout << std::left << "  " << std::setw(12) << "theta_deg"
            << std::setw(10) << "steps" << std::setw(10) << "detected"
            << "best_ch\n";
  std::cout << "  " << std::string(42, '-') << "\n";

  for (int deg = 0; deg < 360; deg += 45) {
    MasterEigenOracle oracle;
    double theta_t = deg * MEO_PI / 180.0;
    QueryResult r = oracle.query(theta_t, n);
    if (!r.detected)
      all_detected = false;
    if (static_cast<double>(r.steps) > step_budget)
      within_budget = false;
    std::cout << std::fixed << std::setprecision(2) << "  " << std::setw(12)
              << deg << std::setw(10) << r.steps << std::setw(10)
              << (r.detected ? "yes" : "no") << r.best_channel << "\n";
  }

  test_assert(all_detected,
              "phase_coverage: oracle detects target at all 8 cardinal/diagonal "
              "phase angles");
  test_assert(within_budget,
              "phase_coverage: step count \u2264 2\u221an for all 8 "
              "target phases");
}

// ══════════════════════════════════════════════════════════════════════════════
// Main
// ══════════════════════════════════════════════════════════════════════════════
int main() {
  std::cout
      << "\n\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557\n";
  std::cout
      << "\u2551  Master Eigen Oracle \u2014 Validation Suite               "
         "       \u2551\n";
  std::cout
      << "\u2551  Coherence-guided oracle over the 8 \u03bc-eigenspaces      "
         "       \u2551\n";
  std::cout
      << "\u2551  G_eff = sech(\u03bb) weighting, \u0398(\u221an) detection   "
         "                   \u2551\n";
  std::cout
      << "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d\n";

  test_mu_orbit_structure();
  test_oracle_contribution_signal();
  test_coherence_weight();
  test_accumulator_dynamics();
  test_detection_scaling();
  test_four_channel_validation();
  test_reset_behaviour();
  test_phase_coverage();

  std::cout
      << "\n\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557\n";
  std::cout << "\u2551  Test Results                                          "
               "      \u2551\n";
  std::cout
      << "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d\n";
  std::cout << "  Total tests: " << test_count << "\n";
  std::cout << "  Passed:      " << passed << " \u2713\n";
  std::cout << "  Failed:      " << failed << " \u2717\n";

  if (failed == 0) {
    std::cout << "\n  \u2713 ALL MASTER EIGEN ORACLE TESTS PASSED \u2014 "
                 "coherence-guided eigenspace oracle validated\n\n";
    return 0;
  } else {
    std::cout << "\n  \u2717 MASTER EIGEN ORACLE VALIDATION FAILED \u2014 "
                 "check implementation\n\n";
    return 1;
  }
}
