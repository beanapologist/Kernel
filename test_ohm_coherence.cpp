/*
 * test_ohm_coherence.cpp — Test Suite for Ohm–Coherence Duality
 *
 * Validates all components of the ohm_coherence_duality.hpp framework:
 *
 *   1. Core duality functions: conductance, resistance, lyapunov_from_coherence
 *   2. CoherentChannel: G_eff, R_eff, coherence
 *   3. MultiChannelSystem: parallel composition, 4-channel redundancy
 *   4. PipelineSystem: series composition, R_tot = Σ R_stage, bottleneck
 *   5. FourChannelModel: eigenvalue structure, error-tolerance validation
 *   6. OUProcess: noise simulation, average conductance, Jensen's inequality
 *   7. QuTritDegradation: qutrit coherence patterns
 *
 * Test style follows test_pipeline_theorems.cpp and test_qudit_kernel.cpp.
 */

#include "ohm_coherence_duality.hpp"

#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace kernel::ohm;

// ── Test framework (matches existing test files)
// ──────────────────────────────
static int test_count = 0;
static int passed = 0;
static int failed = 0;

static void test_assert(bool condition, const std::string &name) {
  ++test_count;
  if (condition) {
    std::cout << "  \u2713 " << name << "\n";
    ++passed;
  } else {
    std::cout << "  \u2717 FAILED: " << name << "\n";
    ++failed;
  }
}

static constexpr double TIGHT_TOL = 1e-12;
static constexpr double FLOAT_TOL = 1e-9;

// ── 1. Core duality functions
// ─────────────────────────────────────────────────
void test_core_functions() {
  std::cout << "\n\u2554\u2550\u2550\u2550 Core Duality: C = sech(\u03bb) = "
               "G_eff = 1 / R_eff \u2550\u2550\u2550\u2557\n";

  // At λ = 0: G_eff = 1, R_eff = 1 (ideal channel)
  test_assert(std::abs(conductance(0.0) - 1.0) < TIGHT_TOL,
              "conductance(0) = 1  (ideal channel)");
  test_assert(std::abs(resistance(0.0) - 1.0) < TIGHT_TOL,
              "resistance(0) = 1  (ideal channel)");

  // G_eff * R_eff = 1 for all λ (duality identity)
  for (double lam : {0.0, 0.5, 1.0, 2.0, 3.0}) {
    double product = conductance(lam) * resistance(lam);
    test_assert(std::abs(product - 1.0) < TIGHT_TOL,
                "G_eff * R_eff = 1 for \u03bb=" + std::to_string(lam));
  }

  // conductance is strictly decreasing for λ > 0
  bool decreasing = true;
  double prev = conductance(0.0);
  for (double lam = 0.1; lam <= 3.0; lam += 0.1) {
    double cur = conductance(lam);
    if (cur >= prev)
      decreasing = false;
    prev = cur;
  }
  test_assert(decreasing, "G_eff strictly decreasing in \u03bb");

  // resistance is strictly increasing for λ > 0
  bool increasing = true;
  double prev_r = resistance(0.0);
  for (double lam = 0.1; lam <= 3.0; lam += 0.1) {
    double cur_r = resistance(lam);
    if (cur_r <= prev_r)
      increasing = false;
    prev_r = cur_r;
  }
  test_assert(increasing, "R_eff strictly increasing in \u03bb");

  // Even symmetry: conductance(λ) = conductance(-λ) (sech is even)
  for (double lam : {0.5, 1.0, 2.0}) {
    test_assert(std::abs(conductance(lam) - conductance(-lam)) < TIGHT_TOL,
                "conductance even symmetry for \u03bb=" + std::to_string(lam));
  }

  // lyapunov_from_coherence is the inverse of conductance
  for (double lam : {0.0, 0.3, 0.7, 1.0, 2.0}) {
    double C = conductance(lam);
    double lam_back = lyapunov_from_coherence(C);
    test_assert(std::abs(lam_back - lam) < FLOAT_TOL,
                "lyapunov_from_coherence inverts conductance at \u03bb=" +
                    std::to_string(lam));
  }

  // lyapunov_from_coherence(1) = 0 (ideal coherence → zero degradation)
  test_assert(std::abs(lyapunov_from_coherence(1.0)) < TIGHT_TOL,
              "lyapunov_from_coherence(1) = 0");
}

// ── 2. CoherentChannel
// ────────────────────────────────────────────────────────
void test_coherent_channel() {
  std::cout << "\n\u2554\u2550\u2550\u2550 CoherentChannel \u2550\u2550\u2550"
               "\u2557\n";

  CoherentChannel ideal(0.0);
  test_assert(std::abs(ideal.G_eff() - 1.0) < TIGHT_TOL,
              "ideal channel: G_eff = 1");
  test_assert(std::abs(ideal.R_eff() - 1.0) < TIGHT_TOL,
              "ideal channel: R_eff = 1");
  test_assert(std::abs(ideal.coherence() - 1.0) < TIGHT_TOL,
              "ideal channel: coherence = 1");

  // Duality: G_eff = coherence for every channel
  for (double lam : {0.0, 0.5, 1.0, 2.0}) {
    CoherentChannel ch(lam);
    test_assert(std::abs(ch.G_eff() - ch.coherence()) < TIGHT_TOL,
                "G_eff == coherence (duality) for \u03bb=" +
                    std::to_string(lam));
  }

  // G_eff * R_eff = 1
  for (double lam : {0.1, 0.5, 1.5, 3.0}) {
    CoherentChannel ch(lam);
    test_assert(std::abs(ch.G_eff() * ch.R_eff() - 1.0) < TIGHT_TOL,
                "G_eff * R_eff = 1 for \u03bb=" + std::to_string(lam));
  }
}

// ── 3. MultiChannelSystem
// ─────────────────────────────────────────────────────
void test_multi_channel() {
  std::cout
      << "\n\u2554\u2550\u2550\u2550 MultiChannelSystem (parallel) \u2550\u2550"
         "\u2550\u2557\n";

  // Single ideal channel: G_tot = 1, R_tot = 1
  MultiChannelSystem single(1, 0.0);
  test_assert(std::abs(single.G_total() - 1.0) < TIGHT_TOL,
              "single ideal channel: G_tot = 1");
  test_assert(std::abs(single.R_total() - 1.0) < TIGHT_TOL,
              "single ideal channel: R_tot = 1");

  // 4 ideal channels: G_tot = 4, R_tot = 1/4
  MultiChannelSystem four_ideal(4, 0.0);
  test_assert(std::abs(four_ideal.G_total() - 4.0) < TIGHT_TOL,
              "4 ideal channels: G_tot = 4");
  test_assert(std::abs(four_ideal.R_total() - 0.25) < TIGHT_TOL,
              "4 ideal channels: R_tot = 0.25");

  // Homogeneous N channels: G_tot = N * sech(λ)
  double lam = 0.5;
  int N = 4;
  MultiChannelSystem hom(N, lam);
  double expected_G = N * conductance(lam);
  test_assert(std::abs(hom.G_total() - expected_G) < TIGHT_TOL,
              "homogeneous 4-channel: G_tot = 4 * sech(\u03bb)");

  // Heterogeneous channels: G_tot = Σ sech(λ_i)
  std::vector<double> lams = {0.0, 0.5, 1.0, 2.0};
  MultiChannelSystem hetero(lams);
  double expected_hetero_G = 0.0;
  for (double l : lams)
    expected_hetero_G += conductance(l);
  test_assert(std::abs(hetero.G_total() - expected_hetero_G) < TIGHT_TOL,
              "heterogeneous channels: G_tot = \u03a3 sech(\u03bb_i)");

  // Weakest channel is the one with highest λ (lowest G_eff)
  test_assert(hetero.weakest_channel() == 3,
              "weakest channel has highest \u03bb (index 3)");

  // Adding more parallel channels always increases G_tot
  MultiChannelSystem two(2, 0.5);
  MultiChannelSystem three(3, 0.5);
  test_assert(three.G_total() > two.G_total(),
              "more parallel channels → higher G_tot");
}

// ── 4. PipelineSystem
// ─────────────────────────────────────────────────────────
void test_pipeline_system() {
  std::cout
      << "\n\u2554\u2550\u2550\u2550 PipelineSystem (series) \u2550\u2550\u2550"
         "\u2557\n";

  // Single ideal stage: R_tot = cosh(0) = 1
  PipelineSystem single({0.0});
  test_assert(std::abs(single.R_total() - 1.0) < TIGHT_TOL,
              "single ideal stage: R_tot = 1");

  // Two identical stages at λ = 0: R_tot = 2
  PipelineSystem two({0.0, 0.0});
  test_assert(std::abs(two.R_total() - 2.0) < TIGHT_TOL,
              "two ideal stages: R_tot = 2");

  // R_tot = Σ cosh(λ_i) (series addition)
  std::vector<double> lams = {0.0, 0.5, 1.0};
  PipelineSystem three(lams);
  double expected_R = 0.0;
  for (double l : lams)
    expected_R += resistance(l);
  test_assert(std::abs(three.R_total() - expected_R) < TIGHT_TOL,
              "R_tot = \u03a3 cosh(\u03bb_i) (series addition)");

  // G_tot = 1 / R_tot
  test_assert(std::abs(three.G_total() - 1.0 / expected_R) < TIGHT_TOL,
              "G_tot = 1 / R_tot");

  // Bottleneck is the stage with the largest λ (highest R_eff)
  std::vector<double> lams2 = {0.2, 2.0, 0.5, 0.1};
  PipelineSystem pipe4(lams2);
  test_assert(pipe4.bottleneck_stage() == 1,
              "bottleneck is stage with highest \u03bb (index 1)");

  // Adding a stage always increases R_tot
  PipelineSystem p3({0.3, 0.5, 0.7});
  PipelineSystem p4({0.3, 0.5, 0.7, 0.4});
  test_assert(p4.R_total() > p3.R_total(),
              "adding a series stage increases R_tot");
}

// ── 5. FourChannelModel
// ───────────────────────────────────────────────────────
void test_four_channel_model() {
  std::cout << "\n\u2554\u2550\u2550\u2550 FourChannelModel (4-eigenvalue "
               "structure) \u2550\u2550\u2550\u2557\n";

  // All ideal: all G_eff = 1, error tolerance passes
  FourChannelModel all_ideal(0.0, 0.0, 0.0, 0.0);
  double eigs[4];
  all_ideal.eigenvalues(eigs);
  bool all_one = true;
  for (int i = 0; i < 4; ++i)
    if (std::abs(eigs[i] - 1.0) > TIGHT_TOL)
      all_one = false;
  test_assert(all_one, "all ideal channels: eigenvalues = 1");
  test_assert(all_ideal.validate_error_tolerance(),
              "all ideal channels: error tolerance passes");

  // One degraded channel (λ=2): 3 coherent → still tolerant
  FourChannelModel one_bad(0.0, 0.0, 0.0, 2.0);
  test_assert(one_bad.validate_error_tolerance(),
              "one degraded channel: error tolerance still passes (3/4 ok)");
  test_assert(one_bad.weakest_channel() == 3,
              "one degraded channel: weakest is channel 3");

  // Two degraded channels: 2 coherent → tolerance fails
  FourChannelModel two_bad(0.0, 0.0, 2.0, 2.0);
  test_assert(!two_bad.validate_error_tolerance(),
              "two degraded channels: error tolerance fails (2/4 ok)");

  // All degraded moderately: threshold=0.5 check
  // sech(0.5) ≈ 0.8868 > 0.5 → all coherent
  FourChannelModel moderate(0.5, 0.5, 0.5, 0.5);
  test_assert(moderate.validate_error_tolerance(0.5),
              "moderate degradation (\u03bb=0.5): threshold 0.5 passes");

  // Eigenvalues strictly decrease with λ
  FourChannelModel mixed(0.0, 0.5, 1.0, 1.5);
  mixed.eigenvalues(eigs);
  bool strictly_dec = true;
  for (int i = 1; i < 4; ++i)
    if (eigs[i] >= eigs[i - 1])
      strictly_dec = false;
  test_assert(strictly_dec,
              "eigenvalues strictly decrease with increasing \u03bb");
}

// ── 6. OUProcess & Jensen's inequality ───────────────────────────────────────
void test_ou_process() {
  std::cout << "\n\u2554\u2550\u2550\u2550 OUProcess & Jensen's Inequality "
               "\u2550\u2550\u2550\u2557\n";

  // Noiseless OU (σ=0): λ converges to μ
  OUProcess noiseless(1.0, 0.5, 0.0);
  auto path = noiseless.simulate(2.0, 500, 0.01);
  double final_lam = path.back();
  test_assert(std::abs(final_lam - 0.5) < 0.01,
              "noiseless OU converges to \u03bc = 0.5");

  // Path length is steps+1
  test_assert(static_cast<int>(path.size()) == 501, "path length = steps+1");

  // average_conductance is in (0, 1]
  double avg_G = OUProcess::average_conductance(path);
  test_assert(avg_G > 0.0 && avg_G <= 1.0 + FLOAT_TOL,
              "average conductance \u27e8G\u27e9 \u2208 (0, 1]");

  // Jensen's inequality: ⟨G⟩ ≤ sech(⟨λ⟩) when sech is locally concave
  // (holds for small noise / small |λ|; we test with μ ≈ 0 for concavity)
  // With noisy OU around μ = 0, sech is concave near 0 → Jensen applies.
  OUProcess noisy(2.0, 0.0, 0.3);
  auto noisy_path = noisy.simulate(0.0, 10000, 0.005, 12345);
  double mean_lam = 0.0;
  for (double l : noisy_path)
    mean_lam += l;
  mean_lam /= static_cast<double>(noisy_path.size());
  double avg_G_noisy = OUProcess::average_conductance(noisy_path);
  double G_at_mean = conductance(mean_lam);
  // Jensen: ⟨sech(λ)⟩ ≤ sech(⟨λ⟩)  (sech concave near λ=0)
  test_assert(avg_G_noisy <= G_at_mean + 1e-3,
              "Jensen's inequality: \u27e8G\u27e9 \u2264 sech(\u27e8\u03bb"
              "\u27e9) for noise around \u03bc=0");

  // Noisy path: mean stays near μ (ergodicity of OU process)
  test_assert(std::abs(mean_lam) < 0.05,
              "noisy OU around \u03bc=0: time-average near 0");

  // Higher noise → lower average conductance (more degradation)
  OUProcess high_noise(2.0, 0.0, 1.0);
  OUProcess low_noise(2.0, 0.0, 0.1);
  double avg_high = OUProcess::average_conductance(
      high_noise.simulate(0.0, 10000, 0.005, 99));
  double avg_low =
      OUProcess::average_conductance(low_noise.simulate(0.0, 10000, 0.005, 99));
  test_assert(avg_high < avg_low,
              "higher noise \u03c3 \u2192 lower \u27e8G\u27e9");
}

// ── 7. QuTritDegradation
// ──────────────────────────────────────────────────────
void test_qutrit_degradation() {
  std::cout
      << "\n\u2554\u2550\u2550\u2550 QuTritDegradation \u2550\u2550\u2550\u2557"
         "\n";

  // Ideal qutrit: all λ = 0 → coherence_avg = 1, coherence_min = 1
  QuTritDegradation ideal(0.0, 0.0, 0.0);
  test_assert(std::abs(ideal.coherence_avg() - 1.0) < TIGHT_TOL,
              "ideal qutrit: coherence_avg = 1");
  test_assert(std::abs(ideal.coherence_min() - 1.0) < TIGHT_TOL,
              "ideal qutrit: coherence_min = 1");

  // coherence_avg = mean of sech(λ_01), sech(λ_02), sech(λ_12)
  QuTritDegradation q(0.5, 1.0, 1.5);
  double expected_avg =
      (conductance(0.5) + conductance(1.0) + conductance(1.5)) / 3.0;
  test_assert(std::abs(q.coherence_avg() - expected_avg) < TIGHT_TOL,
              "coherence_avg = mean of channel conductances");

  // coherence_min is the minimum of the three channels
  double expected_min =
      std::min(conductance(0.5), std::min(conductance(1.0), conductance(1.5)));
  test_assert(std::abs(q.coherence_min() - expected_min) < TIGHT_TOL,
              "coherence_min = minimum channel conductance");

  // coherence_min ≤ coherence_avg (min ≤ mean)
  test_assert(q.coherence_min() <= q.coherence_avg() + TIGHT_TOL,
              "coherence_min \u2264 coherence_avg");

  // Increasing any λ reduces coherence_avg
  QuTritDegradation q_more(0.5, 1.0, 2.0); // λ_12 increased
  test_assert(q_more.coherence_avg() < q.coherence_avg(),
              "increasing \u03bb reduces coherence_avg");

  // Asymmetric degradation: coherence_min < coherence_avg when λ differ
  QuTritDegradation asym(0.0, 0.0, 3.0);
  test_assert(asym.coherence_min() < asym.coherence_avg(),
              "asymmetric degradation: coherence_min < coherence_avg");
}

// ── Main
// ──────────────────────────────────────────────────────────────────────
int main() {
  std::cout
      << "\n\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2557\n";
  std::cout << "\u2551  Ohm\u2013Coherence Duality \u2014 Test Suite          "
               "              \u2551\n";
  std::cout << "\u2551  C = sech(\u03bb) = G_eff = 1 / R_eff  (Theorem 14)   "
               "          \u2551\n";
  std::cout
      << "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u255d\n";

  test_core_functions();
  test_coherent_channel();
  test_multi_channel();
  test_pipeline_system();
  test_four_channel_model();
  test_ou_process();
  test_qutrit_degradation();

  std::cout
      << "\n\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2557\n";
  std::cout << "\u2551  Test Results                                          "
               "      \u2551\n";
  std::cout << "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u255d\n";
  std::cout << "  Total tests: " << test_count << "\n";
  std::cout << "  Passed:      " << passed << " \u2713\n";
  std::cout << "  Failed:      " << failed << " \u2717\n";

  if (failed == 0) {
    std::cout
        << "\n  \u2713 ALL TESTS PASSED \u2014 Ohm\u2013Coherence Duality "
           "verified\n\n";
    return 0;
  } else {
    std::cout << "\n  \u2717 TESTS FAILED \u2014 Check implementation\n\n";
    return 1;
  }
}
