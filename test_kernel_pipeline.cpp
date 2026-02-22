/*
 * test_kernel_pipeline.cpp — Unified Pipeline Test Suite
 *
 * Verifies the KernelState / SpectralBridge / KernelPipeline framework:
 *
 *   1. Core Invariants        — |β_phasor|=1, R(r)=0, R_eff=1 at r=1
 *   2. Phase Structure        — µ-rotation preserves r and C
 *   3. Palindrome Mode        — PalindromePrecession preserves |β|
 *   4. Spectral Bridge        — KernelState ↔ CoherentChannel round-trip
 *   5. Drift Detection        — has_drift() detects r ≠ 1
 *   6. Auto-Renormalization   — corrects drift toward r=1 and logs events
 *   7. Numerical Stability    — invariants hold over 1000 steps
 *   8. Pipeline API           — Pipeline::create / run / verify_invariants
 */

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

#include "KernelPipeline.hpp"

using namespace kernel::pipeline;

// ── Test infrastructure
// ───────────────────────────────────────────────────────
static int test_count = 0;
static int passed = 0;
static int failed = 0;

static void test_assert(bool condition, const std::string &test_name) {
  ++test_count;
  if (condition) {
    std::cout << "  \u2713 " << test_name << "\n";
    ++passed;
  } else {
    std::cout << "  \u2717 FAILED: " << test_name << "\n";
    ++failed;
  }
}

static constexpr double TIGHT_TOL = 1e-12;
static constexpr double FLOAT_TOL = 1e-9;

// ══════════════════════════════════════════════════════════════════════════════
// 1. Core Invariants
// ══════════════════════════════════════════════════════════════════════════════
static void test_core_invariants() {
  std::cout << "\n\u2554\u2550\u2550\u2550 1. Core Invariants "
               "\u2550\u2550\u2550\u2557\n";

  KernelState ks;

  // Invariant 1: unit-circle normalization |α|²+|β|²=1
  test_assert(ks.beta_unit_invariant(),
              "initial state: |α|²+|β|²=1 (unit-circle)");

  // Invariant 2: R(r)=0 at canonical coherent state
  test_assert(ks.palindrome_zero(),
              "initial state: R(r)=0 (palindrome residual)");

  // Invariant 3: R_eff=1 at r=1 (Ohm–Coherence Duality)
  test_assert(ks.r_eff_unity(),
              "initial state: R_eff=1 (ideal Ohm–Coherence channel)");

  // All three simultaneously
  test_assert(ks.all_invariants(),
              "initial state: all three invariants hold simultaneously");

  // Verify derived quantities at coherent fixed point
  test_assert(std::abs(ks.radius() - 1.0) < TIGHT_TOL,
              "r = |β/α| = 1 at coherent fixed point");
  test_assert(std::abs(ks.coherence() - 1.0) < TIGHT_TOL,
              "C = 2|α||β| = 1 at coherent fixed point");
  test_assert(std::abs(ks.lyapunov()) < TIGHT_TOL,
              "\u03bb = ln(1) = 0 at coherent fixed point");

  // Invariants broken at r \u2260 1 (Corollary 13)
  KernelState decoherent;
  decoherent.beta *= 1.5; // r > 1
  decoherent.normalize(); // keep unit normalization
  test_assert(!decoherent.palindrome_zero(),
              "decoherent state (r>1): R(r)\u22600");
  test_assert(!decoherent.r_eff_unity(),
              "decoherent state (r>1): R_eff\u22601");
}

// ══════════════════════════════════════════════════════════════════════════════
// 2. Phase Structure — µ-rotation preserves r and C
// ══════════════════════════════════════════════════════════════════════════════
static void test_phase_structure() {
  std::cout << "\n\u2554\u2550\u2550\u2550 2. Phase Structure "
               "\u2550\u2550\u2550\u2557\n";

  KernelState ks;
  double r_init = ks.radius();
  double C_init = ks.coherence();

  // Apply 8 µ-rotation steps (one full 8-cycle)
  for (int i = 0; i < 8; ++i)
    ks.step();

  // r is invariant under µ-rotation (pure phase change on β)
  test_assert(std::abs(ks.radius() - r_init) < FLOAT_TOL,
              "\u00b5-rotation: r preserved over 8-cycle");

  // C is invariant (depends only on |α|, |β|)
  test_assert(std::abs(ks.coherence() - C_init) < FLOAT_TOL,
              "\u00b5-rotation: C preserved over 8-cycle");

  // Palindrome residual R(r) is preserved
  test_assert(std::abs(ks.palindrome_residual()) < FLOAT_TOL,
              "\u00b5-rotation: R(r)=0 preserved over 8-cycle");

  // Normalization preserved
  test_assert(ks.beta_unit_invariant(),
              "\u00b5-rotation: normalization preserved over 8-cycle");

  // tick counter advances correctly
  test_assert(ks.tick == 8, "tick counter = 8 after 8 steps");

  // Decoherent state: µ-rotation still preserves r (even when r \u2260 1)
  KernelState dec;
  dec.beta *= 1.3;
  dec.normalize();
  double r_dec_init = dec.radius();
  for (int i = 0; i < 8; ++i)
    dec.step();
  test_assert(std::abs(dec.radius() - r_dec_init) < FLOAT_TOL,
              "\u00b5-rotation: r preserved even for decoherent state");
}

// ══════════════════════════════════════════════════════════════════════════════
// 3. Palindrome Mode — PalindromePrecession preserves |β|
// ══════════════════════════════════════════════════════════════════════════════
static void test_palindrome_mode() {
  std::cout << "\n\u2554\u2550\u2550\u2550 3. Palindrome Mode "
               "\u2550\u2550\u2550\u2557\n";

  // Pipeline in PALINDROME mode
  Pipeline pl = Pipeline::create(KernelMode::PALINDROME);
  double r_init = pl.state().radius();
  double beta_mag_init = std::abs(pl.state().beta);

  pl.run(100);

  // r = 1 preserved (|β|/|α| unchanged by pure phase precession)
  test_assert(std::abs(pl.state().radius() - r_init) < FLOAT_TOL,
              "PALINDROME mode: r preserved after 100 steps");

  // |β| preserved
  test_assert(std::abs(std::abs(pl.state().beta) - beta_mag_init) < FLOAT_TOL,
              "PALINDROME mode: |beta| preserved after 100 steps");

  // All three invariants still hold
  test_assert(pl.verify_invariants(),
              "PALINDROME mode: all invariants hold after 100 steps");

  // Spectral verification: G_eff = 1, R_eff = 1
  test_assert(pl.verify_spectral(),
              "PALINDROME mode: spectral bridge G_eff=R_eff=1 after 100 steps");
}

// ══════════════════════════════════════════════════════════════════════════════
// 4. Spectral Bridge — KernelState ↔ CoherentChannel round-trip
// ══════════════════════════════════════════════════════════════════════════════
static void test_spectral_bridge() {
  std::cout << "\n\u2554\u2550\u2550\u2550 4. Spectral Bridge "
               "\u2550\u2550\u2550\u2557\n";

  // Ideal channel: r=1 → λ=0 → G_eff=1, R_eff=1
  KernelState ks_ideal;
  auto ch_ideal = SpectralBridge::to_channel(ks_ideal);
  test_assert(std::abs(ch_ideal.lambda) < TIGHT_TOL,
              "ideal state: \u03bb = 0 in CoherentChannel");
  test_assert(std::abs(ch_ideal.G_eff() - 1.0) < TIGHT_TOL,
              "ideal state: G_eff = 1");
  test_assert(std::abs(ch_ideal.R_eff() - 1.0) < TIGHT_TOL,
              "ideal state: R_eff = 1");

  // Round-trip: KernelState → channel → KernelState preserves r
  for (double r_test : {0.5, 1.0, 1.5, 2.0}) {
    KernelState ks_in;
    // Perturb β to achieve the desired r while keeping unit norm
    double a = 1.0 / std::sqrt(1.0 + r_test * r_test);
    double b = r_test * a;
    ks_in.alpha = Cx{a, 0.0};
    static const Cx CANONICAL_BETA_PHASE{-KS_ETA,
                                         KS_ETA}; // e^{i3\u03c0/4}/\u221a2
    ks_in.beta = CANONICAL_BETA_PHASE * (b / std::abs(CANONICAL_BETA_PHASE));

    auto ch = SpectralBridge::to_channel(ks_in);
    auto ks_out = SpectralBridge::from_channel(ch);

    test_assert(std::abs(ks_out.radius() - r_test) < FLOAT_TOL,
                "round-trip r=" + std::to_string(r_test) + " preserved");
  }

  // G_eff * R_eff = 1 (duality identity, Theorem 14) for any KernelState
  KernelState ks_any;
  ks_any.beta *= 1.4;
  ks_any.normalize();
  auto ch_any = SpectralBridge::to_channel(ks_any);
  test_assert(std::abs(ch_any.G_eff() * ch_any.R_eff() - 1.0) < TIGHT_TOL,
              "G_eff * R_eff = 1 (duality identity)");
}

// ══════════════════════════════════════════════════════════════════════════════
// 5. Drift Detection
// ══════════════════════════════════════════════════════════════════════════════
static void test_drift_detection() {
  std::cout << "\n\u2554\u2550\u2550\u2550 5. Drift Detection "
               "\u2550\u2550\u2550\u2557\n";

  // No drift at coherent fixed point
  KernelState ks_ideal;
  test_assert(!ks_ideal.has_drift(), "coherent state: no drift detected");

  // Drift detected when |R(r)| > tol
  KernelState ks_drift;
  ks_drift.beta *= 1.2;
  ks_drift.normalize();
  test_assert(ks_drift.has_drift(), "r>1 state: drift detected");

  KernelState ks_drift2;
  ks_drift2.beta *= 0.7;
  ks_drift2.normalize();
  test_assert(ks_drift2.has_drift(), "r<1 state: drift detected");

  // No spurious drift for µ-rotated states (pure phase, r invariant)
  KernelState ks_rotated;
  for (int i = 0; i < 32; ++i)
    ks_rotated.step();
  test_assert(!ks_rotated.has_drift(),
              "µ-rotated state (32 steps): no spurious drift");
}

// ══════════════════════════════════════════════════════════════════════════════
// 6. Auto-Renormalization
// ══════════════════════════════════════════════════════════════════════════════
static void test_auto_renormalization() {
  std::cout << "\n\u2554\u2550\u2550\u2550 6. Auto-Renormalization "
               "\u2550\u2550\u2550\u2557\n";

  // No renormalization needed at r=1
  KernelState ks_ideal;
  bool applied = ks_ideal.auto_renormalize();
  test_assert(!applied, "coherent state: auto_renormalize() returns false");
  test_assert(ks_ideal.renorm_log.empty(),
              "coherent state: renorm_log is empty");

  // Renormalization corrects r toward 1
  KernelState ks_drift;
  ks_drift.beta *= 1.5;
  ks_drift.normalize();
  double r_before = ks_drift.radius();
  applied = ks_drift.auto_renormalize();
  double r_after = ks_drift.radius();

  test_assert(applied, "drifted state: auto_renormalize() returns true");
  test_assert(r_after < r_before,
              "after renorm: r closer to 1 (r decreased toward 1)");
  test_assert(ks_drift.beta_unit_invariant(),
              "after renorm: normalization preserved");

  // Renormalization logged
  test_assert(ks_drift.renorm_log.size() == 1,
              "renorm_log has exactly 1 entry after one renormalization");
  const auto &ev = ks_drift.renorm_log[0];
  test_assert(std::abs(ev.r_before - r_before) < FLOAT_TOL,
              "log entry: r_before matches r before correction");
  test_assert(std::abs(ev.r_after - r_after) < FLOAT_TOL,
              "log entry: r_after matches r after correction");
  test_assert(std::abs(ev.R_after) < std::abs(ev.R_before),
              "log entry: |R_after| < |R_before| (residual reduced)");

  // Multiple renormalizations converge toward r=1
  KernelState ks_conv;
  ks_conv.beta *= 3.0;
  ks_conv.normalize();
  for (int i = 0; i < 20; ++i)
    ks_conv.auto_renormalize();
  test_assert(std::abs(ks_conv.radius() - 1.0) < 0.05,
              "20 renormalization steps: r converges toward 1");

  // Renormalization in SPECTRAL mode via Pipeline
  Pipeline pl = Pipeline::create(KernelMode::SPECTRAL);
  pl.run(8);
  // Spectral mode maintains invariants
  test_assert(pl.verify_invariants(),
              "SPECTRAL mode: invariants maintained via auto-renorm");
}

// ══════════════════════════════════════════════════════════════════════════════
// 7. Numerical Stability — invariants hold over many steps
// ══════════════════════════════════════════════════════════════════════════════
static void test_numerical_stability() {
  std::cout << "\n\u2554\u2550\u2550\u2550 7. Numerical Stability "
               "\u2550\u2550\u2550\u2557\n";

  // STANDARD mode: 1000 µ-rotation steps — normalization must not drift
  {
    Pipeline pl = Pipeline::create(KernelMode::STANDARD);
    pl.run(1000);
    double norm_sq = std::norm(pl.state().alpha) + std::norm(pl.state().beta);
    test_assert(std::abs(norm_sq - 1.0) < FLOAT_TOL,
                "STANDARD mode: |α|²+|β|²=1 after 1000 steps");
    test_assert(std::abs(pl.state().radius() - 1.0) < FLOAT_TOL,
                "STANDARD mode: r=1 after 1000 steps");
  }

  // PALINDROME mode: 1000 steps — |β| and r stable
  {
    Pipeline pl = Pipeline::create(KernelMode::PALINDROME);
    pl.run(1000);
    test_assert(pl.verify_invariants(),
                "PALINDROME mode: all invariants after 1000 steps");
  }

  // FULL mode: 1000 steps — all invariants, no spurious renorm on clean state
  {
    Pipeline pl = Pipeline::create(KernelMode::FULL);
    pl.run(1000);
    test_assert(pl.verify_invariants(),
                "FULL mode: all invariants after 1000 steps");
    // No renorm events expected since canonical state never drifts
    test_assert(pl.renorm_log().empty(),
                "FULL mode: no renorm events on canonical state (1000 steps)");
  }

  // Spectral stability: G_eff and R_eff stay at 1.0 throughout FULL mode
  {
    Pipeline pl = Pipeline::create(KernelMode::FULL);
    bool spectral_stable = true;
    for (int i = 0; i < 100; ++i) {
      pl.tick();
      auto ch = pl.channel();
      if (std::abs(ch.G_eff() - 1.0) > FLOAT_TOL ||
          std::abs(ch.R_eff() - 1.0) > FLOAT_TOL) {
        spectral_stable = false;
        break;
      }
    }
    test_assert(spectral_stable,
                "FULL mode: G_eff=R_eff=1 stable over 100 ticks");
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// 8. Pipeline API
// ══════════════════════════════════════════════════════════════════════════════
static void test_pipeline_api() {
  std::cout << "\n\u2554\u2550\u2550\u2550 8. Pipeline API "
               "\u2550\u2550\u2550\u2557\n";

  // create() produces a valid pipeline
  Pipeline pl = Pipeline::create(KernelMode::FULL);
  test_assert(pl.verify_invariants(),
              "Pipeline::create(FULL): invariants hold on fresh pipeline");

  // with_mode() changes the mode
  pl.with_mode(KernelMode::PALINDROME);
  test_assert(pl.mode() == KernelMode::PALINDROME,
              "with_mode(PALINDROME): mode updated");

  // with_state() replaces the state
  KernelState custom;
  custom.beta *= 1.1;
  custom.normalize();
  pl.with_state(custom);
  test_assert(std::abs(pl.state().radius() - custom.radius()) < FLOAT_TOL,
              "with_state(): custom state installed");

  // run() advances ticks
  pl.with_mode(KernelMode::FULL).with_state(KernelState{});
  pl.run(24);
  test_assert(pl.state().tick == 24, "run(24): tick counter = 24");

  // reset() restores canonical state
  pl.reset();
  test_assert(pl.verify_invariants(), "reset(): invariants hold after reset");
  test_assert(pl.state().tick == 0, "reset(): tick = 0");
  test_assert(pl.renorm_log().empty(), "reset(): renorm_log cleared");

  // channel() returns the spectral channel
  auto ch = pl.channel();
  test_assert(std::abs(ch.lambda) < FLOAT_TOL,
              "channel(): lambda = 0 for reset pipeline");

  // SPECTRAL mode corrects injected drift
  Pipeline spectral = Pipeline::create(KernelMode::SPECTRAL);
  KernelState drifted;
  drifted.beta *= 2.0;
  drifted.normalize();
  spectral.with_state(drifted);
  spectral.run(50);
  // After 50 spectral steps the state should be closer to r=1 than initial
  double r_final = spectral.state().radius();
  test_assert(std::abs(r_final - 1.0) < std::abs(drifted.radius() - 1.0),
              "SPECTRAL mode: r converges toward 1 from drifted initial state");
}

// ══════════════════════════════════════════════════════════════════════════════
// Main test runner
// ══════════════════════════════════════════════════════════════════════════════
int main() {
  std::cout << "\n\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2557\n";
  std::cout << "\u2551  Kernel Pipeline \u2014 Unified Framework Test Suite    "
               "          \u2551\n";
  std::cout << "\u2551  KernelState / SpectralBridge / KernelPipeline          "
               "      \u2551\n";
  std::cout << "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u255d\n";

  test_core_invariants();
  test_phase_structure();
  test_palindrome_mode();
  test_spectral_bridge();
  test_drift_detection();
  test_auto_renormalization();
  test_numerical_stability();
  test_pipeline_api();

  std::cout << "\n\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2557\n";
  std::cout << "\u2551  Test Results                                          "
               "      \u2551\n";
  std::cout << "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
               "\u2550\u2550\u2550\u255d\n";
  std::cout << "  Total tests: " << test_count << "\n";
  std::cout << "  Passed:      " << passed << " \u2713\n";
  std::cout << "  Failed:      " << failed << " \u2717\n";

  if (failed == 0) {
    std::cout << "\n  \u2713 ALL PIPELINE INVARIANTS VERIFIED \u2014 "
                 "KernelState / SpectralBridge / KernelPipeline coherent "
                 "end-to-end\n\n";
    return 0;
  } else {
    std::cout << "\n  \u2717 PIPELINE VERIFICATION FAILED \u2014 "
                 "check implementation\n\n";
    return 1;
  }
}
