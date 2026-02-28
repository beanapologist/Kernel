/*
 * test_battery_analogy.cpp — Empirical Proof: Phase Battery Analogy
 *
 * Proves that the KernelSync phase-synchronisation engine satisfies all three
 * structural essentials of an energy-conversion device (battery):
 *
 *   1. SOURCE  — frustrated nodes carrying phase deviation δθ_j > 0
 *                that give up their frustration over time.
 *
 *   2. SINK    — mean attractor ψ̄ that is conserved while receiving the
 *                released frustration and converting it to circular coherence
 * R.
 *
 *   3. MEDIUM  — G_eff = sech(λ) damping that controls the transfer rate:
 *                  g = 0     → open circuit  (no transfer, battery dead)
 *                  g ∈ (0,1) → controlled transfer (useful work / charging)
 *                  g > 1     → over-shoot / oscillation (short circuit)
 *
 * Test structure:
 *   1. source_exists       — frustration E > 0 for non-uniform initial phases.
 *   2. sink_is_conserved   — mean phase ψ̄ is invariant under EMA updates.
 *   3. medium_controls     — no transfer at g=0; monotone decrease at g∈(0,1).
 *   4. three_essentials    — battery only works when all three are present.
 *   5. coherence_monotone  — circular coherence R(t) is non-decreasing for
 *                            g ∈ (0, 1] (dissipative contraction toward sink).
 *   6. g_eff_rate          — higher G_eff (lower λ) → faster convergence.
 *
 * The PhaseBattery model lives in ohm_coherence_duality.hpp.
 * Test style matches test_ohm_coherence.cpp.
 */

#include "ohm_coherence_duality.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace kernel::ohm;

// ── Test framework
// ────────────────────────────────────────────────────────────
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

static constexpr double TOL = 1e-9;
static constexpr double LOOSE_TOL = 1e-6;

// ── Helpers
// ───────────────────────────────────────────────────────────────────

// Wrap angle to (−π, π]
static double wrap(double a) {
  return a - 2.0 * OHM_PI * std::floor((a + OHM_PI) / (2.0 * OHM_PI));
}

// Build N phases uniformly spaced over [−spread, +spread] around centre.
static std::vector<double> uniform_phases(int N, double centre, double spread) {
  std::vector<double> ph(N);
  for (int j = 0; j < N; ++j)
    ph[j] = centre + spread * (2.0 * j / (N - 1.0) - 1.0);
  return ph;
}

// ── 1. Source exists
// ────────────────────────────────────────────────────────── Verify that a
// system with non-uniform initial phases has positive frustration (chemical
// potential / source energy), and that a perfectly uniform system has zero
// frustration (no energy to convert — dead battery).
void test_source_exists() {
  std::cout << "\n\u2554\u2550\u2550\u2550 1. Source: Phase Frustration "
               "\u2550\u2550\u2550\u2557\n";

  // N nodes spread over ±π/2 rad around 0 — high frustration
  const int N = 16;
  auto ph_spread = uniform_phases(N, 0.0, OHM_PI / 2.0);
  PhaseBattery bat_spread(N, 0.3, ph_spread);
  double E_spread = bat_spread.frustration();
  test_assert(E_spread > 0.1,
              "non-uniform phases: frustration E > 0 (source exists)");

  // All phases identical — zero frustration (no source energy)
  std::vector<double> ph_equal(N, 1.0);
  PhaseBattery bat_equal(N, 0.3, ph_equal);
  double E_equal = bat_equal.frustration();
  test_assert(E_equal < TOL,
              "uniform phases: frustration E = 0 (no source — dead battery)");

  // Frustration scales with spread: wider spread → more source energy
  auto ph_narrow = uniform_phases(N, 0.0, OHM_PI / 8.0);
  PhaseBattery bat_narrow(N, 0.3, ph_narrow);
  test_assert(bat_spread.frustration() > bat_narrow.frustration(),
              "wider spread → more frustration (stronger source)");
}

// ── 2. Sink is conserved
// ────────────────────────────────────────────────────── The mean attractor ψ̄
// must remain invariant while frustration flows into it, exactly as the
// electrolyte equilibrium is maintained in a battery.
void test_sink_is_conserved() {
  std::cout << "\n\u2554\u2550\u2550\u2550 2. Sink: Mean Phase Conserved "
               "\u2550\u2550\u2550\u2557\n";

  const int N = 20;
  const int STEPS = 50;

  // Symmetric spread: mean phase should be 0 throughout
  auto ph = uniform_phases(N, 0.0, OHM_PI / 2.0);
  PhaseBattery bat(N, 0.3, ph);
  double psi_bar_init = bat.mean_phase();

  double max_drift = 0.0;
  for (int i = 0; i < STEPS; ++i) {
    bat.step();
    double drift = std::abs(wrap(bat.mean_phase() - psi_bar_init));
    if (drift > max_drift)
      max_drift = drift;
  }
  test_assert(max_drift < LOOSE_TOL,
              "mean phase \u03c8\u0304 invariant over 50 steps "
              "(sink address conserved)");

  // Offset mean: ψ̄ should stay near π/4
  auto ph2 = uniform_phases(N, OHM_PI / 4.0, OHM_PI / 3.0);
  PhaseBattery bat2(N, 0.4, ph2);
  double psi2_init = bat2.mean_phase();
  for (int i = 0; i < STEPS; ++i)
    bat2.step();
  double drift2 = std::abs(wrap(bat2.mean_phase() - psi2_init));
  test_assert(
      drift2 < LOOSE_TOL,
      "offset mean phase stays near \u03c0/4 under EMA (sink invariant)");

  // Frustration must decrease (source gives to sink)
  auto ph3 = uniform_phases(N, 0.0, OHM_PI / 2.0);
  PhaseBattery bat3(N, 0.3, ph3);
  double E_init = bat3.frustration();
  for (int i = 0; i < STEPS; ++i)
    bat3.step();
  test_assert(bat3.frustration() < E_init,
              "frustration decreases over 50 steps (source flows to sink)");
}

// ── 3. Medium controls transfer rate ─────────────────────────────────────────
// g = G_eff controls how quickly frustration transfers from source to sink.
// Without G_eff (g=0) nothing moves.  With g∈(0,1) transfer is controlled.
// With g>2 (|1−g|>1) deviations amplify — analogous to a short circuit.
void test_medium_controls() {
  std::cout << "\n\u2554\u2550\u2550\u2550 3. Medium: G_eff Damping Controls "
               "Transfer \u2550\u2550\u2550\u2557\n";

  const int N = 16;
  const int STEPS = 20;
  auto ph = uniform_phases(N, 0.0, OHM_PI / 2.0);

  // g = 0 (open circuit): no energy transfers
  PhaseBattery bat0(N, 0.0, ph);
  double E0_before = bat0.frustration();
  for (int i = 0; i < STEPS; ++i)
    bat0.step();
  test_assert(std::abs(bat0.frustration() - E0_before) < TOL,
              "g=0 (open circuit): frustration unchanged — battery dead");

  // g = 0.3 (controlled medium): frustration decreases
  PhaseBattery bat3(N, 0.3, ph);
  for (int i = 0; i < STEPS; ++i)
    bat3.step();
  test_assert(
      bat3.frustration() < E0_before * 0.9,
      "g=0.3 (controlled medium): frustration decreases — battery works");

  // g = 1.0 (ideal medium): frustration collapses to near-zero in one step
  PhaseBattery bat1(N, 1.0, ph);
  bat1.step();
  test_assert(bat1.frustration() < TOL,
              "g=1.0 (ideal medium): frustration → 0 in one step");

  // Higher G_eff → faster transfer: after same number of steps, higher g
  // leaves less residual frustration (for g ∈ (0,1))
  PhaseBattery bat_lo(N, 0.1, ph);
  PhaseBattery bat_hi(N, 0.5, ph);
  for (int i = 0; i < STEPS; ++i) {
    bat_lo.step();
    bat_hi.step();
  }
  test_assert(bat_hi.frustration() < bat_lo.frustration(),
              "higher G_eff (g=0.5 > g=0.1) → lower residual frustration "
              "(better medium)");

  // G_eff = sech(λ): verify PhaseBattery with conductance(λ) matches
  double lam = 0.7;
  double g_from_lam = conductance(lam); // sech(0.7)
  PhaseBattery bat_lam(N, g_from_lam, ph);
  double released = bat_lam.step();
  test_assert(released > 0.0,
              "G_eff = sech(\u03bb=0.7): positive energy transfer per step "
              "(medium active)");
}

// ── 4. Three essentials — battery only works with all three
// ─────────────────── Systematically disable each essential and verify the
// battery fails.
void test_three_essentials() {
  std::cout << "\n\u2554\u2550\u2550\u2550 4. All Three Essentials Required "
               "\u2550\u2550\u2550\u2557\n";

  const int N = 20;
  const int STEPS = 30;

  // ── Missing SOURCE (all phases equal → frustration = 0) ──
  std::vector<double> ph_flat(N, OHM_PI / 6.0);
  PhaseBattery bat_nosrc(N, 0.3, ph_flat);
  double R_nosrc_init = bat_nosrc.circular_r();
  for (int i = 0; i < STEPS; ++i)
    bat_nosrc.step();
  // No source → coherence already at 1, nothing to convert
  test_assert(std::abs(bat_nosrc.circular_r() - 1.0) < LOOSE_TOL,
              "no source (uniform phases): R=1 throughout — no conversion");
  test_assert(bat_nosrc.frustration() < TOL,
              "no source: frustration stays at 0 — no energy to release");

  // ── Missing MEDIUM (g=0 → open circuit) ──
  auto ph = uniform_phases(N, 0.0, OHM_PI / 2.0);
  PhaseBattery bat_nomed(N, 0.0, ph);
  double R_nomed_init = bat_nomed.circular_r();
  for (int i = 0; i < STEPS; ++i)
    bat_nomed.step();
  test_assert(std::abs(bat_nomed.circular_r() - R_nomed_init) < TOL,
              "no medium (g=0): coherence R unchanged — open circuit");

  // ── "Short-circuit" (g > 2 → unstable: deviations amplify) ──
  // For |1 − g| > 1 (i.e. g > 2), each EMA step amplifies the deviations
  // rather than shrinking them — analogous to a short circuit overwhelming
  // the electrolyte's capacity to regulate flow.
  // With g = 2.5: δθ_j^new = (1 − 2.5) δθ_j = −1.5 δθ_j, so |δθ| grows
  // by 1.5× per step → frustration grows by 2.25× per step.
  PhaseBattery bat_sc(N, 2.5, ph);
  double E_sc_before = bat_sc.frustration();
  bat_sc.step();
  test_assert(bat_sc.frustration() > E_sc_before,
              "g=2.5 (short-circuit, |1\u2212g|>1): frustration GROWS "
              "\u2014 deviations amplify, battery unstable");

  // ── All three present → battery works ──
  PhaseBattery bat_ok(N, 0.3, ph);
  double R_init = bat_ok.circular_r();
  for (int i = 0; i < STEPS; ++i)
    bat_ok.step();
  double R_final = bat_ok.circular_r();
  test_assert(R_final > R_init + 0.1,
              "all three present (source + medium + sink): coherence rises "
              "\u2014 battery works");
}

// ── 5. Circular coherence R(t) is non-decreasing ─────────────────────────────
// For g ∈ (0, 1] every EMA step is a dissipative contraction of the phase
// deviations.  The circular coherence R = |⟨e^{iψ}⟩| is therefore
// monotonically non-decreasing (mirrors Grover-analogy §6.3).
void test_coherence_monotone() {
  std::cout << "\n\u2554\u2550\u2550\u2550 5. Circular Coherence R(t) "
               "Non-decreasing \u2550\u2550\u2550\u2557\n";

  const int N = 20;
  const int STEPS = 60;

  // Uniformly spaced phases (worst case: fully spread)
  auto ph = uniform_phases(N, 0.0, OHM_PI);
  PhaseBattery bat(N, 0.3, ph);
  double prev_R = bat.circular_r();
  bool monotone = true;
  double max_drop = 0.0;
  for (int i = 0; i < STEPS; ++i) {
    bat.step();
    double R = bat.circular_r();
    if (R < prev_R - TOL) {
      monotone = false;
      double drop = prev_R - R;
      if (drop > max_drop)
        max_drop = drop;
    }
    prev_R = R;
  }
  test_assert(monotone, "R(t) is monotonically non-decreasing for g=0.3 "
                        "(dissipative sink absorbs all released frustration)");

  // Also verify final R is significantly larger than initial R
  auto ph2 = uniform_phases(N, 0.0, OHM_PI);
  PhaseBattery bat2(N, 0.3, ph2);
  double R_start = bat2.circular_r();
  for (int i = 0; i < STEPS; ++i)
    bat2.step();
  test_assert(bat2.circular_r() > R_start + 0.3,
              "R grows by > 0.3 after 60 steps: sink successfully accumulates "
              "coherence");

  // g = 1 (ideal): single step brings all phases to the mean → R → 1
  auto ph3 = uniform_phases(N, 0.0, OHM_PI / 2.0);
  PhaseBattery bat3(N, 1.0, ph3);
  bat3.step();
  test_assert(bat3.circular_r() > 1.0 - LOOSE_TOL,
              "g=1 (ideal medium): one step achieves R = 1 "
              "(all frustration converted)");
}

// ── 6. G_eff modulation: higher G_eff → faster convergence ───────────────────
// Measures convergence speed (steps to reach R > 0.95) as a function of g,
// showing that higher G_eff ↔ lower λ ↔ better medium ↔ faster battery.
void test_g_eff_rate() {
  std::cout << "\n\u2554\u2550\u2550\u2550 6. G_eff Modulation: Transfer Rate "
               "\u2550\u2550\u2550\u2557\n";

  const int N = 20;
  const int MAX_STEPS = 500;
  const double TARGET_R = 0.95;
  auto ph = uniform_phases(N, 0.0, OHM_PI / 2.0);

  auto steps_to_target = [&](double g) -> int {
    PhaseBattery bat(N, g, ph);
    for (int i = 0; i < MAX_STEPS; ++i) {
      if (bat.circular_r() >= TARGET_R)
        return i;
      bat.step();
    }
    return MAX_STEPS; // did not converge
  };

  int steps_lo = steps_to_target(0.1);
  int steps_mid = steps_to_target(0.3);
  int steps_hi = steps_to_target(0.7);

  std::cout << std::fixed << std::setprecision(1);
  std::cout << "    Steps to R > 0.95:  g=0.1 → " << steps_lo << "  g=0.3 → "
            << steps_mid << "  g=0.7 → " << steps_hi << "\n";

  test_assert(steps_lo > steps_mid,
              "g=0.1 slower than g=0.3: higher G_eff → faster convergence");
  test_assert(steps_mid > steps_hi,
              "g=0.3 slower than g=0.7: higher G_eff → faster convergence");

  // Verify via λ: G_eff = sech(λ), so λ small → G_eff large → fast
  double lam_fast = 0.3; // sech(0.3) ≈ 0.956
  double lam_slow = 2.0; // sech(2.0) ≈ 0.266
  int steps_fast = steps_to_target(conductance(lam_fast));
  int steps_slow = steps_to_target(conductance(lam_slow));
  test_assert(steps_fast < steps_slow,
              "\u03bb=0.3 (G_eff\u22480.956) converges faster than \u03bb=2.0 "
              "(G_eff\u22480.266): lower \u03bb = better medium");

  // At g = 0 battery never charges (open circuit): steps = MAX_STEPS
  test_assert(steps_to_target(0.0) == MAX_STEPS,
              "g=0 (G_eff=0, open circuit): never reaches target R — "
              "battery dead without medium");
}

// ── 7. Silver-ratio node generation ──────────────────────────────────────────
// Validates that silver_growth_phases() and silver_folded_phases() produce
// nodes with the correct angular increments.
//
//   Growth phase  : φ_j = base + j · (2π/δ_s²)        outward spiral
//   Folded phase  : odd nodes contracted by 1/δ_s ≈ √2−1  (inward spiral)
//
// Key numerical checks:
//   • δ_s · (1/δ_s) = 1   (reciprocal identity)
//   • SILVER_ANGLE_INC = 2π/δ_s²
//   • Consecutive growth-phase gap = SILVER_ANGLE_INC
//   • Consecutive folded-phase gap = SILVER_ANGLE_INC · (scale of next node)
void test_silver_ratio_nodes() {
  std::cout << "\n\u2554\u2550\u2550\u2550 7. Silver-Ratio Node Generation "
               "\u2550\u2550\u2550\u2557\n";

  // ── Reciprocal identity: δ_s × (1/δ_s) = 1
  test_assert(
      std::abs(SILVER_RATIO * SILVER_RATIO_RECIP - 1.0) < TOL,
      "\u03b4_s \u00d7 (1/\u03b4_s) = 1 (reciprocal identity holds)");

  // ── SILVER_ANGLE_INC = 2π/δ_s²
  double expected_inc = 2.0 * OHM_PI / (SILVER_RATIO * SILVER_RATIO);
  test_assert(std::abs(SILVER_ANGLE_INC - expected_inc) < TOL,
              "SILVER_ANGLE_INC = 2\u03c0/\u03b4_s\u00b2 (definition correct)");

  // ── Growth phases: consecutive gap = SILVER_ANGLE_INC
  const int N = 16;
  auto growth = silver_growth_phases(N, 0.0);
  bool growth_gap_ok = true;
  for (int j = 1; j < N; ++j) {
    if (std::abs((growth[j] - growth[j - 1]) - SILVER_ANGLE_INC) > TOL)
      growth_gap_ok = false;
  }
  test_assert(growth_gap_ok,
              "silver_growth_phases: consecutive gap = SILVER_ANGLE_INC "
              "(outward spiral)");

  // ── Growth phases start at the supplied base angle
  auto growth_base = silver_growth_phases(N, SILVER_BALANCE_ANGLE);
  test_assert(std::abs(growth_base[0] - SILVER_BALANCE_ANGLE) < TOL,
              "silver_growth_phases: first node at base angle (3\u03c0/4)");

  // ── Folded phases: even nodes scale by 1, odd nodes by 1/δ_s
  auto folded = silver_folded_phases(N, 0.0);
  bool even_ok = true, odd_ok = true;
  for (int j = 1; j < N; ++j) {
    double scale = (j % 2 == 0) ? 1.0 : SILVER_RATIO_RECIP;
    double expected = static_cast<double>(j) * SILVER_ANGLE_INC * scale;
    if (std::abs(folded[j] - expected) > TOL) {
      if (j % 2 == 0)
        even_ok = false;
      else
        odd_ok = false;
    }
  }
  test_assert(even_ok,
              "silver_folded_phases: even nodes follow outward growth spiral");
  test_assert(odd_ok,
              "silver_folded_phases: odd nodes contracted by 1/\u03b4_s "
              "(inward/reciprocal fold)");

  // ── Odd folded phases are strictly closer to base than growth phases
  bool folded_smaller = true;
  for (int j = 1; j < N; j += 2) {
    if (std::abs(folded[j]) >= std::abs(growth[j]) - TOL)
      folded_smaller = false;
  }
  test_assert(
      folded_smaller,
      "odd folded phases nearer to base than growth phases (inward < outward)");

  // ── PhaseBattery with growth phases converges (source → sink)
  PhaseBattery bat_grow(N, 0.3, silver_growth_phases(N, 0.0));
  double E_init = bat_grow.frustration();
  for (int i = 0; i < 30; ++i)
    bat_grow.step();
  test_assert(bat_grow.frustration() < E_init,
              "PhaseBattery(silver_growth_phases): frustration decreases "
              "\u2014 source active");

  // ── PhaseBattery with folded phases also converges
  PhaseBattery bat_fold(N, 0.3, silver_folded_phases(N, 0.0));
  double E_fold_init = bat_fold.frustration();
  for (int i = 0; i < 30; ++i)
    bat_fold.step();
  test_assert(bat_fold.frustration() < E_fold_init,
              "PhaseBattery(silver_folded_phases): frustration decreases "
              "\u2014 source active");
}

// ── 8. Silver-ratio balance symmetry (3π/4 pivot, 8-fold stability) ──────────
// Validates mirrored precession through the balance angle 3π/4.
//
//   8-fold cancellation : 8 phases at base + k·(3π/4) (k=0…7) have R = 0
//                         since 3π/4 × 8 = 6π = 3×(2π) (three full rotations),
//                         visiting all 8 directions in multiples of π/4
//                         (equidistant unit vectors cancel).
//
//   Mirror symmetry     : reflecting each phase through 3π/4 (i.e. φ' = 3π/2 − φ)
//                         preserves frustration and circular_r — the
//                         battery observables are invariant under this
//                         reflection (rotational symmetry of the energy).
//
//   Silver-ratio balance: silver_growth_phases centred at 3π/4 converge
//                         symmetrically; frustration decreases while the
//                         mean phase stays near the 3π/4 balance angle.
void test_silver_balance_symmetry() {
  std::cout << "\n\u2554\u2550\u2550\u2550 8. Silver-Ratio Balance Symmetry "
               "(3\u03c0/4 pivot, 8-fold) \u2550\u2550\u2550\u2557\n";

  // ── SILVER_BALANCE_ANGLE = 3π/4
  test_assert(std::abs(SILVER_BALANCE_ANGLE - 3.0 * OHM_PI / 4.0) < TOL,
              "SILVER_BALANCE_ANGLE = 3\u03c0/4 (8-fold symmetry pivot)");

  // ── 8-fold cancellation: 8 equidistant phases at k·(3π/4) have R = 0
  // 3π/4 × 8 = 6π = 3×(2π): three full rotations, visiting all 8 distinct
  // directions (multiples of π/4); equidistant unit vectors sum to zero.
  {
    const int K = 8;
    std::vector<double> phases8(K);
    for (int k = 0; k < K; ++k)
      phases8[k] = static_cast<double>(k) * SILVER_BALANCE_ANGLE;
    PhaseBattery bat8(K, 0.0, phases8); // g=0: open circuit, no evolution
    test_assert(bat8.circular_r() < LOOSE_TOL,
                "8 phases at k\u00b7(3\u03c0/4): circular_r = 0 "
                "(8-fold equidistant cancellation)");
  }

  // ── Mirror symmetry: φ' = 3π/2 − φ preserves frustration and R
  {
    const int N = 16;
    auto orig = silver_growth_phases(N, 0.5); // arbitrary base ≠ 3π/4
    std::vector<double> mirrored(N);
    const double mirror_axis = 2.0 * SILVER_BALANCE_ANGLE; // 3π/2
    for (int j = 0; j < N; ++j)
      mirrored[j] = mirror_axis - orig[j];

    PhaseBattery bat_orig(N, 0.0, orig);
    PhaseBattery bat_mirr(N, 0.0, mirrored);

    test_assert(
        std::abs(bat_orig.frustration() - bat_mirr.frustration()) < LOOSE_TOL,
        "mirrored phases (3\u03c0/2 \u2212 \u03c6): frustration invariant "
        "under 3\u03c0/4 reflection");
    test_assert(
        std::abs(bat_orig.circular_r() - bat_mirr.circular_r()) < LOOSE_TOL,
        "mirrored phases: circular_r invariant under 3\u03c0/4 reflection");
  }

  // ── Silver growth phases centred at 3π/4: mean phase near 3π/4 initially
  {
    const int N = 9;
    // N=9 odd: symmetric about the centre element (j=4 → centre of spiral)
    auto ph = silver_growth_phases(N, SILVER_BALANCE_ANGLE);
    PhaseBattery bat(N, 0.3, ph);
    // After many steps the battery should converge; frustration must decrease
    double E0 = bat.frustration();
    for (int i = 0; i < 50; ++i)
      bat.step();
    test_assert(bat.frustration() < E0,
                "silver_growth_phases(base=3\u03c0/4): frustration decreases "
                "(converges toward balance angle)");
    test_assert(bat.circular_r() > 0.5,
                "silver_growth_phases(base=3\u03c0/4): R > 0.5 after 50 "
                "steps (majority coherence — cross stable threshold for "
                "8-fold symmetric convergence)");
  }

  // ── Folded phases at 3π/4 balance angle: source + sink active
  {
    const int N = 16;
    auto ph = silver_folded_phases(N, SILVER_BALANCE_ANGLE);
    PhaseBattery bat(N, 0.3, ph);
    double R_init = bat.circular_r();
    for (int i = 0; i < 50; ++i)
      bat.step();
    test_assert(bat.circular_r() >= R_init,
                "silver_folded_phases(base=3\u03c0/4): R non-decreasing "
                "(inward+outward spiral converges)");
  }
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
  std::cout << "\u2551  Phase Battery Analogy \u2014 Empirical Proof          "
               "              \u2551\n";
  std::cout << "\u2551  Source (frustration) + Sink (mean) + Medium (G_eff)   "
               "              \u2551\n";
  std::cout
      << "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u255d\n";

  test_source_exists();
  test_sink_is_conserved();
  test_medium_controls();
  test_three_essentials();
  test_coherence_monotone();
  test_g_eff_rate();
  test_silver_ratio_nodes();
  test_silver_balance_symmetry();

  std::cout
      << "\n\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2557\n";
  std::cout << "\u2551  Test Results                                          "
               "      \u2551\n";
  std::cout
      << "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u255d\n";
  std::cout << "  Total tests: " << test_count << "\n";
  std::cout << "  Passed:      " << passed << " \u2713\n";
  std::cout << "  Failed:      " << failed << " \u2717\n";

  if (failed == 0) {
    std::cout
        << "\n  \u2713 ALL TESTS PASSED \u2014 Phase Battery Analogy "
           "empirically verified\n"
           "    Source (phase frustration) + Sink (mean attractor) + Medium "
           "(G_eff)\n"
           "    are all present, active, and balanced in the KernelSync "
           "engine.\n\n";
    return 0;
  } else {
    std::cout << "\n  \u2717 TESTS FAILED \u2014 Check implementation\n\n";
    return 1;
  }
}
