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
#include <fstream>
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

// ── 7. Feedback loop stability
// ──────────────────────────────────────────────── The compute-feedback step
// must remain stable (non-amplifying) under both perfect coherence (R = 1, no
// frustration) and near-perfect coherence (R ≈ 0.99, tight phase spread).  High
// coherence amplifies the feedback gain, but because both sub-steps are
// independently dissipative the system converges.
void test_feedback_loop_stability() {
  std::cout << "\n\u2554\u2550\u2550\u2550 7. Feedback Loop: Stability Under "
               "Perfect / Near-Perfect Coherence \u2550\u2550\u2550\u2557\n";

  const int N = 20;
  const double ALPHA = 0.5; // conservative amplification

  // ── Perfect coherence: R = 1, frustration = 0 ──
  std::vector<double> ph_perfect(N, 0.0);
  PhaseBattery bat_perfect(N, 0.3, ph_perfect);
  test_assert(bat_perfect.circular_r() > 1.0 - TOL,
              "perfect coherence: initial R = 1");
  bat_perfect.feedback_step(ALPHA);
  test_assert(bat_perfect.frustration() < TOL,
              "perfect coherence: frustration stays 0 after feedback step");
  test_assert(bat_perfect.circular_r() > 1.0 - TOL,
              "perfect coherence: R = 1 preserved after feedback step");

  // ── Near-perfect coherence: tight spread → R > 0.98 ──
  auto ph_near = uniform_phases(N, 0.0, 0.1); // ±0.1 rad
  PhaseBattery bat_near(N, 0.3, ph_near);
  test_assert(bat_near.circular_r() > 0.98,
              "near-perfect coherence: initial R > 0.98 for tight spread");

  bool stable = true;
  for (int i = 0; i < 30; ++i) {
    double R_before = bat_near.circular_r();
    bat_near.feedback_step(ALPHA);
    if (bat_near.circular_r() < R_before - TOL)
      stable = false;
  }
  test_assert(stable,
              "near-perfect coherence: R(t) non-decreasing under feedback "
              "(system stable)");
  test_assert(bat_near.circular_r() > 1.0 - LOOSE_TOL,
              "near-perfect coherence: converges to R = 1 under feedback");

  // ── Feedback converges at least as fast as the standard step ──
  auto ph_test = uniform_phases(N, 0.0, OHM_PI / 3.0);
  PhaseBattery bat_std(N, 0.3, ph_test);
  PhaseBattery bat_fb(N, 0.3, ph_test);
  for (int i = 0; i < 20; ++i) {
    bat_std.step();
    bat_fb.feedback_step(1.0);
  }
  test_assert(bat_fb.circular_r() >= bat_std.circular_r(),
              "feedback step achieves \u2265 coherence of standard step after "
              "20 iterations");
}

// ── 8. Interaction energy scales predictably as R → 1 ────────────────────────
// E_interact(R) = R² · N · g is monotonically increasing and reaches its
// maximum N·g at R = 1 (perfect coherence limit), analogous to the lensing
// focal intensity of a metallic mirror concentrating coherent energy.
void test_interaction_energy_scaling() {
  std::cout
      << "\n\u2554\u2550\u2550\u2550 8. Interaction Energy Scaling: "
         "R \u2192 1.0 (Perfect Coherence Limit) \u2550\u2550\u2550\u2557\n";

  const int N = 20;
  const double g = 0.3;

  // Analytic sweep: verify E_interact is monotonically increasing
  std::cout << "    R      E_interact\n";
  double E_prev = interaction_energy(0.0, N, g);
  bool monotone = true;
  for (int k = 1; k <= 10; ++k) {
    double R = k * 0.1;
    double E = interaction_energy(R, N, g);
    std::cout << "    " << std::fixed << std::setprecision(1) << R << "    "
              << std::setprecision(4) << E << "\n";
    if (E < E_prev - TOL)
      monotone = false;
    E_prev = E;
  }
  test_assert(monotone,
              "E_interact = R\u00b2\u00b7N\u00b7g is monotonically increasing "
              "as R \u2192 1.0");

  // Limiting value at perfect coherence: E → N·g
  double E_max = interaction_energy(1.0, N, g);
  test_assert(std::abs(E_max - static_cast<double>(N) * g) < TOL,
              "at R = 1: E_interact = N\u00b7g (maximum focal energy)");

  // Simulated convergence: interaction energy rises as phases align
  auto ph = uniform_phases(N, 0.0, OHM_PI / 2.0);
  PhaseBattery bat(N, g, ph);
  double E_sim_prev = interaction_energy(bat.circular_r(), N, g);
  bool sim_monotone = true;
  std::cout << "\n    Simulated convergence (step, R, E_interact):\n";
  for (int blk = 0; blk < 6; ++blk) {
    for (int j = 0; j < 5; ++j)
      bat.feedback_step(0.5);
    double R = bat.circular_r();
    double E_i = interaction_energy(R, N, g);
    std::cout << "    step " << std::setw(2) << (blk + 1) * 5
              << ":  R = " << std::setprecision(4) << R
              << "  E_interact = " << std::setprecision(4) << E_i << "\n";
    if (E_i < E_sim_prev - TOL)
      sim_monotone = false;
    E_sim_prev = E_i;
  }
  test_assert(sim_monotone,
              "interaction energy increases monotonically during feedback "
              "convergence simulation");
}

// ── 9. Silver balance symmetry with compute feedback ─────────────────────────
// A mirror-symmetric initial phase distribution (ψ_j = −ψ_{N−1−j}) must
// remain symmetric through iterative feedback transformations, and the
// metallic_oscillating_phases function must preserve this balance.
// Multi-phase simulation stacks both transforms and verifies focal coherence.
void test_silver_balance_symmetry() {
  std::cout
      << "\n\u2554\u2550\u2550\u2550 9. Silver Balance Symmetry: Mirrored "
         "Recursion + Compute Feedback \u2550\u2550\u2550\u2557\n";

  const int N = 20;
  const double ALPHA = 0.5;

  // Build mirror-symmetric phases: ψ_j = −ψ_{N−1−j}, centred on 0
  auto ph_sym = uniform_phases(N, 0.0, OHM_PI / 4.0);
  bool init_sym = true;
  for (int j = 0; j < (N + 1) / 2; ++j)
    if (std::abs(ph_sym[j] + ph_sym[N - 1 - j]) > TOL)
      init_sym = false;
  test_assert(init_sym, "initial phases are mirror-symmetric around \u03c8=0 "
                        "(silver balance)");

  // Feedback steps must preserve mirror symmetry
  PhaseBattery bat(N, 0.3, ph_sym);
  bool sym_preserved = true;
  for (int s = 0; s < 20; ++s) {
    bat.feedback_step(ALPHA);
    for (int j = 0; j < (N + 1) / 2; ++j)
      if (std::abs(bat.phases[j] + bat.phases[N - 1 - j]) > LOOSE_TOL)
        sym_preserved = false;
  }
  test_assert(sym_preserved,
              "mirror symmetry \u03c8_j = \u2212\u03c8_{N\u22121\u2212j} "
              "preserved through 20 feedback steps (mirrored recursion "
              "stable)");

  // metallic_oscillating_phases: output spread \u2264 input spread (lensing
  // focuses phases toward the alignment angle)
  auto ph_spread = uniform_phases(N, 0.0, OHM_PI / 3.0);
  auto ph_out = metallic_oscillating_phases(ph_spread, 0.0, 1.0);
  double ss_in = 0.0, ss_out = 0.0;
  for (double p : ph_spread)
    ss_in += p * p;
  for (double p : ph_out)
    ss_out += p * p;
  test_assert(ss_out <= ss_in + TOL,
              "metallic_oscillating_phases: output spread \u2264 input spread "
              "(constructive lensing focuses phases)");

  // Multi-phase simulation: interleave metallic projection + feedback,
  // verifying focal coherence reaches near-unity
  std::cout << "    Multi-phase simulation (metallic projection + feedback "
               "\u03b1=0.5, 10 iter):\n";
  auto ph_multi = uniform_phases(N, 0.0, OHM_PI / 2.0);
  PhaseBattery bat_multi(N, 0.3, ph_multi);
  for (int iter = 0; iter < 10; ++iter) {
    bat_multi.phases = metallic_oscillating_phases(
        bat_multi.phases, bat_multi.mean_phase(), ALPHA);
    bat_multi.feedback_step(ALPHA);
    if (iter % 2 == 1)
      std::cout << "    iter " << std::setw(2) << (iter + 1)
                << ":  R = " << std::fixed << std::setprecision(4)
                << bat_multi.circular_r() << "\n";
  }
  test_assert(bat_multi.circular_r() > 0.99,
              "multi-phase simulation: focal coherence R > 0.99 after 10 "
              "combined feedback iterations");
}

// ── 10. Adaptive α — stable initial conditions (moderate coherence)
// ─────────────────
// With c1 > 0 and c2 > 0, the adaptive α grows when the step produces large
// frustration decay (ΔE) or coherence gain (ΔR).  Under moderate initial
// conditions the adaptation must not destabilise the battery.
void test_adaptive_alpha_stable() {
  std::cout << "\n\u2554\u2550\u2550\u2550 10. Adaptive \u03b1 \u2014 Stable "
               "Initial Conditions \u2550\u2550\u2550\u2557\n";

  const int N = 20;
  const int STEPS = 30;
  auto ph = uniform_phases(N, 0.0, OHM_PI / 4.0); // moderate spread → R ≈ 0.9

  PhaseBattery bat(N, 0.3, ph);
  bat.set_alpha_sensitivity(0.2, 0.2);
  bat.enable_debug(true);

  double R_prev = bat.circular_r();
  bool non_decreasing_R = true;
  for (int i = 0; i < STEPS; ++i) {
    bat.feedback_step(0.5);
    double R = bat.circular_r();
    if (R < R_prev - LOOSE_TOL)
      non_decreasing_R = false;
    R_prev = R;
  }

  test_assert(non_decreasing_R,
              "adaptive \u03b1 (c1=c2=0.2): R(t) non-decreasing under stable "
              "initial conditions");
  test_assert(bat.circular_r() > 0.99,
              "adaptive \u03b1: converges to R > 0.99 from moderate spread");

  // History must be populated when debug is on
  test_assert(static_cast<int>(bat.history_E.size()) == STEPS,
              "debug history_E records one entry per feedback_step");
  test_assert(static_cast<int>(bat.history_R.size()) == STEPS,
              "debug history_R records one entry per feedback_step");
  test_assert(static_cast<int>(bat.history_alpha.size()) == STEPS,
              "debug history_alpha records one entry per feedback_step");

  // All recorded α values must be in [0, 1]
  bool alpha_clamped = true;
  for (double a : bat.history_alpha)
    if (a < 0.0 || a > 1.0 + LOOSE_TOL)
      alpha_clamped = false;
  test_assert(alpha_clamped,
              "all recorded \u03b1_adaptive values are clamped to [0, 1]");
}

// ── 11. Adaptive α — high initial frustration (low coherence)
// ─────────────────── Wide initial spread pushes the battery into a high-ΔE /
// high-ΔR regime, causing the adaptive α to be elevated in early steps.  The
// adaptive variant must still converge and must not inflate α beyond 1.
void test_adaptive_alpha_high_frustration() {
  std::cout << "\n\u2554\u2550\u2550\u2550 11. Adaptive \u03b1 \u2014 High "
               "Initial Frustration \u2550\u2550\u2550\u2557\n";

  const int N = 20;
  const int STEPS = 60;
  auto ph = uniform_phases(N, 0.0, OHM_PI * 0.9); // near-full spread → R ≈ 0.1

  // Compare: fixed α = 0.5 vs adaptive α with c1 = 0.3
  PhaseBattery bat_fixed(N, 0.3, ph);
  PhaseBattery bat_adapt(N, 0.3, ph);
  bat_adapt.set_alpha_sensitivity(0.3, 0.0);

  double E_fixed = bat_fixed.frustration();
  double E_adapt = bat_adapt.frustration();

  for (int i = 0; i < STEPS; ++i) {
    bat_fixed.feedback_step(0.5);
    bat_adapt.feedback_step(0.5);
  }

  test_assert(bat_adapt.frustration() < E_adapt * 0.01,
              "adaptive \u03b1: frustration drops to < 1 % of initial E under "
              "high frustration");
  test_assert(bat_fixed.frustration() < E_fixed * 0.01,
              "fixed \u03b1: frustration also drops to < 1 % (baseline)");

  // Energy efficiency: adaptive α must achieve ΔE/step ≥ fixed α
  double dE_fixed = (E_fixed - bat_fixed.frustration()) / STEPS;
  double dE_adapt = (E_adapt - bat_adapt.frustration()) / STEPS;
  test_assert(dE_adapt >= dE_fixed - LOOSE_TOL,
              "adaptive \u03b1 (\u0394E/step) \u2265 fixed \u03b1 under high "
              "initial frustration");
}

// ── 12. Adaptive α — oscillatory near-critical gain
// ─────────────────────────── With g close to 1, the EMA step is aggressive
// (near-full collapse in one step).  Adaptive α must not push the feedback gain
// above 1, so the system remains stable even in this near-critical regime.
void test_adaptive_alpha_near_critical() {
  std::cout << "\n\u2554\u2550\u2550\u2550 12. Adaptive \u03b1 \u2014 "
               "Near-Critical Gain \u2550\u2550\u2550\u2557\n";

  const int N = 20;
  const int STEPS = 20;
  auto ph = uniform_phases(N, 0.0, OHM_PI / 3.0);

  // g = 0.9 — near-critical but stable (|1−g| = 0.1 < 1)
  PhaseBattery bat(N, 0.9, ph);
  bat.set_alpha_sensitivity(0.5, 0.5); // aggressive sensitivity
  bat.enable_debug(true);

  bool non_decreasing_R = true;
  double R_prev = bat.circular_r();
  for (int i = 0; i < STEPS; ++i) {
    bat.feedback_step(0.5);
    double R = bat.circular_r();
    if (R < R_prev - LOOSE_TOL)
      non_decreasing_R = false;
    R_prev = R;
  }

  test_assert(non_decreasing_R,
              "near-critical g=0.9: R(t) non-decreasing with adaptive \u03b1 "
              "(c1=c2=0.5)");
  test_assert(bat.circular_r() > 1.0 - LOOSE_TOL,
              "near-critical g=0.9: converges to R = 1 with adaptive \u03b1");

  // Feedback gain g_fb = g·α·R must never cause g_fb > 1 (stability proof)
  // Since α_adaptive ≤ 1 and g=0.9 ≤ 1 and R ≤ 1: g_fb ≤ 0.9 < 1 always
  bool all_alpha_valid = true;
  for (double a : bat.history_alpha)
    if (a < 0.0 || a > 1.0 + LOOSE_TOL)
      all_alpha_valid = false;
  test_assert(all_alpha_valid,
              "near-critical: all adaptive \u03b1 values remain in [0, 1]");
}

// ── 13. Debug CSV output
// ────────────────────────────────────────────────────── Verify that
// write_debug_csv produces a well-formed file that can be read back, and that
// the CSV column count and row count match the history vectors.
void test_debug_csv_output() {
  std::cout << "\n\u2554\u2550\u2550\u2550 13. Debug CSV Output "
               "\u2550\u2550\u2550\u2557\n";

  const int N = 10;
  const int STEPS = 5;
  auto ph = uniform_phases(N, 0.0, OHM_PI / 4.0);

  PhaseBattery bat(N, 0.3, ph);
  bat.enable_debug(true);
  for (int i = 0; i < STEPS; ++i)
    bat.feedback_step(0.5);

  const std::string csv_path = "/tmp/test_phase_battery_debug.csv";
  bat.write_debug_csv(csv_path);

  // Read back and count lines (header + STEPS data rows)
  std::ifstream f(csv_path);
  test_assert(f.is_open(), "write_debug_csv: file created successfully");

  int lines = 0;
  std::string line;
  while (std::getline(f, line))
    ++lines;
  f.close();

  test_assert(lines == STEPS + 1,
              "write_debug_csv: file has header + one row per step");
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
  test_feedback_loop_stability();
  test_interaction_energy_scaling();
  test_silver_balance_symmetry();
  test_adaptive_alpha_stable();
  test_adaptive_alpha_high_frustration();
  test_adaptive_alpha_near_critical();
  test_debug_csv_output();

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
