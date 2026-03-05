/*
 * test_time_crystal_simulation.cpp — Integration tests for TimeCrystalSimulation
 *
 * Validates that the PhaseBattery coherence (R) correctly drives the
 * TimeCrystal's periodic dynamics (T) through the TimeCrystalSimulation
 * coupling defined in TimeCrystalSimulation.hpp.
 *
 * Test structure:
 *   1. period_driven_by_R      — T_eff = T_base / R; quasi-energy ε_F = π·R/T_base
 *   2. coherence_convergence   — R rises toward 1, E falls toward 0 over steps
 *   3. floquet_advance_exact   — at R=1 (perfect coherence), one step advances ψ
 *                                by exactly φ_eff = π·T_step/T_base
 *   4. period_doubling_at_R1   — T_base=1, T_step=1, R=1 → φ_eff=π → ψ(2)=ψ(0)
 *   5. simulation_loop_runs    — run(n) executes exactly n steps
 *   6. frustration_decrease    — E is non-increasing during the simulation
 *   7. coherence_nondecreasing — R is non-decreasing during the simulation
 *   8. verbose_output          — run with verbose=true produces console output
 *   9. adaptive_alpha_link     — with c1>0, c2>0 the system still converges
 *  10. psi_norm_invariant      — |ψ| = 1 throughout (Floquet unitarity)
 *
 * Test style matches test_battery_analogy.cpp (using test_assert + pass/fail).
 */

#include "TimeCrystalSimulation.hpp"

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace kernel::ohm;

// ── Test framework (matches test_battery_analogy.cpp) ────────────────────────
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

// ── Helper: build N uniformly-spread phases centred on 0 ─────────────────────
static std::vector<double> uniform_phases(int N, double spread) {
  std::vector<double> ph(static_cast<size_t>(N));
  for (int j = 0; j < N; ++j)
    ph[static_cast<size_t>(j)] =
        (N > 1) ? spread * (2.0 * j / (N - 1.0) - 1.0) : 0.0;
  return ph;
}

// ── 1. T_eff = T_base / R; ε_F = π · R / T_base ─────────────────────────────
void test_period_driven_by_R() {
  std::cout << "\n\u2554\u2550\u2550\u2550 1. Drive Period T_eff Driven by "
               "Coherence R \u2550\u2550\u2550\u2557\n";

  // Construct a sim with fully aligned phases → R = 1 from the start.
  const int N = 8;
  const double T_base = 2.0;
  std::vector<double> ph_perfect(static_cast<size_t>(N), 0.0); // all phases = 0 → R = 1
  TimeCrystalSimulation sim(N, 0.3, ph_perfect, T_base);

  test_assert(sim.coherence() > 1.0 - TOL,
              "perfect alignment: initial R = 1");
  test_assert(std::abs(sim.effective_period() - T_base) < TOL,
              "T_eff = T_base when R = 1");
  test_assert(std::abs(sim.quasi_energy() - OHM_PI / T_base) < TOL,
              "\u03b5_F = \u03c0/T_base when R = 1");

  // Lower coherence: T_eff should grow.
  std::vector<double> ph_spread = uniform_phases(N, OHM_PI / 2.0);
  TimeCrystalSimulation sim2(N, 0.3, ph_spread, T_base);
  double R2 = sim2.coherence();
  double Teff2 = sim2.effective_period();
  test_assert(R2 < 1.0, "spread phases: initial R < 1");
  test_assert(Teff2 > T_base + TOL, "spread phases: T_eff > T_base (period elongated)");
  test_assert(std::abs(Teff2 - T_base / R2) < LOOSE_TOL,
              "T_eff = T_base / R exactly");
  test_assert(
      std::abs(sim2.quasi_energy() - OHM_PI * R2 / T_base) < LOOSE_TOL,
      "\u03b5_F = \u03c0 \u00b7 R / T_base");
}

// ── 2. Coherence rises, frustration falls ────────────────────────────────────
void test_coherence_convergence() {
  std::cout << "\n\u2554\u2550\u2550\u2550 2. Coherence Convergence: R \u2192 1, "
               "E \u2192 0 \u2550\u2550\u2550\u2557\n";

  const int N = 16;
  auto ph = uniform_phases(N, OHM_PI / 3.0);
  TimeCrystalSimulation sim(N, 0.3, ph, 1.0, 0.5);

  double R_init = sim.coherence();
  double E_init = sim.frustration();
  test_assert(R_init < 1.0, "initial R < 1 (non-trivial spread)");
  test_assert(E_init > TOL, "initial E > 0 (frustration present)");

  sim.run(50);

  test_assert(sim.coherence() > R_init, "R increased after 50 steps");
  test_assert(sim.frustration() < E_init, "E decreased after 50 steps");
  test_assert(sim.coherence() > 0.99,
              "R > 0.99 after 50 steps (near-full coherence)");
  test_assert(sim.frustration() < 1e-4, "E < 1e-4 after 50 steps");
}

// ── 3. Exact Floquet advance at R = 1 ────────────────────────────────────────
void test_floquet_advance_exact() {
  std::cout << "\n\u2554\u2550\u2550\u2550 3. Exact Floquet Advance at R = 1 "
               "\u2550\u2550\u2550\u2557\n";

  const int N = 4;
  const double T_base = 2.0; // T_step = 1.0 → φ_eff = π/2 at R=1
  std::vector<double> ph(static_cast<size_t>(N), 0.0); // R = 1 exactly
  TimeCrystalSimulation sim(N, 0.3, ph, T_base);

  // At R=1, φ_eff = π·T_step/T_eff = π·1.0/2.0 = π/2
  // ψ(0) = 1+0i → after 1 step: ψ(1) = exp(−iπ/2)·ψ(0) = (0 − i)·1 = −i
  // ψ(2) = exp(−iπ/2)·(−i) = −i·(−i) = i² = −1
  // ψ(4) = ψ(0) (period-4 Floquet cycle at T_base=2)
  double init_re = sim.psi_re;
  double init_im = sim.psi_im;
  test_assert(std::abs(init_re - 1.0) < TOL && std::abs(init_im) < TOL,
              "initial \u03c8 = 1 + 0i");

  sim.step();
  // exp(−iπ/2)·1 = cos(π/2) − i·sin(π/2) = 0 − i
  test_assert(std::abs(sim.psi_re - 0.0) < LOOSE_TOL,
              "after 1 step (T_base=2): Re(\u03c8) = 0");
  test_assert(std::abs(sim.psi_im - (-1.0)) < LOOSE_TOL,
              "after 1 step (T_base=2): Im(\u03c8) = \u22121");

  sim.step();
  // exp(−iπ/2)·(−i) = −i·(cos π/2 − i·sin π/2) = −i·(−i) = −1
  test_assert(std::abs(sim.psi_re - (-1.0)) < LOOSE_TOL,
              "after 2 steps (T_base=2): Re(\u03c8) = \u22121");
  test_assert(std::abs(sim.psi_im - 0.0) < LOOSE_TOL,
              "after 2 steps (T_base=2): Im(\u03c8) = 0");

  sim.step(); sim.step();
  // After 4 steps: ψ = exp(−i·4·π/2)·ψ(0) = exp(−i2π)·1 = 1
  test_assert(std::abs(sim.psi_re - 1.0) < LOOSE_TOL &&
                  std::abs(sim.psi_im) < LOOSE_TOL,
              "after 4 steps (T_base=2): \u03c8 = 1 (period-4 cycle)");
}

// ── 4. Period-doubling time crystal at T_base = 1, T_step = 1, R = 1 ─────────
void test_period_doubling_at_R1() {
  std::cout << "\n\u2554\u2550\u2550\u2550 4. Period-Doubling Time Crystal at "
               "T_base = 1, R = 1 \u2550\u2550\u2550\u2557\n";

  const int N = 4;
  const double T_base = 1.0; // T_step = 1.0 → φ_eff = π at R=1
  std::vector<double> ph(static_cast<size_t>(N), 0.0); // R = 1 exactly
  TimeCrystalSimulation sim(N, 0.3, ph, T_base);

  // φ_eff = π → ψ → −ψ each step → period-2 time crystal.
  test_assert(std::abs(sim.psi_re - 1.0) < TOL, "initial \u03c8 = 1");

  sim.step();
  // exp(−iπ)·1 = −1
  test_assert(std::abs(sim.psi_re - (-1.0)) < LOOSE_TOL &&
                  std::abs(sim.psi_im) < LOOSE_TOL,
              "after 1 step: \u03c8 = \u22121 (sign flip, period-doubling)");

  sim.step();
  // exp(−iπ)·(−1) = −exp(−iπ)·(−1)... wait, = 1
  test_assert(std::abs(sim.psi_re - 1.0) < LOOSE_TOL &&
                  std::abs(sim.psi_im) < LOOSE_TOL,
              "after 2 steps: \u03c8 = 1 (period-2 recovery, time crystal)");

  // Verify period-2 holds for many more steps
  bool period_2_holds = true;
  for (int k = 0; k < 20; ++k) {
    double re_odd = sim.psi_re;  // will flip to −1
    double im_odd = sim.psi_im;
    sim.step();
    if (std::abs(sim.psi_re + re_odd) > LOOSE_TOL ||
        std::abs(sim.psi_im + im_odd) > LOOSE_TOL)
      period_2_holds = false;
    sim.step(); // return
  }
  test_assert(period_2_holds,
              "period-2 time crystal: \u03c8(t+2) = \u03c8(t) holds for 20 "
              "additional double-steps");
}

// ── 5. run(n) executes exactly n steps ───────────────────────────────────────
void test_simulation_loop_runs() {
  std::cout << "\n\u2554\u2550\u2550\u2550 5. Simulation Loop Runs Configurable "
               "Steps \u2550\u2550\u2550\u2557\n";

  const int N = 8;
  auto ph = uniform_phases(N, OHM_PI / 4.0);
  TimeCrystalSimulation sim(N, 0.3, ph, 1.0, 0.5);

  test_assert(sim.step_count == 0, "initial step_count = 0");

  sim.run(10);
  test_assert(sim.step_count == 10, "after run(10): step_count = 10");

  sim.run(5);
  test_assert(sim.step_count == 15, "after run(5): step_count = 15");

  sim.step();
  test_assert(sim.step_count == 16, "after step(): step_count = 16");

  sim.run(0);
  test_assert(sim.step_count == 16, "run(0) is a no-op");
}

// ── 6. Frustration non-increasing during simulation ───────────────────────────
void test_frustration_decrease() {
  std::cout << "\n\u2554\u2550\u2550\u2550 6. Frustration Non-Increasing During "
               "Simulation \u2550\u2550\u2550\u2557\n";

  const int N = 20;
  auto ph = uniform_phases(N, OHM_PI / 3.0);
  TimeCrystalSimulation sim(N, 0.3, ph, 1.0, 0.5);

  bool non_increasing = true;
  double E_prev = sim.frustration();
  for (int i = 0; i < 40; ++i) {
    sim.step();
    double E = sim.frustration();
    if (E > E_prev + LOOSE_TOL)
      non_increasing = false;
    E_prev = E;
  }
  test_assert(non_increasing,
              "frustration E(t) is non-increasing for g=0.3, \u03b1=0.5 over "
              "40 steps");
}

// ── 7. Coherence non-decreasing during simulation ────────────────────────────
void test_coherence_nondecreasing() {
  std::cout << "\n\u2554\u2550\u2550\u2550 7. Coherence Non-Decreasing During "
               "Simulation \u2550\u2550\u2550\u2557\n";

  const int N = 20;
  auto ph = uniform_phases(N, OHM_PI / 3.0);
  TimeCrystalSimulation sim(N, 0.3, ph, 1.0, 0.5);

  bool non_decreasing = true;
  double R_prev = sim.coherence();
  for (int i = 0; i < 40; ++i) {
    sim.step();
    double R = sim.coherence();
    if (R < R_prev - LOOSE_TOL)
      non_decreasing = false;
    R_prev = R;
  }
  test_assert(non_decreasing,
              "coherence R(t) is non-decreasing for g=0.3, \u03b1=0.5 over "
              "40 steps");
}

// ── 8. Verbose output produces console lines ─────────────────────────────────
void test_verbose_output() {
  std::cout << "\n\u2554\u2550\u2550\u2550 8. Verbose Output \u2550\u2550\u2550"
               "\u2557\n";

  const int N = 8;
  auto ph = uniform_phases(N, OHM_PI / 4.0);

  // Redirect stdout to a stringstream to capture output
  std::streambuf *old_buf = std::cout.rdbuf();
  std::ostringstream captured;
  std::cout.rdbuf(captured.rdbuf());

  TimeCrystalSimulation sim(N, 0.3, ph, 1.0, 0.5, /*verbose=*/true);
  sim.run(3);

  std::cout.rdbuf(old_buf); // restore stdout

  std::string out = captured.str();
  // Expect 3 lines each containing "step=" and "R=" and "E="
  int line_count = 0;
  bool has_R = true;
  bool has_E = true;
  std::istringstream iss(out);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.find("step=") != std::string::npos) {
      ++line_count;
      if (line.find("R=") == std::string::npos)
        has_R = false;
      if (line.find("E=") == std::string::npos)
        has_E = false;
    }
  }

  test_assert(line_count == 3,
              "verbose: 3 output lines for run(3), one per step");
  test_assert(has_R, "verbose: each line contains 'R=' (coherence)");
  test_assert(has_E, "verbose: each line contains 'E=' (frustration)");
}

// ── 9. Adaptive α acceleration ───────────────────────────────────────────────
void test_adaptive_alpha_link() {
  std::cout << "\n\u2554\u2550\u2550\u2550 9. Adaptive \u03b1 Acceleration "
               "(c1 > 0, c2 > 0) \u2550\u2550\u2550\u2557\n";

  const int N = 20;
  const int STEPS = 40;
  auto ph = uniform_phases(N, OHM_PI / 3.0);

  // Fixed α = 0.5
  TimeCrystalSimulation sim_fixed(N, 0.3, ph, 1.0, 0.5);
  sim_fixed.run(STEPS);

  // Adaptive α = 0.5, c1 = 0.2, c2 = 0.1 — should converge at least as fast
  TimeCrystalSimulation sim_adapt(N, 0.3, ph, 1.0, 0.5);
  sim_adapt.battery.set_alpha_sensitivity(0.2, 0.1);
  sim_adapt.run(STEPS);

  test_assert(sim_adapt.coherence() > 0.99,
              "adaptive \u03b1 (c1=0.2, c2=0.1): R > 0.99 after 40 steps");
  test_assert(sim_adapt.coherence() >= sim_fixed.coherence() - LOOSE_TOL,
              "adaptive \u03b1: final R \u2265 fixed \u03b1 R (converges at "
              "least as fast)");
  test_assert(sim_adapt.effective_period() < sim_fixed.effective_period() + LOOSE_TOL,
              "adaptive \u03b1: T_eff smaller or equal (coherence-driven "
              "period tightens)");
}

// ── 10. |ψ| = 1 throughout (Floquet unitarity) ───────────────────────────────
void test_psi_norm_invariant() {
  std::cout << "\n\u2554\u2550\u2550\u2550 10. Crystal State |"
               "\u03c8| Invariant (Floquet Unitarity) \u2550\u2550\u2550\u2557\n";

  const int N = 16;
  auto ph = uniform_phases(N, OHM_PI / 2.0);
  TimeCrystalSimulation sim(N, 0.3, ph, 1.0, 0.5);

  // Initial |ψ| = 1 by construction
  test_assert(std::abs(sim.psi_abs() - 1.0) < TOL,
              "initial |\u03c8| = 1");

  bool norm_invariant = true;
  for (int i = 0; i < 50; ++i) {
    sim.step();
    if (std::abs(sim.psi_abs() - 1.0) > LOOSE_TOL)
      norm_invariant = false;
  }
  test_assert(norm_invariant,
              "|\u03c8| = 1 maintained over 50 steps (exp(\u2212i\u03c6) is "
              "unitary)");
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main() {
  std::cout
      << "\n\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2557\n";
  std::cout
      << "\u2551  TimeCrystal Simulation \u2014 PhaseBattery Coherence "
         "Integration                \u2551\n";
  std::cout
      << "\u2551  R drives T_eff = T_base/R; \u03b5_F = \u03c0\u00b7R/T_base;"
         " Floquet \u03c8 \u2190 exp(\u2212i\u03c6_eff)\u00b7\u03c8          "
         "          \u2551\n";
  std::cout
      << "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
         "\u255d\n";

  test_period_driven_by_R();
  test_coherence_convergence();
  test_floquet_advance_exact();
  test_period_doubling_at_R1();
  test_simulation_loop_runs();
  test_frustration_decrease();
  test_coherence_nondecreasing();
  test_verbose_output();
  test_adaptive_alpha_link();
  test_psi_norm_invariant();

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
        << "\n  \u2713 ALL TESTS PASSED \u2014 PhaseBattery coherence R "
           "correctly drives\n"
           "    TimeCrystal period T_eff = T_base/R; Floquet quasi-energy\n"
           "    \u03b5_F = \u03c0\u00b7R/T_base scales with coherence; "
           "adaptive amplification\n"
           "    (g, \u03b1, c1, c2) links battery dynamics to crystal "
           "periodicity.\n\n";
    return 0;
  } else {
    std::cout << "\n  \u2717 TESTS FAILED \u2014 Check implementation\n\n";
    return 1;
  }
}
