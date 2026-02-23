/*
 * sqrt_n_demo.cpp — O(√N) Classical Hardware Demonstration
 *
 * Six-section end-to-end demonstration of the SqrtNKernel integration layer,
 * showing how KernelPipeline, PalindromePrecession, SpectralBridge, and
 * KernelState unify into an O(√N) classical hardware model.
 *
 * Sections:
 *   1. Invariant Verification   — canonical-state invariants via SqrtNKernel
 *   2. Kernel Construction      — build kernels for representative N values
 *   3. Phase Sweep              — phasor_at(k·√N) unit-circle demonstration
 *   4. G_eff Accumulation       — weighted per-step G_eff via channel()
 *   5. Classical Speedup Metric — quantitative O(N) baseline vs O(√N) kernel
 *   6. Cross-Validation         — G_eff vs theoretical sech(λ) at r = 1
 */

#include "SqrtNKernel.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace kernel::sqrtn;
using namespace kernel::pipeline;
using namespace kernel::quantum;

// ── Output helpers ────────────────────────────────────────────────────────────

static void print_banner(const std::string &title) {
  std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
  std::cout << "║  " << std::left << std::setw(56) << title << "║\n";
  std::cout << "╚══════════════════════════════════════════════════════════╝\n";
}

static void print_section(int n, const std::string &title) {
  std::cout << "\n┌─ Section " << n << ": " << title
            << " ─────────────────────────────────\n";
}

static void check(bool condition, const std::string &label) {
  std::cout << "  " << (condition ? "✓" : "✗") << "  " << label << "\n";
  if (!condition) {
    std::cerr << "  FAIL: " << label << "\n";
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Section 1: Invariant Verification
// Confirm that all three core invariants hold at the canonical coherent state
// through the SqrtNKernel API, before any ticks are executed.
// ══════════════════════════════════════════════════════════════════════════════
static bool section_invariant_verification() {
  print_section(1, "Invariant Verification");
  std::cout << "  Testing canonical-state invariants via SqrtNKernel API\n\n";

  bool all_ok = true;

  for (uint64_t N : {16ULL, 64ULL, 256ULL, 1024ULL}) {
    SqrtNKernel sk(N);

    bool inv = sk.verify_invariants();
    bool spec = sk.verify_spectral();
    double r = sk.state().radius();
    double G = sk.channel().G_eff();
    double R = sk.channel().R_eff();

    std::cout << "  N=" << std::setw(5) << N
              << "  √N=" << std::setw(4) << sk.sqrt_N()
              << "  r=" << std::fixed << std::setprecision(9) << r
              << "  G_eff=" << G
              << "  R_eff=" << R << "\n";

    check(inv, "N=" + std::to_string(N) + ": all_invariants()");
    check(spec, "N=" + std::to_string(N) + ": verify_spectral()");
    check(std::abs(r - 1.0) < 1e-9,
          "N=" + std::to_string(N) + ": r = 1.0 at coherent fixed point");
    check(std::abs(G - 1.0) < 1e-9,
          "N=" + std::to_string(N) + ": G_eff = 1.0 (ideal channel)");
    check(std::abs(R - 1.0) < 1e-9,
          "N=" + std::to_string(N) + ": R_eff = 1.0 (ideal channel)");
    all_ok = all_ok && inv && spec;
    std::cout << "\n";
  }
  return all_ok;
}

// ══════════════════════════════════════════════════════════════════════════════
// Section 2: Kernel Construction
// Build SqrtNKernel instances for a range of N values and confirm that:
//   - sqrt_N() = ⌈√N⌉
//   - speedup_metric() ≈ √N
//   - Pipeline invariants hold immediately after construction
// ══════════════════════════════════════════════════════════════════════════════
static bool section_kernel_construction() {
  print_section(2, "Kernel Construction");
  std::cout << "  Verifying N → √N mapping and speedup metric\n\n";

  std::cout << std::left
            << std::setw(10) << "N"
            << std::setw(10) << "ceil(√N)"
            << std::setw(14) << "speedup N/√N"
            << std::setw(14) << "ref √N"
            << "invariants\n";
  std::cout << std::string(58, '-') << "\n";

  bool all_ok = true;
  for (uint64_t N : {4ULL, 16ULL, 100ULL, 1000ULL, 10000ULL, 1000000ULL}) {
    SqrtNKernel sk(N);
    uint64_t expected_sqrt =
        static_cast<uint64_t>(std::ceil(std::sqrt(static_cast<double>(N))));
    double sp = sk.speedup_metric();
    double ref = std::sqrt(static_cast<double>(N));
    bool ok = sk.verify_invariants() && sk.sqrt_N() == expected_sqrt;
    all_ok = all_ok && ok;

    std::cout << std::setw(10) << N
              << std::setw(10) << sk.sqrt_N()
              << std::setw(14) << std::fixed << std::setprecision(4) << sp
              << std::setw(14) << ref
              << (ok ? "✓" : "✗") << "\n";
  }
  return all_ok;
}

// ══════════════════════════════════════════════════════════════════════════════
// Section 3: Phase Sweep
// Demonstrate PalindromePrecession::phasor_at(k · √N) at each accumulation
// step and confirm:
//   - |P_k| = 1 (unit-circle invariant preserved at every step)
//   - last_phase_phasor() after run() equals phasor_at(√N · √N)
// ══════════════════════════════════════════════════════════════════════════════
static bool section_phase_sweep() {
  print_section(3, "Phase Sweep");
  std::cout
      << "  Verifying |phasor_at(k·√N)| = 1 at each accumulation step\n\n";

  constexpr uint64_t N = 64; // √N = 8 steps — easy to print in full
  SqrtNKernel sk(N);

  std::cout << "  N=" << N << "  sqrt_N=" << sk.sqrt_N() << "\n\n";
  std::cout << std::setw(6) << "k"
            << std::setw(14) << "scaled_step"
            << std::setw(14) << "Re(P_k)"
            << std::setw(14) << "Im(P_k)"
            << std::setw(14) << "|P_k|"
            << "  unit-circle\n";
  std::cout << std::string(66, '-') << "\n";

  bool all_unit = true;
  uint64_t sn = sk.sqrt_N();
  for (uint64_t k = 1; k <= sn; ++k) {
    auto phasor = PalindromePrecession::phasor_at(k * sn);
    double mag = std::abs(phasor);
    bool ok = std::abs(mag - 1.0) < 1e-12;
    all_unit = all_unit && ok;
    std::cout << std::setw(6) << k
              << std::setw(14) << (k * sn)
              << std::setw(14) << std::fixed << std::setprecision(9)
              << phasor.real()
              << std::setw(14) << phasor.imag()
              << std::setw(14) << mag
              << "  " << (ok ? "✓" : "✗") << "\n";
  }

  // Run the kernel and verify the stored last_phase_phasor
  sk.run();
  auto expected_last =
      PalindromePrecession::phasor_at(sn * sn);
  auto actual_last = sk.last_phase_phasor();
  bool phasor_match =
      std::abs(actual_last - expected_last) < 1e-12;

  std::cout << "\n";
  check(all_unit, "all scaled phasors satisfy |P_k| = 1 (unit-circle)");
  check(phasor_match,
        "last_phase_phasor() == phasor_at(√N · √N) after run()");

  return all_unit && phasor_match;
}

// ══════════════════════════════════════════════════════════════════════════════
// Section 4: G_eff Accumulation
// Run the O(√N) sweep and inspect the weighted G_eff accumulation:
//   - accumulated_g_eff() ≈ 1.0 for a coherent state (r = 1, λ = 0)
//   - Each channel().G_eff() = sech(λ) = 1.0 when r = 1
//   - Invariants still hold after run()
// ══════════════════════════════════════════════════════════════════════════════
static bool section_g_eff_accumulation() {
  print_section(4, "G_eff Accumulation");
  std::cout << "  Weighted G_eff per step via channel() — ideal value = 1.0\n\n";

  std::cout << std::setw(10) << "N"
            << std::setw(10) << "sqrt_N"
            << std::setw(18) << "accumulated_g_eff"
            << std::setw(14) << "|g_eff - 1|"
            << "  invariants\n";
  std::cout << std::string(62, '-') << "\n";

  bool all_ok = true;
  for (uint64_t N : {16ULL, 64ULL, 256ULL, 1024ULL, 4096ULL}) {
    SqrtNKernel sk(N);
    sk.run();

    double g = sk.accumulated_g_eff();
    double err = std::abs(g - 1.0);
    bool inv_ok = sk.verify_invariants();
    bool g_ok = err < 1e-9;
    bool ok = inv_ok && g_ok;
    all_ok = all_ok && ok;

    std::cout << std::setw(10) << N
              << std::setw(10) << sk.sqrt_N()
              << std::setw(18) << std::fixed << std::setprecision(12) << g
              << std::setw(14) << std::scientific << std::setprecision(3) << err
              << std::fixed
              << "  " << (ok ? "✓" : "✗") << "\n";
  }
  return all_ok;
}

// ══════════════════════════════════════════════════════════════════════════════
// Section 5: Classical Speedup Metric
// Compare the O(N) baseline cost (N pipeline ticks) against the O(√N) kernel
// cost (⌈√N⌉ ticks).  For each N:
//   - baseline_ticks  = N
//   - kernel_ticks    = ⌈√N⌉
//   - speedup_actual  = baseline_ticks / kernel_ticks ≈ √N
//   - speedup_theory  = √N
//   - Both runs maintain invariants and G_eff ≈ 1
// ══════════════════════════════════════════════════════════════════════════════
static bool section_classical_speedup() {
  print_section(5, "Classical Speedup Metric");
  std::cout
      << "  O(N) baseline vs O(√N) kernel — quantitative speedup comparison\n\n";

  std::cout << std::setw(10) << "N"
            << std::setw(12) << "baseline(N)"
            << std::setw(12) << "kernel(√N)"
            << std::setw(14) << "speedup N/√N"
            << std::setw(14) << "theory √N"
            << std::setw(12) << "|Δspeedup|"
            << "  ok\n";
  std::cout << std::string(76, '-') << "\n";

  bool all_ok = true;
  for (uint64_t N : {4ULL, 16ULL, 100ULL, 256ULL, 1024ULL, 10000ULL}) {
    // O(N) baseline: run a fresh pipeline for exactly N ticks
    Pipeline baseline = Pipeline::create(KernelMode::FULL);
    baseline.run(N);
    bool baseline_ok = baseline.verify_invariants();

    // O(√N) kernel: run the SqrtNKernel
    SqrtNKernel sk(N);
    sk.run();
    bool kernel_ok = sk.verify_invariants();

    double sp_actual = sk.speedup_metric();           // N / √N
    double sp_theory = std::sqrt(static_cast<double>(N));
    double sp_err = std::abs(sp_actual - sp_theory);

    // Speedup error should be at most 1 (integer ceiling artefact)
    bool sp_ok = sp_err <= 1.0 + 1e-9;
    bool ok = baseline_ok && kernel_ok && sp_ok;
    all_ok = all_ok && ok;

    std::cout << std::setw(10) << N
              << std::setw(12) << N
              << std::setw(12) << sk.sqrt_N()
              << std::setw(14) << std::fixed << std::setprecision(4) << sp_actual
              << std::setw(14) << sp_theory
              << std::setw(12) << std::setprecision(6) << sp_err
              << "  " << (ok ? "✓" : "✗") << "\n";
  }

  std::cout << "\n  Speedup error ≤ 1.0 is expected: the ceiling ⌈√N⌉ adds at\n"
            << "  most one extra tick compared to the exact √N.\n";
  return all_ok;
}

// ══════════════════════════════════════════════════════════════════════════════
// Section 6: Cross-Validation
// Validate G_eff = sech(λ) = sech(ln(r)) at each tick against the theoretical
// formula.  For the canonical state r = 1 → λ = 0 → sech(0) = 1.
// Also validates that channel().G_eff() * channel().R_eff() = 1 (duality).
// ══════════════════════════════════════════════════════════════════════════════
static bool section_cross_validation() {
  print_section(6, "Cross-Validation");
  std::cout
      << "  G_eff = sech(λ) and G_eff·R_eff = 1 at every step of the sweep\n\n";

  constexpr uint64_t N = 256;
  SqrtNKernel sk(N, KernelMode::FULL);

  std::cout << "  N=" << N << "  sqrt_N=" << sk.sqrt_N() << "\n\n";
  std::cout << std::setw(6) << "step"
            << std::setw(14) << "r=|β/α|"
            << std::setw(14) << "λ=ln(r)"
            << std::setw(14) << "G_eff"
            << std::setw(14) << "sech(λ)"
            << std::setw(14) << "G·R"
            << "  ok\n";
  std::cout << std::string(82, '-') << "\n";

  bool all_ok = true;
  uint64_t sn = sk.sqrt_N();

  // Rebuild step-by-step to capture intermediate channels
  Pipeline step_pl = Pipeline::create(KernelMode::FULL);
  for (uint64_t k = 1; k <= sn; ++k) {
    step_pl.tick();
    auto ch = step_pl.channel();
    double r = step_pl.state().radius();
    double lam = ch.lambda;
    double G = ch.G_eff();
    double R = ch.R_eff();
    double sech_lam = 1.0 / std::cosh(lam);
    double GR = G * R;

    bool row_ok =
        std::abs(G - sech_lam) < 1e-12 && std::abs(GR - 1.0) < 1e-12;
    all_ok = all_ok && row_ok;

    if (k <= 8 || k == sn) { // print first 8 and last row
      std::cout << std::setw(6) << k
                << std::setw(14) << std::fixed << std::setprecision(9) << r
                << std::setw(14) << lam
                << std::setw(14) << G
                << std::setw(14) << sech_lam
                << std::setw(14) << GR
                << "  " << (row_ok ? "✓" : "✗") << "\n";
    } else if (k == 9) {
      std::cout << "  ... (" << (sn - 9) << " rows omitted) ...\n";
    }
  }

  // Also run the full SqrtNKernel and compare accumulated_g_eff vs theory
  sk.run();
  double g_acc = sk.accumulated_g_eff();
  double g_theory = 1.0; // sech(0) = 1 for ideal coherent state
  bool g_match = std::abs(g_acc - g_theory) < 1e-9;
  all_ok = all_ok && g_match;

  std::cout << "\n";
  check(all_ok, "G_eff = sech(λ) at every accumulation step");
  check(g_match,
        "accumulated_g_eff() = " + std::to_string(g_acc) +
            " ≈ sech(0) = 1.0 (theory)");

  return all_ok;
}

// ══════════════════════════════════════════════════════════════════════════════
// Main
// ══════════════════════════════════════════════════════════════════════════════
int main() {
  print_banner("O(\u221aN) Classical Hardware Demonstration");
  std::cout << "  SqrtNKernel: KernelPipeline + PalindromePrecession\n";
  std::cout << "               + SpectralBridge + KernelState\n";

  bool ok1 = section_invariant_verification();
  bool ok2 = section_kernel_construction();
  bool ok3 = section_phase_sweep();
  bool ok4 = section_g_eff_accumulation();
  bool ok5 = section_classical_speedup();
  bool ok6 = section_cross_validation();

  bool all_passed = ok1 && ok2 && ok3 && ok4 && ok5 && ok6;

  std::cout
      << "\n╔══════════════════════════════════════════════════════════╗\n";
  std::cout << "║  Demo Summary                                            ║\n";
  std::cout << "╠══════════════════════════════════════════════════════════╣\n";
  std::cout << "║  Section 1 — Invariant Verification   "
            << (ok1 ? "✓ PASS" : "✗ FAIL") << "             ║\n";
  std::cout << "║  Section 2 — Kernel Construction      "
            << (ok2 ? "✓ PASS" : "✗ FAIL") << "             ║\n";
  std::cout << "║  Section 3 — Phase Sweep              "
            << (ok3 ? "✓ PASS" : "✗ FAIL") << "             ║\n";
  std::cout << "║  Section 4 — G_eff Accumulation       "
            << (ok4 ? "✓ PASS" : "✗ FAIL") << "             ║\n";
  std::cout << "║  Section 5 — Classical Speedup Metric "
            << (ok5 ? "✓ PASS" : "✗ FAIL") << "             ║\n";
  std::cout << "║  Section 6 — Cross-Validation         "
            << (ok6 ? "✓ PASS" : "✗ FAIL") << "             ║\n";
  std::cout
      << "╚══════════════════════════════════════════════════════════╝\n";

  if (all_passed) {
    std::cout << "\n  ✓ ALL SECTIONS PASSED — O(√N) kernel coherent end-to-end\n\n";
    return 0;
  } else {
    std::cout << "\n  ✗ SOME SECTIONS FAILED — check output above\n\n";
    return 1;
  }
}
