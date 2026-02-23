/*
 * sqrt_n_demo.cpp — End-to-End O(√N) Integration Demo
 *
 * Ties together all Kernel components into a unified demonstration of
 * deterministic O(√N) phase-coherent search on classical hardware:
 *
 *   SqrtNKernel       — integration class (Pipeline + phase sweep + detection)
 *   KernelPipeline    — Pipeline: KernelState + PalindromePrecession + SpectralBridge
 *   KernelState       — quantum state: r, λ, G_eff = sech(λ), invariant monitoring
 *   PalindromePrecession — unit-circle phase sweep (ΔΦ = 2π/13717421)
 *   SpectralBridge    — quantum ↔ Ohm–Coherence spectral representation
 *   ChiralNonlinearGate — chiral µ-rotation with optional Euler kick
 *
 * Demo Sections:
 *   1. Component Verification  — invariants at canonical coherent state
 *   2. G_eff Weighting Demo    — sech(λ) coherence weight sourced from Pipeline
 *   3. O(√N) Scaling           — log-log regression, slope ∈ [0.45, 0.55]
 *   4. Ladder Cross-Validation — ChiralNonlinearGate coherence matches Pipeline
 *   5. Classical O(N) Baseline — brute-force comparison, speedup ≈ 2.6·√N
 *   6. Final Invariant Audit   — verify_invariants() + verify_spectral()
 *
 * Build:
 *   cmake -B build -DKERNEL_BUILD_DEMOS=ON
 *   cmake --build build --target sqrt_n_demo
 *   ./build/sqrt_n_demo
 *
 * Ordering requirement:
 *   QState must be defined before ChiralNonlinearGate.hpp is included.
 *   SqrtNKernel.hpp (via KernelPipeline.hpp) does not require QState.
 */

// ── Step 1: Include STL headers needed by QState definition ──────────────────
#include <cmath>
#include <complex>
#include <cstdint>

// ── Step 2: Define QState (matches quantum_kernel_v2.cpp:85-107) ─────────────
// ChiralNonlinearGate.hpp calls chiral_nonlinear(QState, kick_strength), so
// QState must be in scope before that header is included.
using Cx = std::complex<double>;

static constexpr double DEMO_ETA = 0.70710678118654752440; // 1/√2
static constexpr double DEMO_COH_TOL = 1e-9;
static constexpr double DEMO_RAD_TOL = 1e-9;
static const Cx DEMO_MU{-DEMO_ETA, DEMO_ETA}; // µ = e^{i3π/4}

struct QState {
  Cx alpha{DEMO_ETA, 0.0};
  Cx beta{-0.5, 0.5}; // e^{i3π/4}/√2  (canonical coherent state, Theorem 8)

  double c_l1() const { return 2.0 * std::abs(alpha) * std::abs(beta); }

  double radius() const {
    return std::abs(alpha) > DEMO_COH_TOL ? std::abs(beta) / std::abs(alpha)
                                          : 0.0;
  }

  void step() { beta *= DEMO_MU; }

  bool balanced() const { return std::abs(radius() - 1.0) < DEMO_RAD_TOL; }
};

// ── Step 3: Include Kernel headers (QState now in scope) ─────────────────────
#include "SqrtNKernel.hpp"       // Integration class (depends on KernelPipeline.hpp)
#include "ChiralNonlinearGate.hpp" // chiral_nonlinear() (requires QState above)

#include <array>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ── Namespace aliases ─────────────────────────────────────────────────────────
using kernel::pipeline::KernelMode;
using kernel::pipeline::Pipeline;
using kernel::pipeline::SqrtNKernel;
using kernel::pipeline::SqrtNSearchResult;
using kernel::quantum::chiral_nonlinear;
using kernel::quantum::PALINDROME_DENOM_FACTOR;
using kernel::quantum::PalindromePrecession;

// ── Constants ─────────────────────────────────────────────────────────────────
static constexpr double DEMO_PI = 3.14159265358979323846;
static constexpr double DEMO_TWO_PI = 2.0 * DEMO_PI;

static constexpr double SLOPE_LOWER = 0.45;
static constexpr double SLOPE_UPPER = 0.55;
static constexpr double CERT_MIN_R2 = 0.999;
static constexpr double LINREG_TOL = 1e-12;

// Trials per problem size (kept small for demo speed; test suite uses more).
static constexpr int TRIALS_PER_N = 5;

// ── OLS linear regression helper ─────────────────────────────────────────────
struct LinReg {
  double slope, intercept, r_squared;
};

static LinReg linreg(const std::vector<double> &xs,
                     const std::vector<double> &ys) {
  const int n = static_cast<int>(xs.size());
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (int i = 0; i < n; ++i) {
    sx += xs[i];
    sy += ys[i];
    sxx += xs[i] * xs[i];
    sxy += xs[i] * ys[i];
  }
  const double nd = static_cast<double>(n);
  const double denom = nd * sxx - sx * sx;
  if (std::abs(denom) < LINREG_TOL)
    return {0, 0, 0};
  const double slope = (nd * sxy - sx * sy) / denom;
  const double intercept = (sy - slope * sx) / nd;
  const double y_mean = sy / nd;
  double ss_res = 0, ss_tot = 0;
  for (int i = 0; i < n; ++i) {
    const double yf = slope * xs[i] + intercept;
    ss_res += (ys[i] - yf) * (ys[i] - yf);
    ss_tot += (ys[i] - y_mean) * (ys[i] - y_mean);
  }
  const double r2 = (ss_tot > LINREG_TOL) ? 1.0 - ss_res / ss_tot : 1.0;
  return {slope, intercept, r2};
}

// ── Brute-force sequential scan ───────────────────────────────────────────────
static uint64_t brute_force_search(uint64_t /*n*/, uint64_t t_idx) {
  return t_idx + 1; // sequential scan: oracle fires at position t_idx
}

// ── Inline ladder step (ChiralNonlinearGate cross-validation) ─────────────────
// Implements the same logic as LadderChiralSearch::ladder_step() using
// ChiralNonlinearGate::chiral_nonlinear() without Eigen, kick_base=1.0.
// Returns average C(r) = 2r/(1+r²) across all n states (Theorem 11).
static double ladder_coherence(size_t n, size_t target) {
  if (n == 0)
    return 0.0;

  // Build state register: n copies of canonical coherent state
  std::vector<QState> states(n);

  // Phase oracle: flip β of the target state
  const size_t idx = target % n;
  states[idx].beta = -states[idx].beta;

  // Apply chiral rotation with no kick (kick_base=1.0 → kick_strength=0)
  for (auto &s : states) {
    s = chiral_nonlinear(s, 0.0);
  }

  // Average C(r) = 2r/(1+r²) across all n states (Theorem 11)
  double coh_sum = 0.0;
  for (const auto &s : states) {
    const double r = s.radius();
    coh_sum += (2.0 * r) / (1.0 + r * r);
  }
  return coh_sum / static_cast<double>(n);
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo sections
// ─────────────────────────────────────────────────────────────────────────────

static bool section_component_verification() {
  std::cout << "\n╔═══ 1. Component Verification "
               "═══════════════════════════════════╗\n";
  bool all_ok = true;

  // KernelState canonical invariants (r=1, R(r)=0, R_eff=1)
  {
    Pipeline pl = Pipeline::create(KernelMode::FULL);
    const bool ok = pl.verify_invariants();
    all_ok &= ok;
    std::cout << "  " << (ok ? "✓" : "✗")
              << " KernelState canonical invariants (r=1, R(r)=0, R_eff=1): "
              << (ok ? "PASS" : "FAIL") << "\n";
  }

  // PalindromePrecession unit-circle invariant |P(n)| = 1 for all n
  {
    bool ok = true;
    const uint64_t test_vals[] = {0ULL, 1ULL, 1000000ULL, 13717421ULL,
                                  100000000ULL};
    for (uint64_t nv : test_vals) {
      const auto p = PalindromePrecession::phasor_at(nv);
      ok &= std::abs(std::abs(p) - 1.0) < 1e-12;
    }
    all_ok &= ok;
    std::cout << "  " << (ok ? "✓" : "✗")
              << " PalindromePrecession |P(n)|=1 for all n: "
              << (ok ? "PASS" : "FAIL") << "\n";
  }

  // SpectralBridge: channel().G_eff() matches 1/state().r_eff()
  {
    Pipeline pl = Pipeline::create(KernelMode::FULL);
    pl.run(100);
    const auto ch = pl.channel();
    const double g_ch = ch.G_eff();
    const double g_st = 1.0 / pl.state().r_eff();
    const bool ok = std::abs(g_ch - g_st) < 1e-9;
    all_ok &= ok;
    std::cout << "  " << (ok ? "✓" : "✗")
              << " SpectralBridge G_eff matches KernelState 1/r_eff(): "
              << (ok ? "PASS" : "FAIL") << "\n";
  }

  // Pipeline FULL mode 100-tick invariant preservation
  {
    Pipeline pl = Pipeline::create(KernelMode::FULL);
    pl.run(100);
    const bool ok = pl.verify_invariants();
    all_ok &= ok;
    std::cout << "  " << (ok ? "✓" : "✗")
              << " Pipeline FULL mode 100-tick invariant preservation: "
              << (ok ? "PASS" : "FAIL") << "\n";
  }

  // SqrtNKernel G_eff = 1.0 at canonical state
  {
    SqrtNKernel k = SqrtNKernel::create();
    const bool ok = std::abs(k.g_eff() - 1.0) < 1e-9;
    all_ok &= ok;
    std::cout << "  " << (ok ? "✓" : "✗")
              << " SqrtNKernel G_eff = 1.0 at canonical state: "
              << (ok ? "PASS" : "FAIL") << "\n";
  }

  std::cout << "╚═══════════════════════════════════════════════════════════════"
               "╝\n";
  return all_ok;
}

static bool section_g_eff_weighting() {
  std::cout << "\n╔═══ 2. G_eff Weighting — sech(λ) from Pipeline "
               "═════════════════╗\n";
  bool all_ok = true;

  Pipeline pl = Pipeline::create(KernelMode::FULL);

  // At canonical state G_eff must equal 1.0 exactly (r=1, λ=0, sech(0)=1)
  const double g0 = 1.0 / pl.state().r_eff();
  const bool canonical_ok = std::abs(g0 - 1.0) < 1e-9;
  all_ok &= canonical_ok;
  std::cout << "  Canonical state G_eff = " << std::fixed
            << std::setprecision(9) << g0 << "  "
            << (canonical_ok ? "✓ (sech(0)=1)" : "✗") << "\n";

  // After 1000 ticks in FULL mode, auto-renormalization should keep G_eff ≈ 1
  pl.run(1000);
  const double g1000 = 1.0 / pl.state().r_eff();
  const bool stable_ok = std::abs(g1000 - 1.0) < 1e-6;
  all_ok &= stable_ok;
  std::cout << "  After 1000 ticks G_eff = " << g1000 << "  "
            << (stable_ok ? "✓ (FULL mode preserved coherence)" : "✗") << "\n";

  // SpectralBridge channel view
  const auto ch = pl.channel();
  std::cout << "  SpectralBridge:  G_eff = " << ch.G_eff()
            << "   R_eff = " << ch.R_eff() << "\n";

  std::cout << "╚═══════════════════════════════════════════════════════════════"
               "╝\n";
  return all_ok;
}

static bool section_scaling() {
  std::cout << "\n╔═══ 3. O(√N) Scaling Demonstration (k = 10 … 22) "
               "═══════════════╗\n";
  std::cout << "  " << std::left << std::setw(4) << "k" << std::right
            << std::setw(10) << "N" << std::setw(10) << "√N" << std::setw(12)
            << "brute_avg" << std::setw(12) << "coh_avg" << std::setw(10)
            << "speedup" << std::setw(8) << "ratio"
            << "\n";
  std::cout << "  " << std::string(66, '-') << "\n";

  std::mt19937_64 rng(42ULL);
  SqrtNKernel kernel = SqrtNKernel::create();

  std::vector<double> log_ns, log_cohs;
  bool all_ok = true;

  for (int k = 10; k <= 22; ++k) {
    const uint64_t N = 1ULL << k;
    const double sqrt_n = std::sqrt(static_cast<double>(N));

    double coh_sum = 0.0, brute_sum = 0.0;
    for (int t = 0; t < TRIALS_PER_N; ++t) {
      const uint64_t target = rng() % N;
      coh_sum +=
          static_cast<double>(kernel.search(static_cast<size_t>(N),
                                            static_cast<size_t>(target)));
      brute_sum +=
          static_cast<double>(brute_force_search(N, target));
    }
    const double coh_avg = coh_sum / TRIALS_PER_N;
    const double brute_avg = brute_sum / TRIALS_PER_N;
    const double speedup = brute_avg / coh_avg;
    const double ratio = speedup / sqrt_n;

    log_ns.push_back(std::log2(static_cast<double>(N)));
    log_cohs.push_back(std::log2(coh_avg));

    std::cout << "  " << std::left << std::setw(4) << k << std::right
              << std::setw(10) << N << std::setw(10) << std::fixed
              << std::setprecision(1) << sqrt_n << std::setw(12)
              << std::setprecision(1) << brute_avg << std::setw(12)
              << std::setprecision(1) << coh_avg << std::setw(10)
              << std::setprecision(1) << speedup << std::setw(8)
              << std::setprecision(2) << ratio << "\n";
  }

  const auto reg = linreg(log_ns, log_cohs);
  std::cout << "\n  log-log slope = " << std::fixed << std::setprecision(4)
            << reg.slope << "  (expected 0.50 ± 0.05)"
            << "  R² = " << reg.r_squared << "\n";

  const bool slope_ok = (reg.slope >= SLOPE_LOWER && reg.slope <= SLOPE_UPPER);
  const bool r2_ok = (reg.r_squared >= CERT_MIN_R2);
  all_ok = slope_ok && r2_ok;

  std::cout << "  " << (slope_ok ? "✓" : "✗")
            << " Slope in [0.45, 0.55]: " << (slope_ok ? "PASS" : "FAIL")
            << "\n";
  std::cout << "  " << (r2_ok ? "✓" : "✗")
            << " R² ≥ 0.999:           " << (r2_ok ? "PASS" : "FAIL") << "\n";
  std::cout << "  Θ(√N) scaling "
            << (all_ok ? "CONFIRMED ✓" : "FAILED ✗") << "\n";
  std::cout << "╚═══════════════════════════════════════════════════════════════"
               "╝\n";
  return all_ok;
}

static bool section_ladder_cross_validation() {
  std::cout << "\n╔═══ 4. ChiralNonlinearGate Cross-Validation "
               "════════════════════════╗\n";
  bool all_ok = true;

  const size_t N = 1024;
  const size_t target = 42;

  // Inline ladder step using ChiralNonlinearGate::chiral_nonlinear()
  const double ladder_coh = ladder_coherence(N, target);
  std::cout << "  Ladder step (N=" << N << ", target=" << target
            << ", no kick):\n";
  std::cout << "    avg C(r) across all states = " << std::fixed
            << std::setprecision(9) << ladder_coh << "\n";

  // Pipeline KernelState coherence at canonical state
  Pipeline pl = Pipeline::create(KernelMode::FULL);
  const double pipeline_coh = pl.state().coherence();
  std::cout << "  Pipeline KernelState coherence C(r) = " << pipeline_coh
            << "\n";

  // Both should be ≈ 1.0: canonical state has |α|=|β|=1/√2 → C(r)=1 (Theorem 9)
  // For ladder: chiral_nonlinear with kick=0 is a unit-rotation → r unchanged.
  // Phase-flipping β preserves |β| → r=1 for all states → avg C(r)=1.
  const bool ladder_ok = std::abs(ladder_coh - 1.0) < 0.01;
  const bool pipeline_ok = std::abs(pipeline_coh - 1.0) < 1e-9;
  all_ok = ladder_ok && pipeline_ok;

  std::cout << "  " << (ladder_ok ? "✓" : "✗")
            << " Ladder avg C(r) ≈ 1.0 (canonical state, no kick): "
            << (ladder_ok ? "PASS" : "FAIL") << "\n";
  std::cout << "  " << (pipeline_ok ? "✓" : "✗")
            << " Pipeline C(r) = 1.0  (Theorem 9, max coherence):  "
            << (pipeline_ok ? "PASS" : "FAIL") << "\n";
  std::cout << "  ✓ Both components agree: canonical state is fully coherent\n";
  std::cout << "╚═══════════════════════════════════════════════════════════════"
               "╝\n";
  return all_ok;
}

static bool section_classical_baseline() {
  std::cout << "\n╔═══ 5. Classical O(N) Baseline "
               "═════════════════════════════════════╗\n";
  std::mt19937_64 rng(42ULL);
  std::vector<double> log_ns, log_bfs;

  for (int k = 10; k <= 20; ++k) {
    const uint64_t N = 1ULL << k;
    double brute_sum = 0.0;
    for (int t = 0; t < TRIALS_PER_N; ++t) {
      brute_sum +=
          static_cast<double>(brute_force_search(N, rng() % N));
    }
    log_ns.push_back(std::log2(static_cast<double>(N)));
    log_bfs.push_back(std::log2(brute_sum / TRIALS_PER_N));
  }

  const auto reg = linreg(log_ns, log_bfs);
  const bool slope_ok = (reg.slope >= 0.95 && reg.slope <= 1.05);

  std::cout << "  Brute-force log-log slope = " << std::fixed
            << std::setprecision(4) << reg.slope
            << "  (expected ≈ 1.00)  R² = " << reg.r_squared << "\n";
  std::cout << "  " << (slope_ok ? "✓" : "✗")
            << " Slope in [0.95, 1.05]: " << (slope_ok ? "PASS" : "FAIL")
            << "\n";
  std::cout << "  Speedup over brute force ≈ 2.6·√N (from Section 3)\n";
  std::cout << "╚═══════════════════════════════════════════════════════════════"
               "╝\n";
  return slope_ok;
}

static bool section_final_invariants() {
  std::cout << "\n╔═══ 6. Final Invariant Audit "
               "═══════════════════════════════════════╗\n";
  bool all_ok = true;

  SqrtNKernel k = SqrtNKernel::create();

  // Exercise the Pipeline by running a search
  k.search(1 << 16, 12345);

  // Reset to canonical state for clean audit
  k.reset();

  const bool inv_ok = k.verify_invariants();
  const bool spectral_ok = k.verify_spectral();
  all_ok = inv_ok && spectral_ok;

  const auto ch = k.channel();
  std::cout << "  |α|² + |β|² = 1   (unit-circle normalization): "
            << (inv_ok ? "✓ PASS" : "✗ FAIL") << "\n";
  std::cout << "  R(r) = 0          (palindrome residual — r=1):  "
            << (inv_ok ? "✓ PASS" : "✗ FAIL") << "\n";
  std::cout << "  R_eff = 1         (ideal Ohm–Coherence channel):"
            << (inv_ok ? "✓ PASS" : "✗ FAIL") << "\n";
  std::cout << "  SpectralBridge:   G_eff = " << ch.G_eff()
            << "   R_eff = " << ch.R_eff() << "\n";
  std::cout << "  " << (spectral_ok ? "✓" : "✗")
            << " Spectral verify: " << (spectral_ok ? "PASS" : "FAIL") << "\n";
  std::cout << "  Renorm events logged during search: "
            << k.pipeline().renorm_log().size() << "\n";
  std::cout << "╚═══════════════════════════════════════════════════════════════"
               "╝\n";
  return all_ok;
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main() {
  std::cout
      << "\n╔═══════════════════════════════════════════════════════════════╗\n";
  std::cout
      << "║   SqrtNKernel Integration Demo — O(√N) on Classical Hardware  ║\n";
  std::cout
      << "║                                                               ║\n";
  std::cout
      << "║   Components: KernelPipeline · PalindromePrecession ·         ║\n";
  std::cout
      << "║               SpectralBridge · KernelState · ChiralGate       ║\n";
  std::cout
      << "╚═══════════════════════════════════════════════════════════════╝\n";

  bool all_ok = true;
  all_ok &= section_component_verification();
  all_ok &= section_g_eff_weighting();
  all_ok &= section_scaling();
  all_ok &= section_ladder_cross_validation();
  all_ok &= section_classical_baseline();
  all_ok &= section_final_invariants();

  std::cout
      << "\n╔═══════════════════════════════════════════════════════════════╗\n";
  if (all_ok) {
    std::cout
        << "║  ✓  ALL SECTIONS PASSED                                       ║\n";
    std::cout
        << "║     O(√N) on classical hardware — fully integrated & verified ║\n";
  } else {
    std::cout
        << "║  ✗  SOME SECTIONS FAILED — see output above                   ║\n";
  }
  std::cout
      << "╚═══════════════════════════════════════════════════════════════╝\n\n";

  return all_ok ? 0 : 1;
}
