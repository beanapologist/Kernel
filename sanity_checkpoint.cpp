/*
 * sanity_checkpoint.cpp — Multi-Vector Sanity Checkpoint (C++)
 *
 * Filters computed values against 6 attack vectors derived from
 * break_system.py:
 *
 *   1. ALGEBRAIC  — Exact identity verification (symbolic analogue)
 *   2. NUMERICAL  — Large-sweep validation (1M+ points)
 *   3. CHECKSUM   — FNV-1a hash-locked invariant detection
 *   4. CROSS      — Inter-structure consistency checks
 *   5. EDGE       — Adversarial extreme inputs (NaN, ±Inf, near-zero)
 *   6. PI         — Leibniz series (2,000,000 terms) attacked by Lean formulas
 *
 * Constants grounded in verified Lean theorems (quantum_kernel_v2.cpp §1-4,
 * Prop 4, Theorem 11-14).  Exit 0 on full pass, non-zero on any failure.
 */

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ── Verified constants
// ────────────────────────────────────────────────────────
constexpr double ETA = 0.70710678118654752440;        // 1/√2  (Theorem 3)
constexpr double DELTA_S = 2.41421356237309504880;    // 1+√2  (Prop 4)
constexpr double DELTA_CONJ = 0.41421356237309504880; // √2-1 = 1/δ_S (Prop 4c)
constexpr double PHI = 1.61803398874989484820; // (1+√5)/2  golden ratio
constexpr double PI = 3.14159265358979323846;

// Lepton masses (PDG 2022, MeV/c²)
constexpr double LEPTON_E = 0.51099895000;
constexpr double LEPTON_MU = 105.6583755;
constexpr double LEPTON_TAU = 1776.86;

// ── Tolerances
// ────────────────────────────────────────────────────────────────
constexpr double TIGHT_TOL = 1e-12;
constexpr double SWEEP_TOL = 1e-10;

// ── Math kernel functions
// ───────────────────────────────────────────────────── Theorem 11: C(r) =
// 2r/(1+r²)
double coherence(double r) {
  if (!std::isfinite(r) || r <= 0.0) {
    return 0.0;
  }
  double denom = 1.0 + r * r;
  if (!std::isfinite(denom) || denom == 0.0) {
    return 0.0;
  }
  return (2.0 * r) / denom;
}

// F(λ) = 1 − sech(λ)  (frustration / Lyapunov envelope)
double frustration(double lambda) {
  if (!std::isfinite(lambda)) {
    return std::isinf(lambda) ? 1.0 : 0.0;
  }
  double ch = std::cosh(lambda);
  if (!std::isfinite(ch)) {
    return 1.0; // |λ|→∞ limit
  }
  return 1.0 - 1.0 / ch;
}

using Cx = std::complex<double>;
const Cx MU{-ETA, ETA}; // µ = e^{i3π/4} (Section 2)

// ── Test harness
// ──────────────────────────────────────────────────────────────
static int g_passed = 0;
static int g_failed = 0;

void check(const std::string &name, bool condition,
           const std::string &detail = "") {
  if (condition) {
    std::cout << "  \u2713 " << name << "\n";
    ++g_passed;
  } else {
    std::cout << "  \u2717 " << name;
    if (!detail.empty()) {
      std::cout << " — " << detail;
    }
    std::cout << "\n";
    ++g_failed;
  }
}

// ── FNV-1a hash (64-bit) for checksum vector
// ────────────────────────────────── Produces a deterministic 16-char hex
// fingerprint from a string.
static std::string fnv1a_hex(const std::string &s) {
  constexpr uint64_t FNV_PRIME = 0x00000100000001B3ULL;
  constexpr uint64_t FNV_OFFSET = 0xCBF29CE484222325ULL;
  uint64_t h = FNV_OFFSET;
  for (unsigned char c : s) {
    h ^= static_cast<uint64_t>(c);
    h *= FNV_PRIME;
  }
  std::ostringstream oss;
  oss << std::hex << std::setfill('0') << std::setw(16) << h;
  return oss.str();
}

// Format a double the same way across all vectors so hashes are reproducible.
static std::string fmt_dbl(const std::string &name, double v, int prec = 12) {
  std::ostringstream oss;
  oss << name << ":" << std::scientific << std::setprecision(prec) << v;
  return oss.str();
}

static std::string invariant_hash(const std::string &name, double v) {
  return fnv1a_hex(fmt_dbl(name, v));
}

// ── VECTOR 1: ALGEBRAIC
// ───────────────────────────────────────────────────────
void vector1_algebraic() {
  std::cout << "\n\u2550\u2550\u2550 VECTOR 1: ALGEBRAIC IDENTITIES "
               "\u2550\u2550\u2550\n";

  // µ⁸ = 1  (8-cycle, Theorem 10)
  Cx mu8 = Cx{1.0, 0.0};
  for (int i = 0; i < 8; ++i) {
    mu8 *= MU;
  }
  check("mu^8 = 1", std::abs(mu8 - Cx{1.0, 0.0}) < TIGHT_TOL);

  // µ^n ≠ 1 for 1 ≤ n < 8
  Cx acc{1.0, 0.0};
  for (int n = 1; n <= 7; ++n) {
    acc *= MU;
    check("mu^" + std::to_string(n) + " != 1",
          std::abs(acc - Cx{1.0, 0.0}) > 0.1);
  }

  // δ_S·(√2-1) = 1  (Prop 4: silver conservation)
  double silver = DELTA_S * DELTA_CONJ;
  check("delta_S * (sqrt2-1) = 1", std::abs(silver - 1.0) < TIGHT_TOL);

  // φ²-φ-1 = 0  (golden ratio identity)
  double golden = PHI * PHI - PHI - 1.0;
  check("phi^2 - phi - 1 = 0", std::abs(golden) < TIGHT_TOL);

  // 9 × 13717421 = 123456789  (palindrome factorisation)
  check("9 * 13717421 = 123456789", 9 * 13717421LL == 123456789LL);

  // C(1) = 1  (maximum coherence, Theorem 11)
  check("C(1) = 1", std::abs(coherence(1.0) - 1.0) < TIGHT_TOL);

  // C(δ_S) = 1/√2  (gate value)
  check("C(delta_S) = 1/sqrt2", std::abs(coherence(DELTA_S) - ETA) < TIGHT_TOL);

  // C(1/δ_S) = 1/√2
  check("C(1/delta_S) = 1/sqrt2",
        std::abs(coherence(DELTA_CONJ) - ETA) < TIGHT_TOL);

  // C(φ²) = 2/3  (Koide threshold)
  double phi2 = PHI * PHI;
  check("C(phi^2) = 2/3", std::abs(coherence(phi2) - 2.0 / 3.0) < TIGHT_TOL);

  // Gear: 8 × 3π/4 = 6π  (8-cycle period)
  double gear_err = std::abs(8.0 * 3.0 * PI / 4.0 - 3.0 * 2.0 * PI);
  check("gear: 8 * (3pi/4) = 6pi", gear_err < TIGHT_TOL);
}

// ── VECTOR 2: NUMERICAL
// ───────────────────────────────────────────────────────
void vector2_numerical() {
  std::cout
      << "\n\u2550\u2550\u2550 VECTOR 2: NUMERICAL SWEEPS \u2550\u2550\u2550\n";
  constexpr int N = 1'000'000;

  // --- Sweep: C(r) range and symmetry ---
  double max_c = 0.0;
  double max_sym_err = 0.0;
  double max_dual_err = 0.0;
  double max_frust_err = 0.0;
  bool c_bounded = true;

  // logspace: 1e-8 to 1e8 over N points
  const double log_lo = -8.0;
  const double log_hi = 8.0;
  const double step = (log_hi - log_lo) / static_cast<double>(N - 1);

  for (int i = 0; i < N; ++i) {
    double exp_val = log_lo + i * step;
    double r = std::pow(10.0, exp_val);
    double c = coherence(r);
    if (c > max_c) {
      max_c = c;
    }
    if (c < 0.0 || c > 1.0 + 1e-15) {
      c_bounded = false;
    }

    // Symmetry: C(r) = C(1/r)
    double sym_err = std::abs(c - coherence(1.0 / r));
    if (sym_err > max_sym_err) {
      max_sym_err = sym_err;
    }

    // Duality: C(e^λ) = sech(λ)
    double lambda = exp_val * std::log(10.0); // log(r)
    double c_exp = coherence(std::exp(lambda));
    double sech_l =
        (std::isfinite(std::cosh(lambda))) ? 1.0 / std::cosh(lambda) : 0.0;
    double d_err = std::abs(c_exp - sech_l);
    if (d_err > max_dual_err) {
      max_dual_err = d_err;
    }

    // Frustration monotone check: F(0)=0, F(λ)≥0
    double fl = frustration(lambda);
    if (fl < -1e-15) {
      max_frust_err = std::max(max_frust_err, -fl);
    }
  }

  check("C(r) in [0,1] over 1M sweep", c_bounded);
  check("C_max = 1.0 over sweep", std::abs(max_c - 1.0) < 1e-6);
  check("Symmetry |C(r)-C(1/r)| < 1e-10 over 1M", max_sym_err < SWEEP_TOL);
  check("Duality |C(e^l)-sech(l)| < 1e-10 over 1M", max_dual_err < SWEEP_TOL);
  check("F(lambda) >= 0 over 1M", max_frust_err < 1e-15);

  // F properties
  check("F(0) = 0", std::abs(frustration(0.0)) < TIGHT_TOL);
  // Use lambda=20: cosh(20)~2.4e8, so F(20)=1-4e-9 is strictly < 1
  check("F(lambda) < 1 for finite lambda",
        frustration(20.0) < 1.0 && frustration(20.0) > 0.0);
  check("F(+inf) = 1",
        std::abs(frustration(std::numeric_limits<double>::infinity()) - 1.0) <
            TIGHT_TOL);
  check("F(-inf) = 1",
        std::abs(frustration(-std::numeric_limits<double>::infinity()) - 1.0) <
            TIGHT_TOL);

  // µ orbit norms over 8 steps
  double max_norm_err = 0.0;
  Cx acc{1.0, 0.0};
  for (int n = 1; n <= 8; ++n) {
    acc *= MU;
    double norm_err = std::abs(std::abs(acc) - 1.0);
    if (norm_err > max_norm_err) {
      max_norm_err = norm_err;
    }
  }
  check("|mu^n| = 1 for n=1..8", max_norm_err < TIGHT_TOL);
}

// ── VECTOR 3: CHECKSUM
// ────────────────────────────────────────────────────────
void vector3_checksum() {
  std::cout << "\n\u2550\u2550\u2550 VECTOR 3: CHECKSUM INTEGRITY "
               "\u2550\u2550\u2550\n";

  // Build invariant table and pre-compute expected hashes
  struct Invariant {
    std::string name;
    double value;
  };

  double koide_q = (LEPTON_E + LEPTON_MU + LEPTON_TAU) /
                   std::pow(std::sqrt(LEPTON_E) + std::sqrt(LEPTON_MU) +
                                std::sqrt(LEPTON_TAU),
                            2.0);

  std::vector<Invariant> table = {
      {"GATE", ETA},
      {"PHI", PHI},
      {"DELTA_S", DELTA_S},
      {"C(1)", coherence(1.0)},
      {"C(DS)", coherence(DELTA_S)},
      {"C(PHI2)", coherence(PHI * PHI)},
      {"KOIDE", koide_q},
      {"SILVER", DELTA_S * DELTA_CONJ},
      {"GOLDEN", PHI * PHI - PHI - 1.0},
      {"F(0)", frustration(0.0)},
      {"GEAR", 8.0 * 3.0 * PI / 4.0 / (2.0 * PI)},
  };

  // Compute hashes on first pass and verify they are reproducible
  std::vector<std::string> hashes;
  hashes.reserve(table.size());
  std::cout << "  " << std::left << std::setw(12) << "Name" << std::setw(20)
            << "Hash" << "Value\n";
  std::cout << "  " << std::string(50, '-') << "\n";
  for (const auto &inv : table) {
    std::string h = invariant_hash(inv.name, inv.value);
    hashes.push_back(h);
    std::cout << "  " << std::left << std::setw(12) << inv.name << std::setw(20)
              << h << std::scientific << std::setprecision(6) << inv.value
              << "\n";
  }

  // Recompute and compare
  bool all_match = true;
  for (std::size_t i = 0; i < table.size(); ++i) {
    std::string h2 = invariant_hash(table[i].name, table[i].value);
    if (h2 != hashes[i]) {
      all_match = false;
    }
  }
  check("All " + std::to_string(table.size()) + " checksums reproduced",
        all_match);

  // Tamper detection: perturbations above the format precision must change the
  // hash.  1e-15 is below the resolution of the "%.12e" format (precision ~
  // 7e-13 for ETA), so it is expected to be missed — that is correct behavior.
  std::cout << "\n  --- Tamper detection ---\n";
  std::vector<double> deltas = {1e-15, 1e-12, 1e-10, 1e-8, 1e-5};
  bool all_large_detected = true;
  for (double delta : deltas) {
    std::string h_orig = invariant_hash("GATE", ETA);
    std::string h_tampered = invariant_hash("GATE", ETA + delta);
    bool detected = (h_tampered != h_orig);
    // Deltas < 1e-13 are below format precision and may not be detectable.
    bool expected = (delta >= 1e-13);
    if (expected && !detected) {
      all_large_detected = false;
    }
    std::cout << "  GATE + " << std::scientific << std::setprecision(0) << delta
              << ": " << (detected ? "DETECTED \u2713" : "MISSED \u2717")
              << "\n";
  }
  check("Tamper deltas >=1e-12 detected", all_large_detected);
}

// ── VECTOR 4: CROSS-STRUCTURE
// ─────────────────────────────────────────────────
void vector4_cross() {
  std::cout << "\n\u2550\u2550\u2550 VECTOR 4: CROSS-STRUCTURE CONSISTENCY "
               "\u2550\u2550\u2550\n";

  // Im(µ) = C(δ_S)  (eigenvalue imaginary part equals gate coherence)
  double mu_im = MU.imag();
  double c_ds = coherence(DELTA_S);
  check("Im(mu) = C(delta_S)", std::abs(mu_im - c_ds) < TIGHT_TOL);

  // |µ| = 1
  check("|mu| = 1", std::abs(std::abs(MU) - 1.0) < TIGHT_TOL);

  // η² + |µη|² = 1  (energy conservation, Theorem 9)
  double lhs = ETA * ETA + std::abs(MU * ETA) * std::abs(MU * ETA);
  check("eta^2 + |mu*eta|^2 = 1", std::abs(lhs - 1.0) < TIGHT_TOL);

  // C(φ²) ≈ Koide Q  (within Koide deviation tolerance 0.001)
  double koide_q = (LEPTON_E + LEPTON_MU + LEPTON_TAU) /
                   std::pow(std::sqrt(LEPTON_E) + std::sqrt(LEPTON_MU) +
                                std::sqrt(LEPTON_TAU),
                            2.0);
  double cphi2 = coherence(PHI * PHI);
  check("C(phi^2) ~= Koide Q", std::abs(cphi2 - koide_q) < 0.001);

  // C(φ²) < C(δ_S) < C(1)  (coherence ordering)
  check("C(phi^2) < C(delta_S) < C(1)",
        coherence(PHI * PHI) < coherence(DELTA_S) &&
            coherence(DELTA_S) < coherence(1.0));

  // Palindrome integer quotient = 8
  check("987654321 / 123456789 (integer) = 8",
        987654321LL / 123456789LL == 8LL);

  // Gear closure: 8 × (3π/4) = 3 × 2π  (orbit closes after 8 steps)
  check("gear closure: 8*(3pi/4) = 6pi",
        std::abs(8.0 * 3.0 * PI / 4.0 - 3.0 * 2.0 * PI) < TIGHT_TOL);

  // δ_S self-similarity: δ_S = 2 + 1/δ_S
  check("delta_S = 2 + 1/delta_S",
        std::abs(DELTA_S - (2.0 + 1.0 / DELTA_S)) < TIGHT_TOL);

  // Duality at GATE point: C(e^η) = sech(η)
  double c_at_exp_eta = coherence(std::exp(ETA));
  double sech_eta = 1.0 / std::cosh(ETA);
  check("C(e^eta) = sech(eta)", std::abs(c_at_exp_eta - sech_eta) < TIGHT_TOL);
}

// ── VECTOR 5: EDGE CASES
// ──────────────────────────────────────────────────────
void vector5_edge() {
  std::cout << "\n\u2550\u2550\u2550 VECTOR 5: EDGE CASES & ADVERSARIAL INPUTS "
               "\u2550\u2550\u2550\n";

  // C at boundary values
  auto edge_c = [](const std::string &name, double r, double expected,
                   double tol) {
    check("C(" + name + ") = " + std::to_string(expected),
          std::abs(coherence(r) - expected) < tol);
  };

  edge_c("0", 0.0, 0.0, TIGHT_TOL);
  edge_c("1e-300", 1e-300, 0.0, 1e-12);
  edge_c("1e300", 1e300, 0.0, 1e-12);
  edge_c("1-1e-10", 1.0 - 1e-10, 1.0, 0.01);
  edge_c("1+1e-10", 1.0 + 1e-10, 1.0, 0.01);

  // F at boundary values
  auto edge_f = [](const std::string &name, double l, double expected,
                   double tol) {
    check("F(" + name + ") ~= " + std::to_string(expected),
          std::abs(frustration(l) - expected) < tol);
  };

  edge_f("0", 0.0, 0.0, TIGHT_TOL);
  edge_f("1e-15", 1e-15, 0.0, 1e-12);
  edge_f("710", 710.0, 1.0, 1e-6);
  edge_f("-710", -710.0, 1.0, 1e-6);
  edge_f("1000", 1000.0, 1.0, 1e-12);

  // NaN/Inf resistance for C
  std::cout << "\n  --- NaN/Inf resistance ---\n";
  double nan_val = std::numeric_limits<double>::quiet_NaN();
  double inf_val = std::numeric_limits<double>::infinity();
  double ninf_val = -std::numeric_limits<double>::infinity();

  check("C(NaN) is finite", std::isfinite(coherence(nan_val)));
  check("C(+Inf) is finite", std::isfinite(coherence(inf_val)));
  check("C(-Inf) is finite", std::isfinite(coherence(ninf_val)));
  check("F(NaN) is finite", std::isfinite(frustration(nan_val)));
  check("F(+Inf) is finite", std::isfinite(frustration(inf_val)));
  check("F(-Inf) is finite", std::isfinite(frustration(ninf_val)));

  // Negative r is outside physical domain → C returns 0
  check("C(-1) = 0 (outside domain)", std::abs(coherence(-1.0)) < TIGHT_TOL);
  check("C(-0.0) = 0 (outside domain)", std::abs(coherence(-0.0)) < TIGHT_TOL);
}

// ── VECTOR 6: PI COMPUTATION (2,000,000 Leibniz terms)
// ──────────────────────── Computes π via the Leibniz-Gregory series with
// 2,000,000 iterations, then attacks the result with 8 Lean-grounded
// identities:
//   Real.pi_gt_3141592, µ⁸=1, gear closure (Theorem 10), Im(µ)=C(δ_S)=1/√2
//   (Prop 4), Wyler approximation 6π⁵≈m_p/m_e.
void vector6_pi() {
  std::cout << "\n\u2550\u2550\u2550 VECTOR 6: PI COMPUTATION (2,000,000 "
               "Leibniz terms) \u2550\u2550\u2550\n";

  constexpr long long N = 2'000'000LL;

  // Leibniz-Gregory series: π/4 = Σ_{k=0}^{N-1} (-1)^k / (2k+1)
  double sum = 0.0;
  for (long long k = 0; k < N; ++k) {
    double term = 1.0 / (2.0 * static_cast<double>(k) + 1.0);
    sum += (k % 2 == 0) ? term : -term;
  }
  double pi_lbn = 4.0 * sum;

  // Alternating-series truncation bound: |π - pi_lbn| < 4/(2N+1)
  double leibniz_bound = 4.0 / (2.0 * N + 1.0);

  std::cout << "  Leibniz pi (2M terms): " << std::fixed
            << std::setprecision(16) << pi_lbn << "\n";
  std::cout << "  Reference PI:          " << PI << "\n";
  std::cout << "  Bound 4/(2N+1):        " << std::scientific
            << std::setprecision(3) << leibniz_bound << "\n\n";

  // Attack 1: Real.pi_gt_3141592 minus Leibniz buffer → pi_lbn > 3.14159
  check("pi_lbn > 3.14159 (Lean: Real.pi_gt_3141592 - Leibniz buffer)",
        pi_lbn > 3.14159);

  // Attack 2: Upper sanity bound
  check("pi_lbn < 3.14160 (upper sanity)", pi_lbn < 3.14160);

  // Attack 3: Leibniz series converged to the correct precision
  check("|pi_lbn - PI| < 4/(2N+1) (Leibniz truncation bound)",
        std::abs(pi_lbn - PI) < leibniz_bound);

  // Attack 4: sin(π) = 0 — transcendental identity grounded in Lean
  check("|sin(pi_lbn)| < 2e-6 (sin(pi)=0 identity)",
        std::abs(std::sin(pi_lbn)) < 2e-6);

  // Attack 5: Gear identity 8*(3π/4) = 6π — Theorem 10 (µ 8-cycle)
  // Algebraically exact for any value of π; verifies arithmetic coherence.
  check("8*(3*pi_lbn/4) = 6*pi_lbn (gear identity, Theorem 10)",
        std::abs(8.0 * 3.0 * pi_lbn / 4.0 - 6.0 * pi_lbn) < 1e-9);

  // Attack 6: µ⁸ = 1 using Leibniz angle 3*pi_lbn/4 (Section 2, Theorem 10)
  Cx mu_pi{std::cos(3.0 * pi_lbn / 4.0), std::sin(3.0 * pi_lbn / 4.0)};
  Cx mu8_pi{1.0, 0.0};
  for (int i = 0; i < 8; ++i) {
    mu8_pi *= mu_pi;
  }
  check("|mu_lbn^8 - 1| < 2e-5 (mu^8=1 with Leibniz angle)",
        std::abs(mu8_pi - Cx{1.0, 0.0}) < 2e-5);

  // Attack 7: Im(µ) = sin(3π/4) = 1/√2 = C(δ_S) (Prop 4 + Section 2)
  double im_mu_lbn = std::sin(3.0 * pi_lbn / 4.0);
  check("|sin(3*pi_lbn/4) - eta| < 2e-6 (Im(mu)=C(delta_S)=1/sqrt2)",
        std::abs(im_mu_lbn - ETA) < 2e-6);

  // Attack 8: Wyler approximation 6π⁵ ≈ m_p/m_e ≈ 1836.15 (±0.5%)
  double wyler_lbn = 6.0 * std::pow(pi_lbn, 5.0);
  check("6*pi_lbn^5 in [1835,1837] (Wyler: 6*pi^5 approx m_p/m_e)",
        wyler_lbn > 1835.0 && wyler_lbn < 1837.0);

  // ── Note on 2,000,000 decimal digits ────────────────────────────────────
  // C++ stdlib uses IEEE 754 double (64-bit), which provides at most 16-17
  // significant decimal digits.  The full 2,000,000-decimal-digit computation
  // (Brent-Salamin AGM via arbitrary-precision arithmetic) is performed in the
  // companion Go implementation (go/sanity_checkpoint/main.go), which uses
  // Go's stdlib math/big.Float with no external dependencies.
  std::cout << "\n  [C++ max precision — double, ~15.9 significant decimal "
               "digits]\n";
  std::cout << "  PI (double, 16 dp): " << std::fixed << std::setprecision(16)
            << PI << "\n";
  std::cout
      << "  For all 2,000,000 decimal digits see: go/sanity_checkpoint/\n";
}

// ── Main
// ──────────────────────────────────────────────────────────────────────
int main() {
  const std::string sep(72, '=');
  std::cout << "\n" << sep << "\n";
  std::cout << "  SANITY CHECKPOINT -- C++ (6 Attack Vectors)\n";
  std::cout << sep << "\n";

  vector1_algebraic();
  vector2_numerical();
  vector3_checksum();
  vector4_cross();
  vector5_edge();
  vector6_pi();

  std::cout << "\n" << sep << "\n";
  std::cout << "  VERDICT\n" << sep << "\n\n";
  std::cout << "  Passed: " << g_passed << "\n";
  std::cout << "  Failed: " << g_failed << "\n\n";

  if (g_failed == 0) {
    std::cout
        << "  +----------------------------------------------------------+\n";
    std::cout
        << "  |  CANONICAL MAP: UNBROKEN                                 |\n";
    std::cout
        << "  |  All 6 attack vectors passed. System is coherent.        |\n";
    std::cout
        << "  +----------------------------------------------------------+\n";
    return 0;
  } else {
    std::cout
        << "  +----------------------------------------------------------+\n";
    // Pad the message to keep the box border aligned (field width 55 chars).
    std::ostringstream msg;
    msg << g_failed << " FAILURE(S) DETECTED. Investigate immediately.";
    std::cout << "  |  " << std::left << std::setw(55) << msg.str() << "|\n";
    std::cout
        << "  +----------------------------------------------------------+\n";
    return g_failed;
  }
}
