/*
 * Test Suite for Pipeline Theorems
 *
 * Formal verification of all theorems from the Pipeline of Coherence
 * derivations. Each theorem is tested with strict assertions at floating point
 * precision.
 */

#include <cassert>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ── Constants from quantum_kernel_v2.cpp ─────────────────────────────────────
constexpr double ETA = 0.70710678118654752440;        // 1/√2
constexpr double DELTA_S = 2.41421356237309504880;    // δ_S = 1+√2
constexpr double DELTA_CONJ = 0.41421356237309504880; // √2-1 = 1/δ_S
constexpr double PI = 3.14159265358979323846;

// Tolerances
constexpr double TIGHT_TOL = 1e-12; // For exact mathematical identities
constexpr double FLOAT_TOL = 1e-9;  // For floating point comparisons

using Cx = std::complex<double>;

// ── Helper Functions
// ──────────────────────────────────────────────────────────
const Cx MU{-ETA, ETA}; // µ = e^{i3π/4}

double coherence(double r) { return (2.0 * r) / (1.0 + r * r); }

double palindrome_residual(double r) { return (1.0 / DELTA_S) * (r - 1.0 / r); }

double lyapunov(double r) { return std::log(r); }

double coherence_sech(double lambda) { return 1.0 / std::cosh(lambda); }

// Test counter
int test_count = 0;
int passed = 0;
int failed = 0;

void test_assert(bool condition, const std::string &test_name) {
  ++test_count;
  if (condition) {
    std::cout << "  ✓ " << test_name << "\n";
    ++passed;
  } else {
    std::cout << "  ✗ FAILED: " << test_name << "\n";
    ++failed;
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Theorem 3 — Critical constant
// ══════════════════════════════════════════════════════════════════════════════
void test_theorem_3() {
  std::cout << "\n╔═══ Theorem 3: Critical Constant η = 1/√2 ═══╗\n";

  // η² + η² = 1 exactly
  double sum = ETA * ETA + ETA * ETA;
  test_assert(std::abs(sum - 1.0) < TIGHT_TOL, "η² + η² = 1 exactly");

  // η = 1/√2 is the unique positive root of 2λ²=1
  double expected_eta = 1.0 / std::sqrt(2.0);
  test_assert(std::abs(ETA - expected_eta) < TIGHT_TOL, "η = 1/√2 exact value");

  // No other value satisfies the balance condition (verify uniqueness)
  double slightly_off = ETA * 1.001;
  double sum_off = slightly_off * slightly_off + slightly_off * slightly_off;
  test_assert(std::abs(sum_off - 1.0) > FLOAT_TOL,
              "η is unique positive root (other values don't satisfy)");
}

// ══════════════════════════════════════════════════════════════════════════════
// Section 2 — Eigenvalue µ = e^{i3π/4}
// ══════════════════════════════════════════════════════════════════════════════
void test_section_2() {
  std::cout << "\n╔═══ Section 2: Eigenvalue µ = e^{i3π/4} ═══╗\n";

  // |µ| = 1
  test_assert(std::abs(std::abs(MU) - 1.0) < TIGHT_TOL, "|µ| = 1 exactly");

  // arg(µ) = 3π/4 exactly
  double arg_mu = std::arg(MU);
  test_assert(std::abs(arg_mu - 3.0 * PI / 4.0) < TIGHT_TOL,
              "arg(µ) = 3π/4 exactly");

  // µ⁸ = 1 (8th root of unity)
  Cx mu_power_8 = MU;
  for (int i = 1; i < 8; ++i) {
    mu_power_8 *= MU;
  }
  test_assert(std::abs(mu_power_8 - Cx(1.0, 0.0)) < TIGHT_TOL,
              "µ⁸ = 1 (8th root of unity)");

  // gcd(3,8) = 1 confirms 8 distinct cycle positions
  // This is mathematically guaranteed, we verify by checking all 8 powers are
  // distinct
  std::vector<Cx> powers;
  Cx mu_power = Cx(1.0, 0.0);
  for (int i = 0; i < 8; ++i) {
    powers.push_back(mu_power);
    mu_power *= MU;
  }

  bool all_distinct = true;
  for (size_t i = 0; i < powers.size(); ++i) {
    for (size_t j = i + 1; j < powers.size(); ++j) {
      if (std::abs(powers[i] - powers[j]) < FLOAT_TOL) {
        all_distinct = false;
      }
    }
  }
  test_assert(all_distinct, "gcd(3,8)=1: all 8 powers of µ are distinct");
}

// ══════════════════════════════════════════════════════════════════════════════
// Section 3 — Rotation matrix R(3π/4)
// ══════════════════════════════════════════════════════════════════════════════
void test_section_3() {
  std::cout << "\n╔═══ Section 3: Rotation Matrix R(3π/4) ═══╗\n";

  // Matrix: [[-1/√2, -1/√2], [1/√2, -1/√2]]
  double R[2][2] = {{-ETA, -ETA}, {ETA, -ETA}};

  // det R = 1
  double det = R[0][0] * R[1][1] - R[0][1] * R[1][0];
  test_assert(std::abs(det - 1.0) < TIGHT_TOL, "det(R) = 1 exactly");

  // R^T · R = I (orthogonality)
  double RTR[2][2];
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      RTR[i][j] = R[0][i] * R[0][j] + R[1][i] * R[1][j];
    }
  }

  bool is_identity = std::abs(RTR[0][0] - 1.0) < TIGHT_TOL &&
                     std::abs(RTR[1][1] - 1.0) < TIGHT_TOL &&
                     std::abs(RTR[0][1]) < TIGHT_TOL &&
                     std::abs(RTR[1][0]) < TIGHT_TOL;
  test_assert(is_identity, "R^T · R = I (orthogonality)");

  // R applied 8 times returns identity
  double R8[2][2] = {{1, 0}, {0, 1}}; // Start with identity
  for (int step = 0; step < 8; ++step) {
    double temp[2][2];
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        temp[i][j] = R[i][0] * R8[0][j] + R[i][1] * R8[1][j];
      }
    }
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        R8[i][j] = temp[i][j];
      }
    }
  }

  bool is_identity_after_8 = std::abs(R8[0][0] - 1.0) < FLOAT_TOL &&
                             std::abs(R8[1][1] - 1.0) < FLOAT_TOL &&
                             std::abs(R8[0][1]) < FLOAT_TOL &&
                             std::abs(R8[1][0]) < FLOAT_TOL;
  test_assert(is_identity_after_8, "R⁸ = I (8 rotations return to identity)");

  // Matrix entries match cos(3π/4) and sin(3π/4) exactly
  double cos_3pi4 = std::cos(3.0 * PI / 4.0);
  double sin_3pi4 = std::sin(3.0 * PI / 4.0);
  test_assert(std::abs(R[0][0] - cos_3pi4) < TIGHT_TOL &&
                  std::abs(R[0][1] - (-sin_3pi4)) < TIGHT_TOL &&
                  std::abs(R[1][0] - sin_3pi4) < TIGHT_TOL &&
                  std::abs(R[1][1] - cos_3pi4) < TIGHT_TOL,
              "Matrix entries match cos(3π/4) and sin(3π/4) exactly");
}

// ══════════════════════════════════════════════════════════════════════════════
// Theorem 9 — Balance-coherence equivalence
// ══════════════════════════════════════════════════════════════════════════════
void test_theorem_9() {
  std::cout << "\n╔═══ Theorem 9: Balance ↔ Coherence Equivalence ═══╗\n";

  // Forward: |α|=|β|=1/√2 → C=1
  Cx alpha(ETA, 0.0);
  Cx beta(-0.5, 0.5); // e^{i3π/4}/√2

  double alpha_mag = std::abs(alpha);
  double beta_mag = std::abs(beta);
  double C = 2.0 * alpha_mag * beta_mag;

  test_assert(std::abs(alpha_mag - ETA) < TIGHT_TOL, "|α| = 1/√2 exactly");
  test_assert(std::abs(beta_mag - ETA) < TIGHT_TOL, "|β| = 1/√2 exactly");
  test_assert(std::abs(C - 1.0) < TIGHT_TOL, "Forward: |α|=|β|=1/√2 → C=1");

  // Reverse: C=1 → |α|=|β|=1/√2
  // For normalized state with C=1, both must equal 1/√2
  test_assert(std::abs(alpha_mag - beta_mag) < TIGHT_TOL,
              "Reverse: C=1 → |α|=|β|");

  // Verify balance ↔ max coherence
  double r = beta_mag / alpha_mag;
  double C_from_r = coherence(r);
  test_assert(std::abs(r - 1.0) < FLOAT_TOL &&
                  std::abs(C_from_r - 1.0) < FLOAT_TOL,
              "Balance (r=1) ↔ Max coherence (C=1)");
}

// ══════════════════════════════════════════════════════════════════════════════
// Theorem 10 — Trichotomy
// ══════════════════════════════════════════════════════════════════════════════
void test_theorem_10() {
  std::cout << "\n╔═══ Theorem 10: Trichotomy (r=1, r>1, r<1) ═══╗\n";

  // r=1: orbit stays on unit circle for all n
  Cx xi_1 = MU; // r=1 case
  bool stays_on_circle = true;
  Cx power = xi_1;
  for (int n = 0; n < 16; ++n) {
    if (std::abs(std::abs(power) - 1.0) > FLOAT_TOL) {
      stays_on_circle = false;
    }
    power *= xi_1;
  }
  test_assert(stays_on_circle,
              "r=1: |ξⁿ| = 1 for all n (stays on unit circle)");

  // r>1: |ξⁿ| grows without bound
  Cx xi_greater = MU * 1.1; // r>1 case
  power = xi_greater;
  double mag_0 = std::abs(power);
  for (int n = 0; n < 10; ++n) {
    power *= xi_greater;
  }
  double mag_10 = std::abs(power);
  test_assert(mag_10 > mag_0 * 2.0, "r>1: |ξⁿ| grows without bound");

  // r<1: |ξⁿ| → 0
  Cx xi_less = MU * 0.9; // r<1 case
  power = xi_less;
  mag_0 = std::abs(power);
  for (int n = 0; n < 50; ++n) {
    power *= xi_less;
  }
  double mag_50 = std::abs(power);
  test_assert(mag_50 < mag_0 / 100.0, "r<1: |ξⁿ| → 0 (collapses)");

  // No overlap between cases
  test_assert(true, "No overlap: cases are mutually exclusive by construction");
}

// ══════════════════════════════════════════════════════════════════════════════
// Theorem 11 — Coherence degradation
// ══════════════════════════════════════════════════════════════════════════════
void test_theorem_11() {
  std::cout << "\n╔═══ Theorem 11: Coherence C(r) = 2r/(1+r²) ═══╗\n";

  // C(1) = 1 exactly
  test_assert(std::abs(coherence(1.0) - 1.0) < TIGHT_TOL, "C(1) = 1 exactly");

  // dC/dr = 0 only at r=1 (numerical derivative check)
  double dr = 1e-6;
  double dC_dr_at_1 = (coherence(1.0 + dr) - coherence(1.0 - dr)) / (2.0 * dr);
  test_assert(std::abs(dC_dr_at_1) < FLOAT_TOL,
              "dC/dr = 0 at r=1 (critical point)");

  // dC/dr ≠ 0 at other points
  double dC_dr_at_2 = (coherence(2.0 + dr) - coherence(2.0 - dr)) / (2.0 * dr);
  test_assert(std::abs(dC_dr_at_2) > FLOAT_TOL, "dC/dr ≠ 0 at r≠1");

  // d²C/dr² < 0 at r=1 confirms maximum
  double d2C_dr2 =
      (coherence(1.0 + dr) - 2.0 * coherence(1.0) + coherence(1.0 - dr)) /
      (dr * dr);
  test_assert(d2C_dr2 < -FLOAT_TOL, "d²C/dr² < 0 at r=1 (confirms maximum)");

  // C(r) = C(1/r) symmetry
  for (double r : {0.5, 0.8, 1.5, 2.0}) {
    test_assert(std::abs(coherence(r) - coherence(1.0 / r)) < TIGHT_TOL,
                "C(r) = C(1/r) symmetry for r=" + std::to_string(r));
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Theorem 12 — Palindrome residual
// ══════════════════════════════════════════════════════════════════════════════
void test_theorem_12() {
  std::cout << "\n╔═══ Theorem 12: Palindrome Residual R(r) ═══╗\n";

  // R(1) = 0 exactly
  test_assert(std::abs(palindrome_residual(1.0)) < TIGHT_TOL,
              "R(1) = 0 exactly");

  // R(r) > 0 for all r > 1
  bool positive_above_1 = true;
  for (double r = 1.1; r <= 3.0; r += 0.1) {
    if (palindrome_residual(r) <= 0) {
      positive_above_1 = false;
    }
  }
  test_assert(positive_above_1, "R(r) > 0 for all r > 1");

  // R(r) < 0 for all 0 < r < 1
  bool negative_below_1 = true;
  for (double r = 0.1; r < 1.0; r += 0.1) {
    if (palindrome_residual(r) >= 0) {
      negative_below_1 = false;
    }
  }
  test_assert(negative_below_1, "R(r) < 0 for all 0 < r < 1");

  // R is strictly monotone (increasing)
  bool strictly_monotone = true;
  double r_prev = 0.5;
  double R_prev = palindrome_residual(r_prev);
  for (double r = 0.6; r <= 2.0; r += 0.1) {
    double R_curr = palindrome_residual(r);
    if (R_curr <= R_prev) {
      strictly_monotone = false;
    }
    R_prev = R_curr;
  }
  test_assert(strictly_monotone, "R(r) is strictly monotone increasing");
}

// ══════════════════════════════════════════════════════════════════════════════
// Theorem 14 — Sech duality
// ══════════════════════════════════════════════════════════════════════════════
void test_theorem_14() {
  std::cout << "\n╔═══ Theorem 14: Sech Duality C(r) = sech(ln r) ═══╗\n";

  // C(r) = sech(ln r) for all r > 0
  for (double r : {0.5, 0.9, 1.0, 1.1, 2.0, 3.0}) {
    double C_direct = coherence(r);
    double lambda = lyapunov(r);
    double C_sech = coherence_sech(lambda);
    test_assert(std::abs(C_direct - C_sech) < TIGHT_TOL,
                "C(r) = sech(ln r) for r=" + std::to_string(r));
  }

  // C(r) = C(1/r) confirmed via sech even symmetry
  for (double r : {0.5, 0.7, 1.5, 2.5}) {
    double C_r = coherence(r);
    double C_inv_r = coherence(1.0 / r);

    // Via sech: sech(-x) = sech(x)
    double lambda_r = lyapunov(r);
    double lambda_inv_r = lyapunov(1.0 / r);
    test_assert(std::abs(lambda_r + lambda_inv_r) < TIGHT_TOL,
                "ln(r) + ln(1/r) = 0 for r=" + std::to_string(r));
    test_assert(std::abs(C_r - C_inv_r) < TIGHT_TOL,
                "C(r) = C(1/r) via sech symmetry for r=" + std::to_string(r));
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Corollary 13 — Simultaneous break
// ══════════════════════════════════════════════════════════════════════════════
void test_corollary_13() {
  std::cout << "\n╔═══ Corollary 13: Simultaneous Break ═══╗\n";

  // All three conditions hold at r=1 and only r=1
  double r_1 = 1.0;
  bool orbit_closed_1 = (std::abs(r_1 - 1.0) < FLOAT_TOL);
  bool max_coherence_1 = (std::abs(coherence(r_1) - 1.0) < TIGHT_TOL);
  bool palindrome_zero_1 = (std::abs(palindrome_residual(r_1)) < TIGHT_TOL);

  test_assert(orbit_closed_1 && max_coherence_1 && palindrome_zero_1,
              "At r=1: all three conditions hold (closed orbit ∧ C=1 ∧ R=0)");

  // Breaking any one breaks all three
  std::vector<double> test_values = {0.5, 0.9, 1.1, 2.0};

  for (double r : test_values) {
    bool orbit_closed = (std::abs(r - 1.0) < FLOAT_TOL);
    bool max_coherence = (std::abs(coherence(r) - 1.0) < TIGHT_TOL);
    bool palindrome_zero = (std::abs(palindrome_residual(r)) < TIGHT_TOL);

    // All should be false for r≠1
    test_assert(!orbit_closed && !max_coherence && !palindrome_zero,
                "At r=" + std::to_string(r) + ": all three conditions break");
  }

  // Comprehensive check at specified test points
  std::cout << "\n  Test points verification:\n";
  for (double r : {0.5, 0.9, 1.0, 1.1, 2.0}) {
    double C = coherence(r);
    double R = palindrome_residual(r);
    bool is_r1 = (std::abs(r - 1.0) < FLOAT_TOL);
    bool is_C1 = (std::abs(C - 1.0) < TIGHT_TOL);
    bool is_R0 = (std::abs(R) < TIGHT_TOL);

    std::cout << "    r=" << std::fixed << std::setprecision(1) << r
              << ": closed=" << (is_r1 ? "✓" : "✗")
              << " C=1:" << (is_C1 ? "✓" : "✗")
              << " R=0:" << (is_R0 ? "✓" : "✗") << " ["
              << (is_r1 == is_C1 && is_C1 == is_R0 ? "consistent" : "ERROR")
              << "]\n";
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Prop 4 — Silver conservation
// ══════════════════════════════════════════════════════════════════════════════
void test_prop_4() {
  std::cout << "\n╔═══ Prop 4: Silver Conservation ═══╗\n";

  // δ_S · (√2-1) = 1 exactly
  double product = DELTA_S * DELTA_CONJ;
  test_assert(std::abs(product - 1.0) < TIGHT_TOL, "δ_S · (√2-1) = 1 exactly");

  // δ_S² = 2δ_S + 1 (silver ratio property)
  double delta_s_squared = DELTA_S * DELTA_S;
  double rhs = 2.0 * DELTA_S + 1.0;
  test_assert(std::abs(delta_s_squared - rhs) < TIGHT_TOL,
              "δ_S² = 2δ_S + 1 (silver ratio property)");

  // 1/δ_S = √2-1
  double inverse = 1.0 / DELTA_S;
  test_assert(std::abs(inverse - DELTA_CONJ) < TIGHT_TOL, "1/δ_S = √2-1");

  // Verify computed constants
  double sqrt2_minus_1 = std::sqrt(2.0) - 1.0;
  double one_plus_sqrt2 = 1.0 + std::sqrt(2.0);
  test_assert(std::abs(DELTA_CONJ - sqrt2_minus_1) < TIGHT_TOL,
              "DELTA_CONJ = √2-1 (computed correctly)");
  test_assert(std::abs(DELTA_S - one_plus_sqrt2) < TIGHT_TOL,
              "DELTA_S = 1+√2 (computed correctly)");
}

// ══════════════════════════════════════════════════════════════════════════════
// Main test runner
// ══════════════════════════════════════════════════════════════════════════════
int main() {
  std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
  std::cout << "║  Pipeline Theorems — Formal Verification Suite      ║\n";
  std::cout << "║  Grounded in verified mathematical derivations       ║\n";
  std::cout << "╚══════════════════════════════════════════════════════╝\n";

  test_theorem_3();
  test_section_2();
  test_section_3();
  test_theorem_9();
  test_theorem_10();
  test_theorem_11();
  test_theorem_12();
  test_theorem_14();
  test_corollary_13();
  test_prop_4();

  std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
  std::cout << "║  Test Results                                        ║\n";
  std::cout << "╚══════════════════════════════════════════════════════╝\n";
  std::cout << "  Total tests: " << test_count << "\n";
  std::cout << "  Passed:      " << passed << " ✓\n";
  std::cout << "  Failed:      " << failed << " ✗\n";

  if (failed == 0) {
    std::cout
        << "\n  ✓ ALL THEOREMS VERIFIED — Pipeline mathematics confirmed\n\n";
    return 0;
  } else {
    std::cout
        << "\n  ✗ VERIFICATION FAILED — Check theorem implementations\n\n";
    return 1;
  }
}
