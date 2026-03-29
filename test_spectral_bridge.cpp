/*
 * test_spectral_bridge.cpp — Test Suite for the Spectral Bridge Principle
 *
 * Validates all components of spectral_bridge.hpp:
 *
 *   1. Core spectral functions: spectral_lyapunov, spectral_weight,
 *      spectral_resistance
 *   2. SpectralEigenvalue: unit-circle constraint, weight, R_eff, evolution
 *   3. SpectralBridge: coherence envelope C(r), null slice, R̄_eff fixed point
 *   4. NullSliceBridge: µ₀ = e^{i3π/4}, (µ₀)^8 = 1, 8-cycle coherence
 *   5. Phase evolution: deterministic arg(r) = arg(1) + φ·log(r)
 *   6. Arithmetic periodicity: 8-step cycle closure and distinct powers
 *
 * Test style follows test_ohm_coherence.cpp and test_palindrome_precession.cpp.
 */

#include "spectral_bridge.hpp"

#include <cassert>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace kernel::spectral;

// ── Test framework (matches existing test files) ──────────────────────────────
static int test_count = 0;
static int passed     = 0;
static int failed     = 0;

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

// ── 1. Core spectral functions ────────────────────────────────────────────────
static void test_core_functions() {
    std::cout << "\n\u2554\u2550\u2550\u2550 Core Spectral Functions "
                 "\u2550\u2550\u2550\u2557\n";

    // spectral_lyapunov(1) = 0 (null slice condition)
    test_assert(std::abs(spectral_lyapunov(1.0)) < TIGHT_TOL,
                "spectral_lyapunov(1) = 0  (null slice)");

    // spectral_lyapunov is symmetric around 1 on the log scale
    test_assert(std::abs(spectral_lyapunov(2.0) - spectral_lyapunov(0.5)) < TIGHT_TOL,
                "spectral_lyapunov(2) = spectral_lyapunov(0.5)  (log symmetry)");

    // spectral_lyapunov(e) = 1
    test_assert(std::abs(spectral_lyapunov(std::exp(1.0)) - 1.0) < TIGHT_TOL,
                "spectral_lyapunov(e) = 1");

    // spectral_weight(1) = 1  (maximal weight on unit circle)
    test_assert(std::abs(spectral_weight(1.0) - 1.0) < TIGHT_TOL,
                "spectral_weight(1) = 1  (unit circle)");

    // spectral_weight is strictly < 1 away from unit circle
    test_assert(spectral_weight(0.5) < 1.0 - FLOAT_TOL,
                "spectral_weight(0.5) < 1");
    test_assert(spectral_weight(2.0) < 1.0 - FLOAT_TOL,
                "spectral_weight(2.0) < 1");

    // spectral_weight * spectral_resistance = 1  (sech × cosh = 1)
    for (double x : {0.5, 1.0, 2.0, 0.1, 3.0}) {
        double product = spectral_weight(x) * spectral_resistance(x);
        test_assert(std::abs(product - 1.0) < TIGHT_TOL,
                    "spectral_weight * spectral_resistance = 1 at x="
                    + std::to_string(x));
    }

    // Coherence fixed point: spectral_resistance(1) = 1
    test_assert(std::abs(spectral_resistance(1.0) - 1.0) < TIGHT_TOL,
                "spectral_resistance(1) = 1  (coherence fixed point)");

    // spectral_resistance(x) > 1 for x ≠ 1  (deviation from unit circle)
    test_assert(spectral_resistance(0.5) > 1.0 + FLOAT_TOL,
                "spectral_resistance(0.5) > 1");
    test_assert(spectral_resistance(2.0) > 1.0 + FLOAT_TOL,
                "spectral_resistance(2.0) > 1");
}

// ── 2. SpectralEigenvalue ─────────────────────────────────────────────────────
static void test_spectral_eigenvalue() {
    std::cout << "\n\u2554\u2550\u2550\u2550 SpectralEigenvalue "
                 "\u2550\u2550\u2550\u2557\n";

    // Unit-circle eigenvalue: |λ| = 1, arg = π/4
    SpectralEigenvalue uev(1.0, SBP_PI / 4.0);
    test_assert(uev.on_unit_circle(),
                "unit-circle eigenvalue: on_unit_circle() = true");
    test_assert(std::abs(uev.lyapunov()) < TIGHT_TOL,
                "unit-circle eigenvalue: lyapunov = 0");
    test_assert(std::abs(uev.weight() - 1.0) < TIGHT_TOL,
                "unit-circle eigenvalue: weight = 1");
    test_assert(std::abs(uev.R_eff() - 1.0) < TIGHT_TOL,
                "unit-circle eigenvalue: R_eff = 1  (coherence fixed point)");

    // Off-unit-circle eigenvalue: |λ| = 2
    SpectralEigenvalue off_ev(2.0, 0.0);
    test_assert(!off_ev.on_unit_circle(),
                "off-unit-circle eigenvalue: on_unit_circle() = false");
    test_assert(off_ev.lyapunov() > FLOAT_TOL,
                "off-unit-circle eigenvalue: lyapunov > 0");
    test_assert(off_ev.weight() < 1.0 - FLOAT_TOL,
                "off-unit-circle eigenvalue: weight < 1");
    test_assert(off_ev.R_eff() > 1.0 + FLOAT_TOL,
                "off-unit-circle eigenvalue: R_eff > 1");

    // Complex value: real = |λ|·cos(arg), imag = |λ|·sin(arg)
    SpectralEigenvalue cev(2.0, SBP_PI / 3.0);
    auto cv = cev.complex_value();
    test_assert(std::abs(cv.real() - 2.0 * std::cos(SBP_PI / 3.0)) < TIGHT_TOL,
                "complex_value real part");
    test_assert(std::abs(cv.imag() - 2.0 * std::sin(SBP_PI / 3.0)) < TIGHT_TOL,
                "complex_value imag part");

    // Phase evolution: at r=1, arg unchanged
    SpectralEigenvalue ev(1.0, SBP_PI / 6.0);
    SpectralEigenvalue ev_at_1 = ev.evolved(1.0, 1.0);
    test_assert(std::abs(ev_at_1.argument - ev.argument) < TIGHT_TOL,
                "evolved(r=1, phi=1): argument unchanged (log(1)=0)");
    test_assert(std::abs(ev_at_1.modulus - ev.modulus) < TIGHT_TOL,
                "evolved(r=1): modulus unchanged");

    // Phase evolution: at r=e, arg increased by phi
    SpectralEigenvalue ev2(1.0, 0.0);
    SpectralEigenvalue ev2_evolved = ev2.evolved(std::exp(1.0), SBP_PI / 4.0);
    test_assert(std::abs(ev2_evolved.argument - SBP_PI / 4.0) < FLOAT_TOL,
                "evolved(r=e, phi=pi/4): argument = phi  (log(e)=1)");
}

// ── 3. SpectralBridge ─────────────────────────────────────────────────────────
static void test_spectral_bridge() {
    std::cout << "\n\u2554\u2550\u2550\u2550 SpectralBridge (coherence envelope) "
                 "\u2550\u2550\u2550\u2557\n";

    // Single unit-circle eigenvalue: |C| = weight = 1
    std::vector<SpectralEigenvalue> single_evs = {SpectralEigenvalue(1.0, 0.0)};
    SpectralBridge single(single_evs);
    test_assert(single.is_null_slice(),
                "single unit-circle eigenvalue: is_null_slice = true");
    test_assert(std::abs(single.mean_R_eff() - 1.0) < TIGHT_TOL,
                "single unit-circle: mean_R_eff = 1  (coherence fixed point)");
    test_assert(single.validate_coherence_fixed_point(),
                "single unit-circle: validate_coherence_fixed_point = true");
    test_assert(std::abs(single.coherence_modulus() - 1.0) < TIGHT_TOL,
                "single unit-circle at arg=0: |C| = 1");

    // Off-unit-circle eigenvalue: not null slice, R_eff > 1
    std::vector<SpectralEigenvalue> off_evs = {SpectralEigenvalue(2.0, 0.0)};
    SpectralBridge off(off_evs);
    test_assert(!off.is_null_slice(),
                "single off-circle eigenvalue: is_null_slice = false");
    test_assert(off.mean_R_eff() > 1.0 + FLOAT_TOL,
                "single off-circle: mean_R_eff > 1");
    test_assert(!off.validate_coherence_fixed_point(),
                "single off-circle: validate_coherence_fixed_point = false");

    // Mixed bridge: one on circle, one off — not null slice
    std::vector<SpectralEigenvalue> mixed = {
        SpectralEigenvalue(1.0, 0.0),
        SpectralEigenvalue(0.5, SBP_PI / 2.0)
    };
    SpectralBridge mb(mixed);
    test_assert(!mb.is_null_slice(),
                "mixed bridge: is_null_slice = false");

    // Coherence sum: C = (1/N) Σ e^{i·arg} * f(|λ|)
    // For two eigenvalues at arg=0 and arg=π/2, each on unit circle:
    std::vector<SpectralEigenvalue> two_unit = {
        SpectralEigenvalue(1.0, 0.0),
        SpectralEigenvalue(1.0, SBP_PI / 2.0)
    };
    SpectralBridge tb(two_unit);
    auto cs = tb.coherence_sum();
    // C = (1/2)[(1+0i) + (0+1i)] = 0.5 + 0.5i
    test_assert(std::abs(cs.real() - 0.5) < TIGHT_TOL,
                "two unit-circle evs (0, pi/2): C.real = 0.5");
    test_assert(std::abs(cs.imag() - 0.5) < TIGHT_TOL,
                "two unit-circle evs (0, pi/2): C.imag = 0.5");
    test_assert(std::abs(tb.mean_R_eff() - 1.0) < TIGHT_TOL,
                "two unit-circle evs: mean_R_eff = 1");

    // Phase evolution of a bridge: args shift by phi*log(r), moduli unchanged
    std::vector<double> phis = {1.0, -1.0};
    double r = std::exp(1.0); // log(r) = 1
    SpectralBridge evolved = tb.evolved(r, phis);
    // First eigenvalue: arg was 0, phi=1, new arg = 0 + 1*1 = 1 rad (normalised)
    test_assert(std::abs(evolved.eigenvalues[0].argument - 1.0) < FLOAT_TOL,
                "evolved bridge: first eigenvalue arg shifted by phi*log(r)");
    // Moduli unchanged by evolution
    for (const auto &ev : evolved.eigenvalues)
        test_assert(std::abs(ev.modulus - 1.0) < TIGHT_TOL,
                    "evolved bridge: modulus unchanged");
}

// ── 4. NullSliceBridge: µ₀ = e^{i3π/4}, period-8 unitarity ──────────────────
static void test_null_slice_bridge() {
    std::cout << "\n\u2554\u2550\u2550\u2550 NullSliceBridge (\u00b5\u2080 = "
                 "e^{i3\u03c0/4}, 8-cycle) \u2550\u2550\u2550\u2557\n";

    // µ₀ = e^{i3π/4}: modulus = 1, argument = 3π/4
    SpectralEigenvalue mu0 = NullSliceBridge::mu0();
    test_assert(std::abs(mu0.modulus - 1.0) < TIGHT_TOL,
                "\u00b5\u2080: modulus = 1  (unit circle)");
    test_assert(std::abs(mu0.argument - 3.0 * SBP_PI / 4.0) < TIGHT_TOL,
                "\u00b5\u2080: argument = 3\u03c0/4");
    test_assert(mu0.on_unit_circle(),
                "\u00b5\u2080: on_unit_circle = true");
    test_assert(std::abs(mu0.R_eff() - 1.0) < TIGHT_TOL,
                "\u00b5\u2080: R_eff = 1  (coherence fixed point)");

    // Periodic unitarity: (µ₀)^8 = 1
    test_assert(NullSliceBridge::verify_periodicity(),
                "(\u00b5\u2080)^8 = 1  (periodic unitarity)");

    // Manual check: 8 × 3π/4 = 6π, and e^{i6π} = 1
    {
        double total = 8.0 * NullSliceBridge::MU0_ARG; // = 6π
        double expected = 3.0 * 2.0 * SBP_PI;           // 3 × 2π
        test_assert(std::abs(total - expected) < TIGHT_TOL,
                    "8 \u00d7 3\u03c0/4 = 6\u03c0 = 3 \u00d7 2\u03c0");
    }

    // µ₀^0 = 1 (argument = 0)
    SpectralEigenvalue mu0_0 = NullSliceBridge::mu0_power(0);
    test_assert(std::abs(mu0_0.argument) < TIGHT_TOL,
                "(\u00b5\u2080)^0: argument = 0");

    // µ₀^1 = µ₀ (argument = 3π/4)
    SpectralEigenvalue mu0_1 = NullSliceBridge::mu0_power(1);
    test_assert(std::abs(mu0_1.argument - 3.0 * SBP_PI / 4.0) < TIGHT_TOL,
                "(\u00b5\u2080)^1: argument = 3\u03c0/4");

    // µ₀^4 = e^{i3π} = −1 (argument = π or −π, both represent the same point)
    SpectralEigenvalue mu0_4 = NullSliceBridge::mu0_power(4);
    test_assert(std::abs(mu0_4.modulus - 1.0) < TIGHT_TOL,
                "(\u00b5\u2080)^4: modulus = 1");
    test_assert(std::abs(std::abs(mu0_4.argument) - SBP_PI) < FLOAT_TOL,
                "(\u00b5\u2080)^4: argument = \u00b1\u03c0  (= \u22121)");

    // µ₀^8 = 1 (argument = 0)
    SpectralEigenvalue mu0_8 = NullSliceBridge::mu0_power(8);
    test_assert(std::abs(mu0_8.argument) < FLOAT_TOL,
                "(\u00b5\u2080)^8: argument = 0  (cycle closure)");

    // All 8 powers are on the unit circle
    for (int k = 0; k < NullSliceBridge::PERIOD; ++k) {
        SpectralEigenvalue ev = NullSliceBridge::mu0_power(k);
        test_assert(ev.on_unit_circle(),
                    "(\u00b5\u2080)^" + std::to_string(k) +
                    ": on unit circle");
    }

    // Distinctness: all 8 powers have different arguments
    test_assert(NullSliceBridge::verify_distinct_powers(),
                "8 powers of \u00b5\u2080 are all distinct (all 8th roots covered)");
}

// ── 5. 8-cycle SpectralBridge coherence ───────────────────────────────────────
static void test_8cycle_bridge() {
    std::cout << "\n\u2554\u2550\u2550\u2550 8-cycle SpectralBridge coherence "
                 "\u2550\u2550\u2550\u2557\n";

    SpectralBridge bridge = NullSliceBridge::build_8cycle_bridge();

    // All eigenvalues on the unit circle → null slice
    test_assert(bridge.is_null_slice(),
                "8-cycle bridge: is_null_slice = true");

    // Coherence fixed point: R̄_eff = 1
    test_assert(std::abs(bridge.mean_R_eff() - 1.0) < TIGHT_TOL,
                "8-cycle bridge: mean_R_eff = 1  (R_eff(r=1) = 1)");
    test_assert(bridge.validate_coherence_fixed_point(),
                "8-cycle bridge: validate_coherence_fixed_point = true");

    // Mean Lyapunov = 0 for all-unit-circle eigenvalues
    test_assert(std::abs(bridge.mean_lyapunov()) < TIGHT_TOL,
                "8-cycle bridge: mean_lyapunov = 0");

    // Each eigenvalue has weight = 1  (f(1) = 1)
    for (const auto &ev : bridge.eigenvalues)
        test_assert(std::abs(ev.weight() - 1.0) < TIGHT_TOL,
                    "8-cycle bridge: each eigenvalue weight = 1");

    // Coherence sum: 8th roots of unity sum to 0
    // |C| = 0 for balanced / symmetric eigenvalue distribution
    auto cs = bridge.coherence_sum();
    test_assert(std::abs(cs.real()) < FLOAT_TOL,
                "8-cycle bridge: coherence_sum real part = 0 (balanced phases)");
    test_assert(std::abs(cs.imag()) < FLOAT_TOL,
                "8-cycle bridge: coherence_sum imag part = 0 (balanced phases)");
}

// ── 6. Phase evolution and arithmetic periodicity ─────────────────────────────
static void test_phase_evolution() {
    std::cout << "\n\u2554\u2550\u2550\u2550 Phase Evolution & Arithmetic "
                 "Periodicity \u2550\u2550\u2550\u2557\n";

    // Deterministic evolution: arg(r) = arg(1) + φ·log(r)
    // At r=1: arg unchanged (log(1) = 0)
    SpectralEigenvalue ev(1.0, SBP_PI / 4.0);
    SpectralEigenvalue ev_r1 = ev.evolved(1.0, 2.0);
    test_assert(std::abs(ev_r1.argument - SBP_PI / 4.0) < TIGHT_TOL,
                "phase evolution pinned at r=1: arg unchanged");

    // At r=e, phi=1: arg shifts by exactly 1
    SpectralEigenvalue ev_re = ev.evolved(std::exp(1.0), 1.0);
    double expected_arg = SBP_PI / 4.0 + 1.0;
    test_assert(std::abs(ev_re.argument - expected_arg) < FLOAT_TOL,
                "phase evolution at r=e, phi=1: arg shifted by 1");

    // Phase evolution is continuous at r=1 (limit agrees with value)
    double small_eps = 1e-8;
    SpectralEigenvalue ev_near1 = ev.evolved(1.0 + small_eps, 1.0);
    test_assert(std::abs(ev_near1.argument - (SBP_PI / 4.0 + std::log(1.0 + small_eps))) < 1e-6,
                "phase evolution continuous near r=1");

    // Arithmetic periodicity: applying same φ·log(r) twice = 2× shift
    SpectralEigenvalue ev_base(1.0, 0.0);
    double r_step = std::exp(SBP_PI / 4.0); // log(r_step) = π/4
    SpectralEigenvalue step1 = ev_base.evolved(r_step, 1.0);
    SpectralEigenvalue step2 = step1.evolved(r_step, 1.0); // cumulative: 2 × π/4
    // Note: evolved() works on the current argument, not cumulatively from r=1
    // so step2.argument = step1.argument + π/4 = π/4 + π/4 = π/2
    test_assert(std::abs(step2.argument - SBP_PI / 2.0) < FLOAT_TOL,
                "arithmetic periodicity: two equal phase steps double the shift");

    // µ₀ 8-cycle periodicity: argument after 8 steps = argument after 0 steps
    double running_arg = 0.0;
    for (int k = 0; k < NullSliceBridge::PERIOD; ++k)
        running_arg += NullSliceBridge::MU0_ARG;
    // running_arg = 8 × 3π/4 = 6π; normalised = 0
    double normalised = std::fmod(running_arg, 2.0 * SBP_PI);
    test_assert(normalised < FLOAT_TOL || std::abs(normalised - 2.0 * SBP_PI) < FLOAT_TOL,
                "8 × 3\u03c0/4 normalises to 0 mod 2\u03c0");

    // Coherence preservation: unit-circle eigenvalues stay on unit circle
    // after any number of phase evolutions (modulus is never changed by evolved())
    SpectralBridge bridge = NullSliceBridge::build_8cycle_bridge();
    std::vector<double> phis(8, 0.5);
    SpectralBridge evolved = bridge.evolved(2.0, phis);
    test_assert(evolved.is_null_slice(),
                "coherence preservation: unit-circle eigenvalues remain on unit circle after evolution");
    test_assert(std::abs(evolved.mean_R_eff() - 1.0) < TIGHT_TOL,
                "coherence preservation: R_eff stays at 1 after phase evolution");
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
    std::cout << "\u2551  Spectral Bridge Principle \u2014 Test Suite          "
                 "              \u2551\n";
    std::cout << "\u2551  Classical Invariance \u2194 Quantum Coherence via "
                 "Spectral Geometry   \u2551\n";
    std::cout
        << "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
           "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
           "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
           "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
           "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
           "\u255d\n";

    test_core_functions();
    test_spectral_eigenvalue();
    test_spectral_bridge();
    test_null_slice_bridge();
    test_8cycle_bridge();
    test_phase_evolution();

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
        std::cout << "\n  \u2713 ALL TESTS PASSED \u2014 Spectral Bridge "
                     "Principle verified\n\n";
        return 0;
    } else {
        std::cout << "\n  \u2717 TESTS FAILED \u2014 Check implementation\n\n";
        return 1;
    }
}
