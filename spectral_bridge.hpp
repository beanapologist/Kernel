/*
 * spectral_bridge.hpp — Spectral Bridge Principle
 *
 * Connects classical invariance and quantum coherence through spectral geometry.
 *
 * Core principle: When all spectral eigenvalues satisfy |λ_i| = 1 (null slice
 * r = 1), the system occupies the coherence fixed point R_eff(r=1) = 1,
 * bridging classical unitarity with quantum coherence via the Ohm–Coherence
 * duality (Theorem 14 in Pipeline of Coherence).
 *
 * Key components:
 *   SpectralEigenvalue  : complex eigenvalue with modulus and argument
 *   SpectralBridge      : N-eigenvalue spectral coherence envelope C(r)
 *   NullSliceBridge     : null slice with µ₀ = e^{i3π/4}, period-8 unitarity
 *
 * Mathematical foundation:
 *
 *   Phase weight function (consistent with Ohm–Coherence duality):
 *     f(|λ|) = sech(|log|λ||) = 1 / cosh(|log|λ||)
 *     f(|λ|) → 1  as |λ| → 1   (weight is maximal on the unit circle)
 *     f(|λ|) → 0  as |λ| → 0 or |λ| → ∞
 *
 *   Spectral Lyapunov exponent:
 *     Λ(|λ|) = |log|λ||         (deviation of |λ| from 1 on the log scale)
 *     Λ = 0  ⟺  |λ| = 1        (null slice condition)
 *
 *   Coherence envelope (complex spectral sum):
 *     C(r) = (1/N) Σ_i  e^{i·arg(λ_i(r))} · f(|λ_i(r)|)
 *
 *   Deterministic phase evolution with spectral parameter r:
 *     arg(λ_i(r)) = arg(λ_i(1)) + φ_i · log(r)
 *     At r = 1: log(1) = 0, so phases are pinned to their null-slice values.
 *
 *   Coherence fixed point (null slice, r = 1):
 *     |λ_i(1)| = 1  for all i  →  Λ_i = 0  →  R_eff = cosh(0) = 1
 *
 *   Null slice bridge (µ₀ = e^{i3π/4}):
 *     µ₀ is the 8-cycle balance primitive: (µ₀)^8 = e^{i6π} = 1
 *     Powers {µ₀^k : k = 0,…,7} trace all 8th roots of unity.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kernel::spectral {

// ── Mathematical constants ────────────────────────────────────────────────────
static constexpr double SBP_PI  = 3.14159265358979323846;
static constexpr double SBP_TOL = 1e-12;

// ── Core spectral functions ───────────────────────────────────────────────────

// Spectral Lyapunov exponent: Λ(x) = |log x|, x = |λ| > 0
// Measures the deviation of x from 1 on the log scale.
// Λ = 0 ⟺ x = 1 (null slice / unit circle condition).
inline double spectral_lyapunov(double modulus) {
    if (modulus <= 0.0)
        throw std::domain_error("spectral_lyapunov: modulus must be positive");
    return std::abs(std::log(modulus));
}

// Phase weight function: f(x) = sech(|log x|) = 1 / cosh(|log x|)
// Satisfies f(x) → 1 as x → 1, and f(x) → 0 as x → 0 or x → ∞.
// Consistent with the Ohm–Coherence duality: G_eff = sech(Λ).
inline double spectral_weight(double modulus) {
    return 1.0 / std::cosh(spectral_lyapunov(modulus));
}

// Spectral R_eff from a single modulus: R_eff(|λ|) = cosh(|log|λ||)
// At the null slice |λ| = 1: R_eff = cosh(0) = 1  (coherence fixed point).
inline double spectral_resistance(double modulus) {
    return std::cosh(spectral_lyapunov(modulus));
}

// ── SpectralEigenvalue ────────────────────────────────────────────────────────
// A single complex eigenvalue λ = |λ| · e^{i·arg(λ)}, represented in polar
// form.  Eigenvalues on the unit circle (|λ| = 1) satisfy the null slice
// condition and contribute weight f = 1 to the coherence envelope.
struct SpectralEigenvalue {
    double modulus;   // |λ|, must be > 0
    double argument;  // arg(λ) ∈ (−π, π]

    SpectralEigenvalue(double mod, double arg) : modulus(mod), argument(arg) {
        if (mod <= 0.0)
            throw std::domain_error("SpectralEigenvalue: modulus must be > 0");
    }

    // Complex value λ = |λ| · (cos θ + i sin θ)
    std::complex<double> complex_value() const {
        return std::polar(modulus, argument);
    }

    // Spectral Lyapunov exponent Λ = |log|λ||
    double lyapunov() const { return spectral_lyapunov(modulus); }

    // Phase weight f(|λ|) = sech(Λ)
    double weight() const { return spectral_weight(modulus); }

    // Effective resistance R_eff = cosh(Λ)
    double R_eff() const { return spectral_resistance(modulus); }

    // True iff |λ| = 1 to within tol (unit circle / null slice condition)
    bool on_unit_circle(double tol = SBP_TOL) const {
        return std::abs(modulus - 1.0) < tol;
    }

    // Deterministic phase evolution with spectral parameter r:
    //   arg(r) = arg(1) + φ · log(r)
    // Returns a new eigenvalue at parameter r (modulus unchanged).
    SpectralEigenvalue evolved(double r, double phi) const {
        if (r <= 0.0)
            throw std::domain_error("SpectralEigenvalue::evolved: r must be > 0");
        double new_arg = argument + phi * std::log(r);
        // Normalise to (−π, π]
        new_arg = std::fmod(new_arg, 2.0 * SBP_PI);
        if (new_arg > SBP_PI)  new_arg -= 2.0 * SBP_PI;
        if (new_arg <= -SBP_PI) new_arg += 2.0 * SBP_PI;
        return SpectralEigenvalue(modulus, new_arg);
    }
};

// ── SpectralBridge ────────────────────────────────────────────────────────────
// An N-eigenvalue spectral system.  Computes the complex coherence envelope:
//
//   C = (1/N) Σ_i  e^{i·arg(λ_i)} · f(|λ_i|)
//
// and the mean effective resistance from spectral Lyapunov exponents.
struct SpectralBridge {
    std::vector<SpectralEigenvalue> eigenvalues;

    explicit SpectralBridge(const std::vector<SpectralEigenvalue> &evs)
        : eigenvalues(evs) {
        if (evs.empty())
            throw std::invalid_argument("SpectralBridge: eigenvalue list is empty");
    }

    // Null slice check: all |λ_i| = 1
    bool is_null_slice(double tol = SBP_TOL) const {
        for (const auto &ev : eigenvalues)
            if (!ev.on_unit_circle(tol)) return false;
        return true;
    }

    // Complex coherence envelope:
    //   C = (1/N) Σ_i  e^{i·arg(λ_i)} · f(|λ_i|)
    std::complex<double> coherence_sum() const {
        std::complex<double> s{0.0, 0.0};
        for (const auto &ev : eigenvalues) {
            double w = ev.weight();
            s += std::complex<double>(std::cos(ev.argument) * w,
                                      std::sin(ev.argument) * w);
        }
        return s / static_cast<double>(eigenvalues.size());
    }

    // Modulus of the coherence envelope |C|
    double coherence_modulus() const { return std::abs(coherence_sum()); }

    // Mean spectral Lyapunov exponent across all eigenvalues
    double mean_lyapunov() const {
        double sum = 0.0;
        for (const auto &ev : eigenvalues)
            sum += ev.lyapunov();
        return sum / static_cast<double>(eigenvalues.size());
    }

    // Mean effective resistance R̄_eff = cosh(mean_lyapunov)
    // At the null slice (all |λ_i| = 1): mean_lyapunov = 0, R̄_eff = 1.
    double mean_R_eff() const {
        return std::cosh(mean_lyapunov());
    }

    // Validate coherence fixed point: R̄_eff = 1 at null slice
    bool validate_coherence_fixed_point(double tol = SBP_TOL) const {
        return is_null_slice(tol) && std::abs(mean_R_eff() - 1.0) < tol;
    }

    // Evolve all eigenvalues to spectral parameter r, with per-eigenvalue
    // phase rates stored in `phis` (must match size of eigenvalues).
    SpectralBridge evolved(double r,
                           const std::vector<double> &phis) const {
        if (phis.size() != eigenvalues.size())
            throw std::invalid_argument("SpectralBridge::evolved: phis size mismatch");
        std::vector<SpectralEigenvalue> evs;
        evs.reserve(eigenvalues.size());
        for (std::size_t i = 0; i < eigenvalues.size(); ++i)
            evs.push_back(eigenvalues[i].evolved(r, phis[i]));
        return SpectralBridge(evs);
    }
};

// ── NullSliceBridge ───────────────────────────────────────────────────────────
// Canonical null slice bridge using the 8-cycle balance primitive
//   µ₀ = e^{i3π/4}
// with periodic unitarity confirmed by  (µ₀)^8 = e^{i6π} = 1.
//
// The 8 powers {µ₀^k : k = 0,…,7} are all distinct and trace all 8th roots
// of unity, establishing arithmetic periodicity in a single 8-step cycle.
struct NullSliceBridge {
    // Argument of µ₀ = e^{i3π/4}
    static constexpr double MU0_ARG = 3.0 * SBP_PI / 4.0;
    // Period of the 8-cycle: (µ₀)^8 = 1
    static constexpr int PERIOD = 8;

    // µ₀ = e^{i3π/4}: modulus 1, argument 3π/4
    static SpectralEigenvalue mu0() {
        return SpectralEigenvalue(1.0, MU0_ARG);
    }

    // k-th power (µ₀)^k: modulus 1, argument k·3π/4 normalised to (−π, π]
    static SpectralEigenvalue mu0_power(int k) {
        double arg = std::fmod(static_cast<double>(k) * MU0_ARG, 2.0 * SBP_PI);
        if (arg > SBP_PI)  arg -= 2.0 * SBP_PI;
        if (arg <= -SBP_PI) arg += 2.0 * SBP_PI;
        return SpectralEigenvalue(1.0, arg);
    }

    // Verify periodic unitarity: (µ₀)^8 = 1
    // (µ₀)^8 = e^{i·8·3π/4} = e^{i6π} = e^{i·3·2π} = 1
    static bool verify_periodicity() {
        double total_arg = static_cast<double>(PERIOD) * MU0_ARG; // = 6π
        double remainder = std::fmod(total_arg, 2.0 * SBP_PI);    // ≈ 0
        return remainder < SBP_TOL || std::abs(remainder - 2.0 * SBP_PI) < SBP_TOL;
    }

    // Build the 8-eigenvalue bridge from µ₀ powers:
    //   {µ₀^0, µ₀^1, …, µ₀^7}  — all on the unit circle, all distinct
    static SpectralBridge build_8cycle_bridge() {
        std::vector<SpectralEigenvalue> evs;
        evs.reserve(PERIOD);
        for (int k = 0; k < PERIOD; ++k)
            evs.push_back(mu0_power(k));
        return SpectralBridge(evs);
    }

    // Verify that the 8 powers are all distinct (no two arguments coincide)
    static bool verify_distinct_powers() {
        const double tol = 1e-9;
        for (int j = 0; j < PERIOD; ++j)
            for (int k = j + 1; k < PERIOD; ++k) {
                double darg = std::abs(mu0_power(j).argument -
                                       mu0_power(k).argument);
                // Wrap difference to [0, π]
                if (darg > SBP_PI) darg = 2.0 * SBP_PI - darg;
                if (darg < tol) return false;
            }
        return true;
    }
};

} // namespace kernel::spectral
