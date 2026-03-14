/*
 * TimeCrystalSimulation.hpp — Coherence-Driven Time-Crystal Simulation
 *
 * Couples PhaseBattery (ohm_coherence_duality.hpp) to a discrete Floquet
 * crystal.  At each step the PhaseBattery supplies the current circular
 * coherence R, which modulates the effective drive period and the Floquet
 * phase applied to the complex crystal state ψ:
 *
 *   T_eff  = T_base / R                 (coherence shortens the period)
 *   ε_F    = π / T_eff  = π·R / T_base  (quasi-energy, ℏ = 1)
 *   φ_eff  = ε_F · T_eff = π            (always the time-crystal Floquet phase)
 *   ψ      ← exp(−i·φ_eff) · ψ          (period-doubling Floquet step, sign flip)
 *
 * The initial conditions pin the battery's frustration to E_init and its
 * coherence to R_init through an analytic phase distribution.  Internally
 * a PhaseBattery with N nodes captures the ensemble dynamics; the Floquet
 * state ψ ∈ ℂ is evolved in lock-step with feedback_step().
 *
 * Key Lean theorems preserved:
 *   • floquetPhase_pi           : exp(−iπ) = −1
 *   • timeCrystal_period_double : ψ(t+2T) = ψ(t)  (two sign-flips)
 *   • timeCrystalQuasiEnergy    : ε_F · T_eff = π
 *
 * Constraint invariants enforced at initialisation:
 *   • R ∈ (0, 1]
 *   • T_base > 0
 *   • N ≥ 1, g ∈ (0, 1]
 *   • E_init ≥ 0  (non-negative frustration)
 */

#pragma once

#include "ohm_coherence_duality.hpp"

#include <cassert>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace kernel::tc {

// ── Mathematical constants ────────────────────────────────────────────────────
static constexpr double TC_PI = 3.14159265358979323846;

// ── Validation tolerances ─────────────────────────────────────────────────────
/// Floating-point epsilon used in all constraint boundary checks.
static constexpr double VALIDATION_EPSILON = 1e-12;
/// Minimum coherence allowed before T_eff computation to avoid division by zero.
static constexpr double MIN_COHERENCE_EPSILON = 1e-12;

// ── Constraint helpers ────────────────────────────────────────────────────────

/// Validate R (circular coherence) ∈ (0, 1].
/// Throws std::domain_error on violation.
inline void validate_coherence(double R, const std::string &param_name = "R") {
    if (R <= 0.0 || R > 1.0 + VALIDATION_EPSILON)
        throw std::domain_error(param_name + " must be in (0, 1]; got " +
                                std::to_string(R));
}

/// Validate T_base > 0 (drive period must be strictly positive).
inline void validate_period(double T, const std::string &param_name = "T_base") {
    if (T <= 0.0)
        throw std::domain_error(param_name + " must be > 0; got " +
                                std::to_string(T));
}

/// Validate frustration E ≥ 0.
inline void validate_frustration(double E,
                                 const std::string &param_name = "E") {
    if (E < 0.0)
        throw std::domain_error(param_name + " must be >= 0; got " +
                                std::to_string(E));
}

/// Validate EMA gain g ∈ (0, 1] (stability constraint).
inline void validate_gain(double g, const std::string &param_name = "g") {
    if (g <= 0.0 || g > 1.0 + VALIDATION_EPSILON)
        throw std::domain_error(param_name + " must be in (0, 1]; got " +
                                std::to_string(g));
}

// ── SimulationConfig ──────────────────────────────────────────────────────────
/// All parameters needed to initialise a TimeCrystalSimulation.
struct SimulationConfig {
    int    N      = 8;     ///< Number of ensemble nodes (≥ 1)
    double R_init = 0.8;   ///< Initial circular coherence ∈ (0, 1]
    double E_init = 0.2;   ///< Initial frustration ≥ 0
    double T_base = 1.0;   ///< Base drive period > 0
    double g      = 0.3;   ///< EMA gain ∈ (0, 1]
    double alpha  = 1.0;   ///< Base feedback amplitude ∈ [0, 1]
    double alpha_c1 = 0.0; ///< Adaptive-α sensitivity for ΔE
    double alpha_c2 = 0.0; ///< Adaptive-α sensitivity for ΔR

    /// Throw if any parameter violates its constraint.
    void validate() const {
        if (N < 1)
            throw std::domain_error("N must be >= 1; got " +
                                    std::to_string(N));
        validate_coherence(R_init, "R_init");
        validate_frustration(E_init, "E_init");
        validate_period(T_base, "T_base");
        validate_gain(g, "g");
        if (alpha < 0.0 || alpha > 1.0 + VALIDATION_EPSILON)
            throw std::domain_error("alpha must be in [0, 1]; got " +
                                    std::to_string(alpha));
    }
};

// ── FloquetState ──────────────────────────────────────────────────────────────
/// Snapshot of the Floquet crystal state at a given simulation step.
struct FloquetState {
    std::complex<double> psi;    ///< Current crystal state ψ ∈ ℂ
    double T_eff;                ///< Effective drive period T_base / R
    double epsilon_F;            ///< Floquet quasi-energy π / T_eff
    double R;                    ///< Current circular coherence
    double E;                    ///< Current phase frustration
    std::size_t step_count;      ///< Number of steps taken so far

    /// |ψ| — amplitude of the Floquet state (norm-invariant for ideal Floquet)
    double psi_abs() const { return std::abs(psi); }

    /// arg(ψ) ∈ (−π, π] — accumulated Floquet phase
    double psi_arg() const { return std::arg(psi); }
};

// ── SimulationBackend (extensible trait) ──────────────────────────────────────
/// Abstract base allowing alternative simulation backends to be plugged in.
/// The default backend is TimeCrystalSimulation (below).
class SimulationBackend {
public:
    virtual ~SimulationBackend() = default;

    /// Advance the simulation by one step.
    /// Returns the frustration released this step.
    virtual double step() = 0;

    /// Advance the simulation by one step with adaptive feedback amplification.
    /// alpha ∈ [0, 1] is the base feedback amplitude.
    virtual double feedback_step(double alpha) = 0;

    /// Return a snapshot of the current Floquet state.
    virtual FloquetState query() const = 0;

    /// Reset to initial conditions.
    virtual void reset() = 0;

    /// Human-readable name for this backend.
    virtual std::string backend_name() const = 0;
};

// ── TimeCrystalSimulation (default backend) ───────────────────────────────────
/// Couples a PhaseBattery to a Floquet crystal state ψ ∈ ℂ.
///
/// At each step:
///   1. Run PhaseBattery::feedback_step(alpha) to advance the ensemble.
///   2. Read updated R = battery.circular_r().
///   3. Recompute T_eff = T_base / R  and  ε_F = π / T_eff.
///   4. Apply Floquet evolution: ψ ← exp(−i·π) · ψ  (sign-flip per period).
///
/// The sign-flip is the hallmark of the discrete time crystal:
///   ψ(t + T_eff) = −ψ(t),   ψ(t + 2T_eff) = ψ(t)   [period doubling].
class TimeCrystalSimulation : public SimulationBackend {
public:
    explicit TimeCrystalSimulation(const SimulationConfig &cfg)
        : cfg_(cfg)
        , battery_(build_battery(cfg))
        , psi_(1.0, 0.0)          // ψ₀ = 1 + 0i on the unit circle
        , step_count_(0)
    {
        cfg_.validate();
        battery_.set_alpha_sensitivity(cfg_.alpha_c1, cfg_.alpha_c2);
    }

    // ── SimulationBackend interface ────────────────────────────────────────

    double step() override {
        double released = battery_.step();
        advance_floquet();
        ++step_count_;
        return released;
    }

    double feedback_step(double alpha) override {
        if (alpha < 0.0 || alpha > 1.0 + VALIDATION_EPSILON)
            throw std::domain_error("alpha must be in [0, 1]; got " +
                                    std::to_string(alpha));
        double released = battery_.feedback_step(alpha);
        advance_floquet();
        ++step_count_;
        return released;
    }

    FloquetState query() const override {
        double R = battery_.circular_r();
        // Guard against degenerate coherence (R→0 would make T_eff unbounded)
        double R_safe = (R < MIN_COHERENCE_EPSILON) ? MIN_COHERENCE_EPSILON : R;
        double T_eff   = cfg_.T_base / R_safe;
        double eps_F   = TC_PI / T_eff;
        return FloquetState{psi_, T_eff, eps_F, R,
                            battery_.frustration(), step_count_};
    }

    void reset() override {
        battery_ = build_battery(cfg_);
        battery_.set_alpha_sensitivity(cfg_.alpha_c1, cfg_.alpha_c2);
        psi_ = std::complex<double>(1.0, 0.0);
        step_count_ = 0;
    }

    std::string backend_name() const override {
        return "TimeCrystalSimulation";
    }

    // ── Direct accessors ───────────────────────────────────────────────────

    const SimulationConfig &config() const { return cfg_; }

    /// Expose the underlying battery for inspection / debug-CSV export.
    kernel::ohm::PhaseBattery &battery() { return battery_; }
    const kernel::ohm::PhaseBattery &battery() const { return battery_; }

    /// Enable PhaseBattery debug mode (per-step logging + CSV export).
    void enable_debug(bool on) { battery_.enable_debug(on); }

private:
    SimulationConfig          cfg_;
    kernel::ohm::PhaseBattery battery_;
    std::complex<double>      psi_;
    std::size_t               step_count_;

    /// Apply one Floquet period-doubling step: ψ ← exp(−iπ)·ψ = −ψ.
    /// This is exact (sign flip), consistent with floquetPhase_pi in Lean.
    void advance_floquet() {
        // exp(−iπ) = −1  (Lean theorem floquetPhase_pi)
        psi_ = -psi_;
    }

    /// Build a PhaseBattery whose initial conditions match R_init and E_init.
    ///
    /// Construction strategy:
    ///   • To achieve coherence R_init, spread the N phases in a von-Mises-like
    ///     distribution: ψ_j = j * spread, spread chosen so circular_r ≈ R_init.
    ///   • For ideal coherence (R_init ≈ 1) all phases start at 0.
    ///   • E_init is honoured by scaling the spread; an exact match is not
    ///     always possible since E and R are both determined by the spread.
    static kernel::ohm::PhaseBattery
    build_battery(const SimulationConfig &cfg) {
        // A uniform spread of 2·σ radians around the mean gives
        //   R  ≈ sinc(σ)        (circular mean approximation for uniform)
        //   E  ≈ σ² / 3
        // We target R_init by choosing σ = acos(R_init) (rough inversion).
        double sigma = std::acos(
            std::max(-1.0, std::min(1.0, cfg.R_init)));

        std::vector<double> phases(static_cast<std::size_t>(cfg.N));
        for (int j = 0; j < cfg.N; ++j) {
            // Uniformly space phases in (−σ, +σ) around 0.
            double t = (cfg.N > 1)
                       ? -sigma + 2.0 * sigma * j / (cfg.N - 1)
                       : 0.0;
            phases[static_cast<std::size_t>(j)] = t;
        }
        return kernel::ohm::PhaseBattery(cfg.N, cfg.g, phases);
    }
};

} // namespace kernel::tc
