/*
 * TimeCrystalSimulation.hpp — PhaseBattery-driven discrete-time Floquet simulation
 *
 * Integrates the PhaseBattery coherence engine (ohm_coherence_duality.hpp) with
 * the discrete-time Floquet model of a time crystal (described in
 * formal-lean/TimeCrystal.lean and docs/phase_battery_spec.md).
 *
 * ── Integration design ──────────────────────────────────────────────────────
 *
 *   The PhaseBattery (N oscillators, EMA gain g) produces:
 *     R  = circular coherence  |⟨e^{iψ_j}⟩|  ∈ [0, 1]    (rises toward 1)
 *     E  = phase frustration   (1/N) Σ δθ_j²              (falls toward 0)
 *
 *   Coherence R drives the time crystal's effective drive period:
 *     T_eff = T_base / R                                   (R=1 → T_base)
 *
 *   This means:
 *     • High coherence (R ≈ 1): crystal runs at its natural period T_base.
 *     • Low  coherence (R ≈ 0): crystal period elongates / drive slows.
 *
 *   The Floquet quasi-energy scales accordingly:
 *     ε_F = π / T_eff = π · R / T_base
 *
 *   At each simulation step the crystal state ψ ∈ ℂ advances by the Floquet
 *   phase accumulated over one step (T_step, default = 1.0):
 *     φ_eff = ε_F · T_step = π · T_step / T_eff
 *     ψ ← exp(−i · φ_eff) · ψ
 *
 *   At R = 1, T_base = 1, T_step = 1:
 *     φ_eff = π → ψ → −ψ each step → period-2 time crystal.
 *
 * ── Adaptive amplification ──────────────────────────────────────────────────
 *
 *   PhaseBattery.feedback_step(α) implements a two-sub-step update:
 *
 *     Sub-step 1 (standard EMA, gain g):
 *       ψ̂_j ← ψ̂_j − g · δθ_j
 *
 *     Adaptive α: measured dynamics adapt the base amplification α:
 *       ΔE = E_before − E_mid        (frustration decay)
 *       ΔR = R_mid − R_before        (coherence gain)
 *       α_adaptive = clamp(α + c1·ΔE + c2·ΔR, 0, 1)
 *
 *     Sub-step 2 (coherence-amplified feedback, gain g · α_adaptive · R):
 *       ψ̂_j ← ψ̂_j − g · α_adaptive · R · δθ_j
 *
 *   With default c1 = c2 = 0 the adaptive term vanishes and α_adaptive = α
 *   (backward-compatible).  Setting c1 > 0 or c2 > 0 accelerates convergence
 *   in high-frustration or high-ΔR transients.
 *
 * ── Key parameters ───────────────────────────────────────────────────────────
 *
 *   g       EMA gain = G_eff = sech(λ) ∈ (0, 1].  Controls dissipation rate.
 *           g = 0     → open circuit (no transfer, R fixed)
 *           g ∈ (0,1) → controlled discharge (useful coherence work)
 *           g > 1     → over-shoot / oscillation (unstable)
 *
 *   R       Circular coherence = |⟨e^{iψ_j}⟩| ∈ [0, 1].
 *           Produced by PhaseBattery; drives T_eff = T_base / R.
 *
 *   E       Phase frustration = (1/N) Σ δθ_j².
 *           Source energy in the battery analogy; decreases toward 0.
 *
 *   α       Adaptive amplification base for feedback_step.
 *           Feedback gain g_fb = g · α_adaptive · R ≤ g (always stable).
 *
 *   T_base  Base drive period: T_eff = T_base when R = 1.
 *           Sets the natural quasi-energy ε_F = π / T_base.
 *
 *   T_step  Duration of one simulation step (default = 1.0).
 *           Floquet phase per step: φ_eff = π · T_step / T_eff.
 *
 * ── Usage example ────────────────────────────────────────────────────────────
 *
 *   #include "TimeCrystalSimulation.hpp"
 *   using namespace kernel::ohm;
 *
 *   // 16 oscillators, g = 0.3, scattered phases, T_base = 1.0, α = 0.5
 *   std::vector<double> init(16);
 *   for (int j = 0; j < 16; ++j)
 *       init[j] = (j - 7.5) * OHM_PI / 16.0;          // spread over (−π, π)
 *
 *   TimeCrystalSimulation sim(16, 0.3, init, 1.0, 0.5, true);
 *   //                                       ^T_base  ^alpha  ^verbose
 *   sim.battery.set_alpha_sensitivity(0.2, 0.1);        // adaptive α tuning
 *   sim.run(20);                                         // 20 steps
 *
 *   // After convergence: R ≈ 1, T_eff ≈ T_base, period-2 crystal.
 *   double R_final = sim.coherence();
 *   double E_final = sim.frustration();
 */

#pragma once

#include "ohm_coherence_duality.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace kernel::ohm {

// ── TimeCrystalSimulation ─────────────────────────────────────────────────────
//
// Couples a PhaseBattery (N oscillators) with a discrete-time Floquet
// simulation of a time crystal state ψ ∈ ℂ.
//
// At each simulation step:
//   1. PhaseBattery.feedback_step(α): frustration E decays, coherence R grows.
//   2. T_eff = T_base / R is recomputed from the new coherence.
//   3. The crystal state advances:  ψ ← exp(−i · φ_eff) · ψ
//      where φ_eff = π · T_step / T_eff = π · R · T_step / T_base.
//   4. If verbose, print the step summary to stdout.
//
struct TimeCrystalSimulation {
  // ── Battery ────────────────────────────────────────────────────────────
  PhaseBattery battery; ///< N-node phase oscillator ensemble (source + medium + sink)

  // ── Crystal state ψ = psi_re + i · psi_im ─────────────────────────────
  double psi_re; ///< Real part of the time-crystal state ψ
  double psi_im; ///< Imaginary part of the time-crystal state ψ

  // ── Parameters ────────────────────────────────────────────────────────
  /// Base drive period.  T_eff = T_base when R = 1 (full coherence).
  double T_base;

  /// Duration of one simulation step (default = 1.0).
  /// Floquet phase per step: φ_eff = π · T_step / T_eff.
  double T_step;

  /// Adaptive amplification base α fed to feedback_step.
  /// Feedback gain: g_fb = g · α_adaptive · R (always ≤ g ≤ 1, stable).
  double alpha;

  // ── Simulation state ──────────────────────────────────────────────────
  int step_count; ///< Number of steps executed so far

  /// When true, print R, E, T_eff, ε_F, |ψ|, and arg(ψ)/π to stdout
  /// at the end of every step.
  bool verbose;

  // ── Constructor ───────────────────────────────────────────────────────
  /// @param n_nodes     Number of phase nodes in the PhaseBattery (N ≥ 1).
  /// @param g           EMA gain = G_eff = sech(λ) ∈ (0, 1].
  /// @param init_phases Initial phases ψ̂_j for each node (radians).
  /// @param T_base_     Base drive period (T_eff = T_base_ when R = 1).
  /// @param alpha_      Adaptive amplification base for feedback_step (default 1.0).
  /// @param verbose_    Enable per-step console output (default false).
  TimeCrystalSimulation(int n_nodes, double g,
                        const std::vector<double> &init_phases, double T_base_,
                        double alpha_ = 1.0, bool verbose_ = false)
      : battery(n_nodes, g, init_phases), psi_re(1.0), psi_im(0.0),
        T_base(T_base_), T_step(1.0), alpha(alpha_), step_count(0),
        verbose(verbose_) {
    if (T_base_ <= 0.0)
      throw std::domain_error(
          "TimeCrystalSimulation: T_base must be positive");
  }

  // ── coherence ─────────────────────────────────────────────────────────
  /// Current circular coherence R = |⟨e^{iψ_j}⟩| ∈ [0, 1].
  double coherence() const { return battery.circular_r(); }

  // ── frustration ───────────────────────────────────────────────────────
  /// Current phase frustration E = (1/N) Σ δθ_j² (source energy).
  double frustration() const { return battery.frustration(); }

  // ── effective_period ─────────────────────────────────────────────────
  /// Effective drive period T_eff = T_base / R.
  ///
  /// Coherence R drives the crystal period: high R → T_eff ≈ T_base (natural
  /// drive); low R → T_eff grows (crystal drive slows).  Guarded against R = 0
  /// by clamping at T_base / OHM_TOL.
  double effective_period() const {
    double R = battery.circular_r();
    if (R < OHM_TOL)
      return T_base / OHM_TOL;
    return T_base / R;
  }

  // ── quasi_energy ──────────────────────────────────────────────────────
  /// Floquet quasi-energy ε_F = π / T_eff.
  ///
  /// From TimeCrystal §6: quasi-energy reconstructs the Floquet phase via
  /// ε_F · T_eff = π.  Scales with coherence as ε_F = π · R / T_base.
  double quasi_energy() const { return OHM_PI / effective_period(); }

  // ── psi_abs ───────────────────────────────────────────────────────────
  /// Absolute value |ψ| of the crystal state.
  double psi_abs() const {
    return std::sqrt(psi_re * psi_re + psi_im * psi_im);
  }

  // ── psi_arg ───────────────────────────────────────────────────────────
  /// Argument arg(ψ) ∈ (−π, π] of the crystal state (radians).
  double psi_arg() const { return std::atan2(psi_im, psi_re); }

  // ── step ─────────────────────────────────────────────────────────────
  /// Advance the simulation by one step.
  ///
  /// 1. PhaseBattery feedback_step(α): adaptive EMA + coherence-amplified
  ///    pass; updates R and E.
  /// 2. Recompute T_eff from new R: T_eff = T_base / R.
  /// 3. Advance crystal: ψ ← exp(−i · φ_eff) · ψ
  ///    where φ_eff = π · T_step / T_eff = quasi_energy() · T_step.
  /// 4. Increment step_count; print state if verbose.
  ///
  /// Returns the frustration released this step (ΔE ≥ 0).
  double step() {
    // Step 1 — dissipate frustration, build coherence (adaptive amplification)
    double delta_E = battery.feedback_step(alpha);

    double R = battery.circular_r();
    double E = battery.frustration();
    double Teff = effective_period();
    double eps_F = OHM_PI / Teff; // quasi-energy = π / T_eff

    // Step 3 — Floquet advance: ψ ← exp(−i · φ_eff) · ψ
    // exp(−iφ)(a + ib) = (a·cosφ + b·sinφ) + i(b·cosφ − a·sinφ)
    double phi_eff = eps_F * T_step;
    double cos_phi = std::cos(phi_eff);
    double sin_phi = std::sin(phi_eff);
    double new_re = psi_re * cos_phi + psi_im * sin_phi;
    double new_im = psi_im * cos_phi - psi_re * sin_phi;
    psi_re = new_re;
    psi_im = new_im;

    ++step_count;

    if (verbose) {
      std::cout << std::fixed << std::setprecision(6) << "step=" << std::setw(4)
                << step_count << "  R=" << R << "  E=" << E
                << "  T_eff=" << Teff << "  eps_F=" << eps_F
                << "  |psi|=" << psi_abs()
                << "  arg(psi)/pi=" << psi_arg() / OHM_PI << "\n";
    }

    return delta_E;
  }

  // ── run ───────────────────────────────────────────────────────────────
  /// Run the simulation for n_steps steps.
  ///
  /// If verbose is true, a one-line summary is printed for each step showing
  /// the current coherence R, frustration E, effective period T_eff,
  /// quasi-energy ε_F, crystal amplitude |ψ|, and normalised argument arg(ψ)/π.
  ///
  /// @param n_steps  Number of steps to execute (≥ 0).
  void run(int n_steps) {
    for (int i = 0; i < n_steps; ++i)
      step();
  }

  // ── print_header ──────────────────────────────────────────────────────
  /// Print a column header for the verbose simulation output.
  static void print_header() {
    std::cout << std::setw(6) << "step" << "  " << std::setw(10) << "R"
              << "  " << std::setw(10) << "E" << "  " << std::setw(10)
              << "T_eff" << "  " << std::setw(10) << "eps_F" << "  "
              << std::setw(10) << "|psi|" << "  " << std::setw(12)
              << "arg(psi)/pi" << "\n";
    std::cout << std::string(80, '-') << "\n";
  }
};

} // namespace kernel::ohm
