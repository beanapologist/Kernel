/*
 * KernelPipeline.hpp — Unified Pipeline: Single-Header API
 *
 * Brings together KernelState, SpectralBridge, and PalindromePrecession into
 * one coherent, end-to-end simulation pipeline.
 *
 * Design goals (from problem statement):
 *   1. Unified workflow: coherent, functional, usable end-to-end.
 *   2. Invariant enforcement: |β|=1 (unit-circle), R(r)=0, R_eff=1.
 *   3. Drift detection and auto-renormalization with event logging.
 *   4. User-friendly API — no manual setup required.
 *
 * Quick start:
 *
 *   #include "KernelPipeline.hpp"
 *   using namespace kernel::pipeline;
 *
 *   // Create and run a full pipeline (all modes enabled)
 *   Pipeline pl = Pipeline::create(KernelMode::FULL);
 *   pl.run(16);
 *   bool ok = pl.verify_invariants();   // true if r ≈ 1 throughout
 *
 *   // Inspect the renormalization log
 *   for (const auto &ev : pl.renorm_log())
 *       std::cout << "tick " << ev.tick << ": r " << ev.r_before
 *                 << " → " << ev.r_after << "\n";
 *
 *   // Access the underlying spectral channel
 *   auto ch = pl.channel();
 *   std::cout << "G_eff=" << ch.G_eff() << " R_eff=" << ch.R_eff() << "\n";
 *
 * Activating individual modes:
 *
 *   Pipeline::create(KernelMode::STANDARD)   // µ-rotation only
 *   Pipeline::create(KernelMode::PALINDROME) // + PalindromePrecession
 *   Pipeline::create(KernelMode::SPECTRAL)   // + auto-renormalization
 *   Pipeline::create(KernelMode::FULL)       // all of the above
 */

#pragma once

#include "KernelState.hpp"
#include "PalindromePrecession.hpp"
#include "SpectralBridge.hpp"
#include "ohm_coherence_duality.hpp"

#include <cstdint>
#include <iostream>
#include <string>

namespace kernel::pipeline {

// ── Pipeline
// ──────────────────────────────────────────────────────────────────
//
// End-to-end simulation pipeline combining all submodules.
//
// - Maintains a KernelState (the quantum state with invariant monitoring).
// - Maintains a PalindromePrecession counter (persistent across run() calls).
// - Delegates per-step evolution to SpectralBridge::step().
// - Exposes invariant verification and spectral conversion.
//
class Pipeline {
public:
  // ── Factory ───────────────────────────────────────────────────────────────

  // Create a Pipeline in the given mode, starting from the canonical
  // coherent state (Theorem 8: |ψ⟩ = (1/√2)|0⟩ + e^{i3π/4}/√2|1⟩).
  static Pipeline create(KernelMode mode = KernelMode::FULL) {
    Pipeline pl;
    pl.mode_ = mode;
    return pl;
  }

  // ── Configuration ─────────────────────────────────────────────────────────

  // Override the initial state (must satisfy |α|² + |β|² = 1).
  Pipeline &with_state(KernelState s) {
    state_ = std::move(s);
    pp_.reset();
    return *this;
  }

  // Enable or disable renormalization logging (enabled by default in SPECTRAL
  // and FULL modes; this flag controls verbose stdout output only — the
  // in-memory log is always maintained).
  Pipeline &with_logging(bool enable = true) {
    logging_ = enable;
    return *this;
  }

  // Change the active mode.
  Pipeline &with_mode(KernelMode mode) {
    mode_ = mode;
    return *this;
  }

  // ── Execution ─────────────────────────────────────────────────────────────

  // Advance the pipeline by one tick.  When logging is enabled and a
  // renormalization event is emitted this tick, prints it to stdout.
  void tick() {
    std::size_t log_before = state_.renorm_log.size();
    SpectralBridge::step(state_, mode_, pp_);
    if (logging_ && state_.renorm_log.size() > log_before) {
      const auto &ev = state_.renorm_log.back();
      std::cout << "[pipeline] tick=" << ev.tick << " renorm: r " << ev.r_before
                << " \u2192 " << ev.r_after << " R(r) " << ev.R_before
                << " \u2192 " << ev.R_after << "\n";
    }
  }

  // Advance the pipeline by n ticks.
  void run(uint64_t n) {
    for (uint64_t i = 0; i < n; ++i)
      tick();
  }

  // ── Invariant verification ─────────────────────────────────────────────────

  // Returns true iff all three core invariants hold on the current state:
  //   1. |α|² + |β|² = 1   (unit-circle — no amplitude drift)
  //   2. R(r) = 0            (palindrome residual zero — r = 1)
  //   3. R_eff = 1           (ideal Ohm–Coherence channel)
  bool verify_invariants() const { return SpectralBridge::verify(state_); }

  // Verify using the spectral representation.
  bool verify_spectral() const {
    return SpectralBridge::verify_spectral(state_);
  }

  // ── Accessors ─────────────────────────────────────────────────────────────

  // Current quantum state.
  const KernelState &state() const { return state_; }

  // Auto-renormalization event log (all ticks since last reset).
  const std::vector<RenormEvent> &renorm_log() const {
    return state_.renorm_log;
  }

  // Current spectral (Ohm–Coherence) channel corresponding to this state.
  kernel::ohm::CoherentChannel channel() const {
    return SpectralBridge::to_channel(state_);
  }

  // Active kernel mode.
  KernelMode mode() const { return mode_; }

  // ── Reset ─────────────────────────────────────────────────────────────────

  // Reset to canonical coherent state and clear the precession counter.
  void reset() {
    state_.reset();
    pp_.reset();
  }

private:
  KernelState state_;
  quantum::PalindromePrecession pp_;
  KernelMode mode_ = KernelMode::FULL;
  bool logging_ = false;
};

} // namespace kernel::pipeline
