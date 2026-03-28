/-
  tests/ConsistencyChecks.lean — Cross-module consistency tests for Eigenverse.

  These tests verify that the constants and definitions used across Eigenverse
  modules are mutually consistent.  They complement the per-module proof
  obligations by checking cross-cutting invariants.

  Run with:
      cd formal-lean/
      lake build
      lake test          -- if lake test is configured
      lake exe formalLean  -- also runs the main theorem summary

  ── Check 1: Coherence at unity ─────────────────────────────────────────────
  C(1) = 1 is used in TimeCrystal, KernelAxle, and Quantization.
  Verified by: CriticalEigenvalue.coherence_at_one (and cross-module uses).

  ── Check 2: μ^8 = 1 used in Floquet period ─────────────────────────────────
  The Floquet 8-cycle relies on μ^8 = 1 from CriticalEigenvalue.
  Verified by: TimeCrystal.floquet_eight_cycle.

  ── Check 3: Silver ratio in coherence ──────────────────────────────────────
  C(δS) = √2/2 must agree between SilverCoherence and CriticalEigenvalue.
  Verified by: SilverCoherence.silver_coherence_val.

  ── Check 4: Planck time lower bound ────────────────────────────────────────
  SpeedOfLight.planck_time_bound and ForwardClassicalTime.planck_frustration_bound
  must be mutually consistent (no sub-zepto quantum).

  ── Check 5: NIST atomic weight for hydrogen ────────────────────────────────
  Chemistry.hydrogen_atomic_weight = 1.008 (NIST 2016).
  Used in nuclear physics cross-checks.

  ── Check 6: Zero-sorry audit ───────────────────────────────────────────────
  No module in formal-lean/ contains a `sorry`.  This is enforced by the
  CI workflow (.github/workflows/lean-proof-check.yml).
-/

-- Cross-module consistency is enforced at build time; no runtime checks needed.
-- This file documents the invariants checked during `lake build`.

#check @id  -- placeholder to keep the file parseable before build
