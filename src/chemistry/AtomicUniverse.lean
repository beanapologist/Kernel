/-
  src/chemistry/AtomicUniverse.lean — NIST-validated chemistry formalizations.

  This module collects the Lean proofs for chemical and coherence-dynamics
  results that anchor Eigenverse to empirical measurement:

    • Atomic structure   NIST 2016 standard atomic weights for H, He, C, N, O;
                         isotopic abundances; weighted-average mass formula.
    • Ohm–coherence      G · R = 1 duality; parallel/series coherence laws;
                         sech coherence at triality scales; OhmTriality 24 thms.

  Sources (all proofs in formal-lean/, 0 sorry each)
  ────────────────────────────────────────────────────
  formal-lean/Chemistry.lean           (20 theorems)
  formal-lean/OhmTriality.lean         (24 theorems)

  Usage
  ─────
      import Eigenverse.Chemistry.AtomicUniverse
-/

import FormalLean.Chemistry
import FormalLean.OhmTriality
