/-
  src/quantum/QuantumUniverse.lean — Quantum mechanics formalizations.

  This module organises all quantum-mechanical theorems verified in Eigenverse:

    • Time crystals    Floquet driving, period-2T symmetry breaking,
                       quasi-energy, eigenvalue recipe.
    • Gravity-quantum  Orthogonality principle, dark-energy density,
                       duality gap, equilibrium (‖ψ‖² = 2).
    • Quantization     20 Theorem-Q arms (phase, energy, Floquet,
                       amplitude), no sorry.
    • Forward time     Classical frustration harvesting, Planck floor,
                       coherence-preserving evolution.
    • Bidirectional    F_bi(lf,lb) = F_fwd(lf) + F_fwd(lb); symmetry,
    time               dominance, arrow of time, Planck frustration floor.

  Sources (all proofs in formal-lean/, 0 sorry each)
  ────────────────────────────────────────────────────
  formal-lean/TimeCrystal.lean           (33 theorems)
  formal-lean/GravityQuantumDuality.lean (22 theorems)
  formal-lean/Quantization.lean          (20 theorems)
  formal-lean/ForwardClassicalTime.lean  (21 theorems)
  formal-lean/BidirectionalTime.lean     (24 theorems)

  Usage
  ─────
      import Eigenverse.Quantum.QuantumUniverse
-/

import FormalLean.TimeCrystal
import FormalLean.GravityQuantumDuality
import FormalLean.Quantization
import FormalLean.ForwardClassicalTime
import FormalLean.BidirectionalTime
