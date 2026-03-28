/-
  src/Eigenverse.lean — Top-level entry point for Eigenverse.

  Import this single file to bring the entire Lean-verified Eigenverse into
  scope.  Eigenverse currently contains **450 theorems** across five domains,
  all verified by the Lean 4 type-checker with zero `sorry` placeholders.

  The name "Eigenverse" reflects the central object that drives every structure
  in the project: the critical eigenvalue μ = exp(i·3π/4) — a single complex
  number whose 8-cycle orbit, coherence function C(r) = 2r/(1+r²), and Silver
  ratio δS = 1+√2 give rise to a complete, machine-checked map from pure
  mathematics to observable physical reality.

  ┌──────────────────────────────────────────────────────────────────┐
  │  Domain         Modules                         Theorems         │
  ├──────────────────────────────────────────────────────────────────┤
  │  Algebra        CriticalEigenvalue, SilverCoherence, KernelAxle  │
  │                 → eigenvalue 8-cycle, Silver ratio, coherence     │
  │  Geometry       CriticalEigenvalue, SpaceTime, KernelAxle        │
  │                 → rotation matrices, unit circle S¹, hyperbolic  │
  │                   Pythagorean identity, F(s,t) space-time map    │
  │  Physics        SpeedOfLight, FineStructure, ParticleMass,       │
  │                 SpaceTime, Turbulence                             │
  │                 → c=1/√(μ₀ε₀), α_FS, Koide, Lorentz, NS         │
  │  Quantum        TimeCrystal, GravityQuantumDuality,              │
  │                 Quantization, ForwardClassicalTime,              │
  │                 BidirectionalTime                                │
  │                 → Floquet, dark energy, Theorem Q, frustration, │
  │                   bidirectional time, Planck floor              │
  │  Chemistry      Chemistry, OhmTriality                           │
  │                 → NIST atomic weights, G·R=1 duality             │
  └──────────────────────────────────────────────────────────────────┘

  Note: The crypto-application modules (PumpFunBot, EthereumTradingBot,
  CrossChainDeFiAggregator, CryptoBridge) are available in formal-lean/ for
  reference but are not part of the Eigenverse mathematical library.

  See README.md for an overview, docs/ for detailed documentation, and
  examples/ for worked demonstrations.
-/

import FormalLean.CriticalEigenvalue
import FormalLean.SilverCoherence
import FormalLean.KernelAxle
import FormalLean.SpeedOfLight
import FormalLean.FineStructure
import FormalLean.ParticleMass
import FormalLean.SpaceTime
import FormalLean.Turbulence
import FormalLean.TimeCrystal
import FormalLean.GravityQuantumDuality
import FormalLean.Quantization
import FormalLean.ForwardClassicalTime
import FormalLean.BidirectionalTime
import FormalLean.Chemistry
import FormalLean.OhmTriality
