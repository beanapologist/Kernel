/-
  src/physics/FundamentalConstants.lean — Verified physical constants.

  This module collects the formal proofs relating fundamental physical
  constants that appear throughout Eigenverse:

    • Speed of light:   c = 1 / √(μ₀ · ε₀)   (Maxwell relation)
    • Fine structure:   α_FS ≈ 1/137.036 (dimensionless coupling)
    • Particle masses:  Koide formula, proton/electron mass ratio
    • Space-time:       Lorentz geometry, Minkowski metric
    • Turbulence:       Navier-Stokes energy dissipation bound

  Sources (all proofs in formal-lean/, 0 sorry each)
  ────────────────────────────────────────────────────
  formal-lean/SpeedOfLight.lean        (19 theorems)
  formal-lean/FineStructure.lean       (30 theorems)
  formal-lean/ParticleMass.lean        (38 theorems)
  formal-lean/SpaceTime.lean           (43 theorems)
  formal-lean/Turbulence.lean          (29 theorems)

  Usage
  ─────
  Import this file to bring all fundamental-constant theorems into scope:

      import Eigenverse.Physics.FundamentalConstants
-/

import FormalLean.SpeedOfLight
import FormalLean.FineStructure
import FormalLean.ParticleMass
import FormalLean.SpaceTime
import FormalLean.Turbulence
