/-
  lakefile.lean — Lean 4 project configuration for Kernel formalizations.

  This project formalizes core theorems from the Kernel research repository.
  Mathematical background: ../docs/master_derivations.pdf

  Dependency: Mathlib4 (https://github.com/leanprover-community/mathlib4)

  Quick start:
    lake exe cache get   -- download pre-built Mathlib cache (recommended)
    lake build           -- build all modules
    lake test            -- run tests (if configured)
-/
import Lake
open Lake DSL

package «formalLean» where

-- Mathlib provides complex numbers, real analysis, linear algebra, and more.
-- Pin to a specific tag for reproducibility; update with `lake update`.
require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.14.0"

/-- Library containing the core Kernel formalizations. -/
lean_lib «FormalLean» where
  roots := #[`CriticalEigenvalue, `TimeCrystal, `SpaceTime, `Turbulence, `FineStructure,
             `ParticleMass, `OhmTriality, `SilverCoherence, `KernelAxle, `SpeedOfLight,
             `BidirectionalTime]

/-- Executable entry point that prints a summary of verified theorems. -/
@[default_target]
lean_exe «formalLean» where
  root := `Main
