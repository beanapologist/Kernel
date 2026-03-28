# Architecture

This document explains how the Lean-Verified Mathematical Universe is
structured internally and how the different layers relate to each other.

---

## Layer Model

```
┌─────────────────────────────────────────────────────────────────────┐
│  Consumer layer  (src/)                                             │
│  Topic-organised entry points: algebra/, physics/, quantum/,        │
│  chemistry/, MathUniverse.lean                                      │
├─────────────────────────────────────────────────────────────────────┤
│  Proof layer  (formal-lean/)                                        │
│  Individual *.lean files, one per domain.  These are the files      │
│  that Lean type-checks.                                             │
├─────────────────────────────────────────────────────────────────────┤
│  Foundation layer  (Mathlib4)                                       │
│  Complex numbers, real analysis, linear algebra, group theory …     │
│  Pinned at v4.14.0 via lakefile.lean.                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Dependency Graph

The `formal-lean/` modules import each other in a strict DAG
(no cycles):

```
Mathlib
  └─ FineStructure
       └─ TimeCrystal
            └─ SpeedOfLight
                 ├─ SpaceTime
                 ├─ Turbulence
                 ├─ ParticleMass
                 ├─ OhmTriality
                 ├─ GravityQuantumDuality
                 └─ Chemistry
       └─ Quantization (imports TimeCrystal + FineStructure)
  └─ CriticalEigenvalue  (standalone)
  └─ SilverCoherence     (standalone)
  └─ KernelAxle          (standalone)
  └─ ForwardClassicalTime (standalone)
```

---

## Module Responsibilities

### `CriticalEigenvalue.lean`
The algebraic core of the universe.  Defines and proves properties of:
- μ = exp(I·3π/4) and its 8-cycle μ⁸=1.
- Rotation matrix R(3π/4) and orthogonality.
- Coherence function C(r) = 2r/(1+r²) and its Lyapunov duality.
- Silver ratio δS = 1+√2 and palindrome residual.
- Z/8Z memory bank addressing.

### `FineStructure.lean`
Proves that the fine structure constant α ≈ 1/137.036 is dimensionless
and derives it from first principles using the Kernel coherence structure.

### `TimeCrystal.lean`
Formalises Floquet theory: time-evolution operator, Floquet states,
period-doubling criterion (φ=π), and the Kernel eigenvalue recipe for
constructing time crystals from μ, C(r), η, δS.

### `SpeedOfLight.lean`
Proves c = 1/√(μ₀ε₀) from the Maxwell-equation structural isomorphism,
and establishes Planck time as the absolute smallest temporal unit.

### `SpaceTime.lean`
Lorentz geometry, Minkowski metric, time-dilation, and length-contraction
from first principles, all linked to the Kernel coherence invariants.

### `Turbulence.lean`
Navier-Stokes energy dissipation bounds, Reynolds-number thresholds,
and the turbulence cascade verified from coherence monotonicity.

### `ParticleMass.lean`
Koide formula for lepton masses, proton/electron mass ratio ≈ 1836,
and coherence triality identities.

### `OhmTriality.lean`
Ohm-coherence duality G·R = 1 at three triality scales; parallel and
series coherence laws; sech-coherence representation.

### `SilverCoherence.lean`
C(δS) = √2/2; uniqueness of the balanced point; Im(μ) = C(δS);
45°-physics (phase-angle alignment).

### `KernelAxle.lean`
The axle parameter μ; gear ratio 3:8; cross-section identity;
engine-loop closure.

### `GravityQuantumDuality.lean`
Orthogonality of Re and Im components; Newtonian potential and binding
energy; zero-point energy; dark-energy density; duality gap; Kernel
equilibrium ‖ψ‖²=2.

### `Quantization.lean`
20 Theorem-Q arms: §1 phase, §2 energy, §3 Floquet, §4 amplitude,
§5 Theorem Q synthesis.  Imports TimeCrystal and FineStructure.

### `ForwardClassicalTime.lean`
Frustration harvesting, Planck floor `planck_frustration_bound`, and
coherence-preserving forward time evolution.

### `Chemistry.lean`
NIST 2016 standard atomic weights; isotopic abundances for H, He, C, N, O;
weighted-average mass theorem; 20 theorems, 0 sorry.

---

## Adding a New Module

1. Create `formal-lean/MyTopic.lean` following the header convention.
2. Add `` `MyTopic `` to the `roots` list in `formal-lean/lakefile.lean`.
3. Create `src/<domain>/MyTopic.lean` that imports `FormalLean.MyTopic`.
4. Update `src/MathUniverse.lean` to include the new import.
5. Add an entry to the table in `docs/overview.md`.
6. Submit a PR — the CI workflow will build and verify all proofs.
