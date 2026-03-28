# Mathematical Universe — Overview

## What is the Mathematical Universe?

The **Lean-Verified Mathematical Universe** is a fully formal, machine-checked
library of theorems that span algebra, physics, quantum mechanics, and
chemistry.  Every statement is verified by the [Lean 4](https://leanprover.github.io/)
type-checker; there are **zero `sorry` placeholders** anywhere in the codebase.

This means that if Lean accepts a proof, the theorem is *logically guaranteed*
to follow from the stated axioms — no hand-waving, no gaps.

---

## Scope

| Domain | Key Results | Theorems |
|--------|------------|----------|
| **Algebra** | Critical eigenvalue μ = exp(I·3π/4), 8-cycle closure μ⁸=1, Silver ratio δS=1+√2, coherence function C(r), Z/8Z rotational memory | 127 |
| **Physics** | c=1/√(μ₀ε₀), fine structure constant α≈1/137, Koide mass formula, Lorentz geometry, Navier-Stokes turbulence bounds | 159 |
| **Quantum** | Floquet time crystals, gravity-quantum duality, Theorem Q quantization arms, forward classical time | 96 |
| **Chemistry** | NIST 2016 atomic weights, isotopic abundances, Ohm-coherence duality G·R=1, triality-scale coherence | 44 |
| **Total** | | **426** |

---

## Repository Layout

```
src/
├── algebra/
│   └── Eigenvalue.lean         # μ, δS, C(r), Z/8Z memory
├── physics/
│   └── FundamentalConstants.lean  # c, α, mass ratios, spacetime, turbulence
├── quantum/
│   └── QuantumUniverse.lean    # Time crystals, duality, quantization
├── chemistry/
│   └── AtomicUniverse.lean     # NIST atomic weights, Ohm-coherence
└── MathUniverse.lean           # Single-import entry point

formal-lean/                    # Lean 4 source files (the proof engine)
docs/                           # This documentation tree
examples/                       # Worked demonstrations
tests/                          # Consistency and cross-module checks
```

---

## Design Principles

1. **Zero sorry** — every proof compiles without axiom bypasses.
2. **Mathlib grounded** — all arithmetic, linear algebra, and analysis
   reasoning is delegated to [Mathlib4](https://github.com/leanprover-community/mathlib4).
3. **Modular** — each domain is an independent `formal-lean/*.lean` file;
   `src/` organises them by topic for downstream consumers.
4. **NIST-anchored** — numerical constants (atomic weights, isotopic
   compositions, fundamental constants) are taken directly from official
   NIST publications.
5. **Extensible** — adding a new theorem requires only a new `.lean` file
   and a line in `lakefile.lean`; see [CONTRIBUTING.md](../CONTRIBUTING.md).

---

## Quick Start

```bash
cd formal-lean/
lake exe cache get   # download pre-built Mathlib cache (~5 min)
lake build           # build all modules
lake exe formalLean  # print summary of verified theorems
```

---

## References

- Lean 4: <https://leanprover.github.io/>
- Mathlib4: <https://leanprover-community.github.io/mathlib4_docs/>
- NIST Atomic Weights 2016: <https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses>
- Wilczek, F. (2012). Quantum Time Crystals. *Phys. Rev. Lett.* **109**, 160401.
- Sacha & Zakrzewski (2018). Time crystals: a review. *Rep. Prog. Phys.*
