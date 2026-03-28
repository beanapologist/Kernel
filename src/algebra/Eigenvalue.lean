/-
  src/algebra/Eigenvalue.lean — Critical eigenvalue and 8-cycle algebra.

  This module organizes the core algebraic results around the critical
  eigenvalue  μ = exp(I · 3π/4), the Silver ratio δS = 1 + √2, and the
  coherence function  C(r) = 2r/(1 + r²).

  Source: formal-lean/CriticalEigenvalue.lean (78 theorems, 0 sorry)
          formal-lean/SilverCoherence.lean    (29 theorems, 0 sorry)

  Topics covered
  ──────────────
  • μ = exp(I · 3π/4): definition, 8-cycle closure (μ^8 = 1), distinctness
    of the eight powers, rotation-matrix representation.
  • Coherence function  C(r) ∈ (0, 1], monotonicity, Lyapunov duality
    C(exp λ) = sech λ.
  • Silver ratio δS = 1 + √2: uniqueness, palindrome residual R(r).
  • Z/8Z rotational memory: bank addressing, μ^(j+8) = μ^j.
  • Palindrome arithmetic: digit identities, gcd/lcm of torus periods.

  To build this module (from the formal-lean/ project root):
    lake build FormalLean.CriticalEigenvalue
    lake build FormalLean.SilverCoherence
-/

-- Re-export the core algebra modules so downstream files only need to
-- import this single entry point.
import FormalLean.CriticalEigenvalue
import FormalLean.SilverCoherence
import FormalLean.KernelAxle
