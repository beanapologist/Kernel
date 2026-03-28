/-
  src/geometry/GeometricStructures.lean — Geometric structures of Eigenverse.

  This module collects the formally-verified geometric results that underpin
  every domain in Eigenverse.  The geometry lives on three interlocking
  structures, all driven by the critical eigenvalue μ = exp(i·3π/4):

  1. **Complex unit circle & rotation matrices** (from CriticalEigenvalue)
     The point μ sits at angle 135° on the unit circle.  The associated 2×2
     rotation matrix R(3π/4) is orthogonal (det = 1, Rᵀ = R⁻¹) and has a
     finite order-8 orbit: R(3π/4)⁸ = I.  The full orbit of μ under
     multiplication traces 8 evenly-spaced vertices on S¹.

  2. **Hyperbolic geometry & the Lyapunov duality** (from CriticalEigenvalue)
     The coherence function C(r) = 2r/(1+r²) is the hyperbolic secant of the
     natural Lyapunov exponent: C(exp λ) = sech λ.  This gives a direct
     connection between the unit-circle geometry (coherence, Silver ratio) and
     the hyperbolic plane (sech, sinh, tanh).  The hyperbolic Pythagorean
     identity C(exp λ)² + tanh² λ = 1 ties the two geometries together.

  3. **Complex space-time map F(s,t) = t + i·s** (from SpaceTime & KernelAxle)
     The observer reality map F embeds (t, s) ∈ ℝ × ℝ into ℂ, placing time
     on the real axis and space on the imaginary axis.  Multiplication by i
     is a 90° rotation: F rotates space into time.  The axle gear ratio
     (3 full rotations per 8-step Floquet orbit, gear = 3:8) is a direct
     consequence of the unit-circle geometry.

  Sources (all proofs in formal-lean/, 0 sorry each)
  ────────────────────────────────────────────────────
  formal-lean/CriticalEigenvalue.lean  (78 theorems — §5 rotation matrix,
                                        §7 unit circle, §10 Lyapunov duality,
                                        §14 torus/palindrome, §22 hyperbolic
                                        Pythagorean identity)
  formal-lean/SpaceTime.lean           (43 theorems — §1 space-time domains,
                                        §2 F(s,t) reality map, §4 i·s rotation,
                                        §5 Lorentz geometry)
  formal-lean/KernelAxle.lean          (20 theorems — gear ratio 3:8,
                                        unit-circle constraint, orbit radius,
                                        cross-section identity)

  Key theorems
  ────────────
  • `rotMat_det`       det R(3π/4) = 1          (orientation-preserving)
  • `rotMat_orthog`    R(3π/4) · R(3π/4)ᵀ = I  (isometric rotation)
  • `rotMat_pow_eight` R(3π/4)⁸ = I             (order-8 orbit closure)
  • `mu_abs_one`       |μ| = 1                  (μ on the unit circle)
  • `mu_pow_orbit`     |μⁿ| = 1 for all n       (orbit stays on S¹)
  • `lyapunov_coherence_duality`  C(exp λ) = sech λ
  • `hyperbolic_pythagorean`      C(exp λ)² + tanh²λ = 1
  • `silver_sech`      sech(log δS) = η = 1/√2  (δS characterised via sech)
  • `F_eq_reality`     F(s,t) = t + i·s         (observer map definition)
  • `axle_gear_ratio`  8 · (3π/4) = 3 · (2π)   (3 full rotations per orbit)
  • `axle_orbit_radius` |μⁿ| = 1               (unit-circle orbit)

  Usage
  ─────
      import Eigenverse.Geometry.GeometricStructures
-/

import FormalLean.CriticalEigenvalue
import FormalLean.SpaceTime
import FormalLean.KernelAxle
