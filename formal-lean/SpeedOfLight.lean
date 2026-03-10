/-
  SpeedOfLight.lean — Lean 4 formalization of the speed of light c = 1/√(μ₀ε₀)
  and its structural correspondence with the Kernel critical eigenvalue μ.

  The central result is that both c = 1/√(μ₀ε₀) (Maxwell's electromagnetic
  derivation) and η = 1/√2 (the canonical Kernel amplitude from
  CriticalEigenvalue) share the same algebraic skeleton:

      For P > 0, the unique positive x satisfying  P · x² = 1  is  x = 1/√P.

  Instances of this "balance derivation" pattern:
    • Maxwell framework:  P = μ₀ · ε₀,  constraint: μ₀ε₀ · c² = 1,  c = 1/√(μ₀ε₀)
    • Kernel framework:   P = 2,         constraint: 2 · η² = 1,       η = 1/√2

  The Kernel canonical amplitude η = Im(μ) = 1/√2 arises from the
  canonical-state normalisation η² + |μ·η|² = 1 combined with |μ| = 1.
  The Maxwell speed c arises from the electromagnetic vacuum relation
  μ₀ε₀c² = 1.  Both are characterised by the same abstract balance pattern.

  Exact alignment: when μ₀ε₀ = 2 (the Kernel balance number), c = η exactly.
  This is the machine-checked statement that the two frameworks coincide at
  a specific normalisation.

  Fine structure connection:
  In Hartree atomic units (ℏ = e = mₑ = 4πε₀ = 1), the fine structure
  constant satisfies α = 1/c_au, so c_au = 1/α_FS = 137 (using the
  Sommerfeld approximation α_FS = 1/137 from FineStructure.lean).

  Sections
  ────────
  1.  Abstract balance derivation — P·x² = 1 → x = 1/√P
  2.  Vacuum constants and the Maxwell definition of c
  3.  Properties of c_maxwell
  4.  Canonical amplitude η as the Kernel balance instance
  5.  Structural isomorphism between the Maxwell and Kernel frameworks
  6.  Fine structure connection — c in Hartree atomic units

  Proof status
  ────────────
  All 19 theorems have complete machine-checked proofs.
  No `sorry` placeholders remain.

  Limitations
  ───────────
  • μ₀ and ε₀ are treated as abstract positive real parameters; their
    SI values (μ₀ = 4π×10⁻⁷ H/m, ε₀ ≈ 8.854×10⁻¹² F/m) are physical
    measurements that Lean cannot verify from first principles.
  • The fine structure connection uses the Sommerfeld rational approximation
    α_FS = 1/137 rather than the CODATA 2018 value; this is inherited from
    FineStructure.lean and does not affect the structural results.
  • The structural isomorphism is algebraic: it demonstrates that both
    derivations are instances of the same abstract pattern, not that the
    two frameworks are physically equivalent.
-/

import ForwardClassicalTime

open Complex Real

noncomputable section

-- ════════════════════════════════════════════════════════════════════════════
-- Section 1 — Abstract Balance Derivation
-- For P > 0, the unique positive x satisfying P · x² = 1 is x = 1/√P.
-- This algebraic identity is the skeleton shared by the Maxwell and Kernel
-- frameworks.
-- ════════════════════════════════════════════════════════════════════════════

/-- The balance value 1/√P satisfies the constraint P · (1/√P)² = 1 for P > 0.

    This is the core identity of the balance derivation pattern: the inverse
    square root of P is the unique positive reciprocal normalised by P. -/
theorem balance_constraint (P : ℝ) (hP : 0 < P) :
    P * (1 / Real.sqrt P) ^ 2 = 1 := by
  rw [div_pow, one_pow, Real.sq_sqrt hP.le]
  field_simp [hP.ne']

/-- The balance value 1/√P is the unique positive solution to P · x² = 1.

    Proof: from P · x² = 1 and x > 0, we get x² = (1/√P)², and since both
    values are non-negative, x = 1/√P (unique positive square root). -/
theorem balance_unique (P x : ℝ) (hP : 0 < P) (hx : 0 < x)
    (h : P * x ^ 2 = 1) : x = 1 / Real.sqrt P := by
  -- 1/√P is positive
  have hbalance_pos : 0 < 1 / Real.sqrt P :=
    div_pos one_pos (Real.sqrt_pos.mpr hP)
  -- Both x² and (1/√P)² equal 1/P
  have hxsq : x ^ 2 = (1 / Real.sqrt P) ^ 2 := by
    have h1 : x ^ 2 = 1 / P := by
      rw [eq_div_iff hP.ne', mul_comm (x ^ 2) P]; exact h
    have h2 : (1 / Real.sqrt P) ^ 2 = 1 / P := by
      rw [div_pow, one_pow, Real.sq_sqrt hP.le]
    linarith
  -- From x² = y² with x, y ≥ 0, conclude x = y via √
  calc x = Real.sqrt (x ^ 2)               := (Real.sqrt_sq hx.le).symm
    _ = Real.sqrt ((1 / Real.sqrt P) ^ 2)  := by rw [hxsq]
    _ = 1 / Real.sqrt P                    := Real.sqrt_sq hbalance_pos.le

-- ════════════════════════════════════════════════════════════════════════════
-- Section 2 — Vacuum Constants and the Maxwell Definition of c
-- The vacuum permeability μ₀ and permittivity ε₀ are positive real parameters.
-- Their product μ₀ε₀ > 0 enables the algebraic definition c = 1/√(μ₀ε₀).
-- ════════════════════════════════════════════════════════════════════════════

/-- The Maxwell speed of light derived from vacuum constants μ₀ and ε₀.

    This is the algebraic definition: for any positive reals representing
    vacuum permeability and permittivity, the associated propagation speed is
    their inverse geometric mean.  The SI values μ₀ = 4π×10⁻⁷ H/m and
    ε₀ ≈ 8.854×10⁻¹² F/m give c ≈ 2.998×10⁸ m/s (CODATA 2018). -/
noncomputable def c_maxwell (μ₀ ε₀ : ℝ) : ℝ := 1 / Real.sqrt (μ₀ * ε₀)

/-- The fundamental Maxwell vacuum relation: μ₀ε₀ · c² = 1.

    This is the algebraic content of Maxwell's equations: the speed of
    electromagnetic wave propagation satisfies μ₀ε₀c² = 1, so that
    c = 1/√(μ₀ε₀) is uniquely determined by the vacuum constants. -/
theorem maxwell_vacuum_relation (μ₀ ε₀ : ℝ) (hμ : 0 < μ₀) (hε : 0 < ε₀) :
    μ₀ * ε₀ * c_maxwell μ₀ ε₀ ^ 2 = 1 := by
  unfold c_maxwell
  exact balance_constraint (μ₀ * ε₀) (mul_pos hμ hε)

-- ════════════════════════════════════════════════════════════════════════════
-- Section 3 — Properties of c_maxwell
-- ════════════════════════════════════════════════════════════════════════════

/-- c_maxwell is strictly positive for positive vacuum constants. -/
theorem c_maxwell_pos (μ₀ ε₀ : ℝ) (hμ : 0 < μ₀) (hε : 0 < ε₀) :
    0 < c_maxwell μ₀ ε₀ := by
  unfold c_maxwell
  exact div_pos one_pos (Real.sqrt_pos.mpr (mul_pos hμ hε))

/-- c² = 1/(μ₀ε₀): the square of the speed of light is the reciprocal of the
    product of the vacuum constants. -/
theorem c_maxwell_sq (μ₀ ε₀ : ℝ) (hμ : 0 < μ₀) (hε : 0 < ε₀) :
    c_maxwell μ₀ ε₀ ^ 2 = 1 / (μ₀ * ε₀) := by
  unfold c_maxwell
  rw [div_pow, one_pow, Real.sq_sqrt (mul_pos hμ hε).le]

/-- 1/c = √(μ₀ε₀): the reciprocal of the speed of light equals the geometric
    mean of the vacuum constants. -/
theorem c_maxwell_inv (μ₀ ε₀ : ℝ) (hμ : 0 < μ₀) (hε : 0 < ε₀) :
    1 / c_maxwell μ₀ ε₀ = Real.sqrt (μ₀ * ε₀) := by
  unfold c_maxwell
  field_simp [(Real.sqrt_pos.mpr (mul_pos hμ hε)).ne']

/-- c_maxwell is symmetric in μ₀ and ε₀: the speed of light is the same
    regardless of whether we call the vacuum constant "μ₀" or "ε₀". -/
theorem c_maxwell_symm (μ₀ ε₀ : ℝ) :
    c_maxwell μ₀ ε₀ = c_maxwell ε₀ μ₀ := by
  unfold c_maxwell; rw [mul_comm]

/-- c_maxwell is uniquely determined by the vacuum relation: if a positive real
    c satisfies μ₀ε₀ · c² = 1, then c = c_maxwell μ₀ ε₀. -/
theorem c_maxwell_unique (μ₀ ε₀ c : ℝ) (hμ : 0 < μ₀) (hε : 0 < ε₀)
    (hc : 0 < c) (h : μ₀ * ε₀ * c ^ 2 = 1) : c = c_maxwell μ₀ ε₀ := by
  unfold c_maxwell
  exact balance_unique (μ₀ * ε₀) c (mul_pos hμ hε) hc h

-- ════════════════════════════════════════════════════════════════════════════
-- Section 4 — Canonical Amplitude η as the Kernel Balance Instance
-- η = 1/√2 arises from the same balance pattern with P = 2.
-- The balance constraint 2·η² = 1 follows from the canonical-state
-- normalisation η² + |μ·η|² = 1 combined with |μ| = 1.
-- ════════════════════════════════════════════════════════════════════════════

/-- The canonical amplitude squared: η² = 1/2.
    Follows directly from η = 1/√2. -/
theorem eta_squared : η ^ 2 = 1 / 2 := by
  unfold η
  rw [div_pow, one_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)]

/-- The Kernel canonical balance constraint: 2 · η² = 1.

    This is the Kernel instance of the balance pattern.  It follows from the
    canonical-state normalisation η² + |μ·η|² = 1 (theorem `canonical_norm`
    in CriticalEigenvalue.lean): since |μ| = 1, we have |μ·η|² = η², giving
    η² + η² = 1, i.e., 2η² = 1. -/
theorem kernel_balance_constraint : (2 : ℝ) * η ^ 2 = 1 := by
  rw [eta_squared]; norm_num

/-- η is the unique positive solution to the Kernel balance equation 2·x² = 1. -/
theorem eta_unique (x : ℝ) (hx : 0 < x) (h : 2 * x ^ 2 = 1) : x = η := by
  have h' : x = 1 / Real.sqrt 2 :=
    balance_unique 2 x (by norm_num) hx h
  rw [h']; rfl

-- ════════════════════════════════════════════════════════════════════════════
-- Section 5 — Structural Isomorphism
-- The Maxwell and Kernel frameworks share the algebraic skeleton
-- "P·x² = 1 → x = 1/√P", with P = μ₀ε₀ (Maxwell) and P = 2 (Kernel).
-- The machine-checked isomorphism is that both c and η satisfy the same
-- abstract constraint, differing only in their balance number P.
-- ════════════════════════════════════════════════════════════════════════════

/-- **The structural isomorphism**: c_maxwell and η arise from the same algebraic
    pattern, parametrised by their balance numbers.

    Maxwell:  balance number P_c = μ₀ε₀,  constraint μ₀ε₀ · c² = 1,  c = 1/√(μ₀ε₀)
    Kernel:   balance number P_η = 2,       constraint 2 · η² = 1,       η = 1/√2

    Both satisfy: (balance number) · (value)² = 1
    Both equal:   value = 1 / √(balance number)

    This is the machine-checked statement that the Maxwell derivation of c and
    the Kernel derivation of η are instances of the same algebraic skeleton. -/
theorem maxwell_kernel_structural_iso (μ₀ ε₀ : ℝ) (hμ : 0 < μ₀) (hε : 0 < ε₀) :
    -- Maxwell instance: vacuum relation
    (μ₀ * ε₀ * c_maxwell μ₀ ε₀ ^ 2 = 1) ∧
    -- Kernel instance: canonical balance
    ((2 : ℝ) * η ^ 2 = 1) ∧
    -- Both are inverse-root derivations from their balance number
    (c_maxwell μ₀ ε₀ = 1 / Real.sqrt (μ₀ * ε₀)) ∧
    (η = 1 / Real.sqrt 2) :=
  ⟨maxwell_vacuum_relation μ₀ ε₀ hμ hε,
   kernel_balance_constraint,
   rfl,
   rfl⟩

/-- Balance instances with the same balance number have the same canonical value:
    if two positive reals x and y both satisfy P·x² = 1 and P·y² = 1,
    then x = y. -/
theorem balance_iso_same_number (P x y : ℝ) (hP : 0 < P)
    (hx : 0 < x) (hy : 0 < y)
    (hx_eq : P * x ^ 2 = 1) (hy_eq : P * y ^ 2 = 1) : x = y := by
  rw [balance_unique P x hP hx hx_eq, balance_unique P y hP hy hy_eq]

/-- **Exact alignment**: when μ₀ε₀ = 2 (the Kernel balance number), the Maxwell
    speed c_maxwell equals the Kernel canonical amplitude η = 1/√2.

    This is the unique normalisation at which the two frameworks coincide
    exactly, demonstrating that the Maxwell and Kernel balance patterns are
    not merely analogous — they share a common fixed point. -/
theorem c_equals_eta_when_balance_two (μ₀ ε₀ : ℝ) (hμ : 0 < μ₀) (hε : 0 < ε₀)
    (h : μ₀ * ε₀ = 2) : c_maxwell μ₀ ε₀ = η := by
  unfold c_maxwell η
  rw [h]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 6 — Fine Structure Connection
-- In Hartree atomic units (ℏ = e = mₑ = 4πε₀ = 1), α = 1/c, so the
-- natural-unit speed of light is c_nat = 1/α_FS = 137 (Sommerfeld approx.).
-- ════════════════════════════════════════════════════════════════════════════

/-- The natural-unit speed of light: c_nat = 1/α_FS.

    In Hartree atomic units (ℏ = e = mₑ = 4πε₀ = 1), the fine structure
    constant formula α = e²/(4πε₀ℏc) reduces to α = 1/c, so the natural-unit
    speed of light is c_nat = 1/α_FS. -/
noncomputable def c_natural : ℝ := 1 / α_FS

/-- c_natural = 137: in Hartree atomic units and with the Sommerfeld
    approximation α_FS = 1/137, the speed of light is exactly 137. -/
theorem c_natural_val : c_natural = 137 := by
  unfold c_natural α_FS; norm_num

/-- The natural-unit speed of light is strictly positive. -/
theorem c_natural_pos : 0 < c_natural := by
  unfold c_natural
  exact div_pos one_pos α_FS_pos

/-- The fine structure constant is the reciprocal of the natural-unit speed:
    α_FS = 1/c_natural.

    This is the Kernel framework's bridge between electromagnetic coupling
    and the velocity of light: the fine structure constant measures how
    much smaller the speed of light's reciprocal is compared to unity. -/
theorem α_FS_inv_c_natural : α_FS = 1 / c_natural := by
  unfold c_natural
  field_simp [α_FS_pos.ne']

/-- The product of the fine structure constant and the natural-unit speed
    equals 1: α_FS · c_natural = 1.

    This is the natural-unit statement of α = 1/c, confirming that α and c
    are multiplicative inverses in the Hartree unit system. -/
theorem c_natural_alpha_product : α_FS * c_natural = 1 := by
  unfold c_natural
  field_simp [α_FS_pos.ne']

/-- c_natural is the unique positive real satisfying α_FS · c = 1. -/
theorem c_natural_unique (c : ℝ) (hc : 0 < c) (h : α_FS * c = 1) :
    c = c_natural := by
  unfold c_natural
  rw [eq_div_iff α_FS_pos.ne', mul_comm]
  exact h

end
