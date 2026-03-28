/-
  Chemistry.lean — Lean 4 formalization of chemical validation using NIST data.

  Data source:
    NIST Atomic Weights and Isotopic Compositions with Relative Atomic Masses (2016)
    https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses

  This file formalizes five groups of chemistry principles, with all numerical
  constants taken from the NIST 2016 standard atomic weights and representative
  isotopic compositions tables.

  Selected elements and their NIST 2016 standard atomic weights (in unified
  atomic mass units, u):

      H  (Z=1)  aw = 1.008  u   — hydrogen
      He (Z=2)  aw = 4.0026 u   — helium
      C  (Z=6)  aw = 12.011 u   — carbon
      N  (Z=7)  aw = 14.007 u   — nitrogen
      O  (Z=8)  aw = 15.999 u   — oxygen

  Selected isotopic abundances (NIST 2016 representative compositions):

      Hydrogen:  H-1  (protium)   x(H-1)  = 0.999885
                 H-2  (deuterium) x(H-2)  = 0.000115
      Carbon:    C-12             x(C-12) = 0.9893
                 C-13             x(C-13) = 0.0107
      Oxygen:    O-16             x(O-16) = 0.99757
                 O-17             x(O-17) = 0.00038
                 O-18             x(O-18) = 0.00205

  The standard atomic weight of an element is the abundance-weighted mean of the
  masses of all naturally occurring isotopes (IUPAC 2016 definition):

      Ar(E) = Σᵢ  x(Eᵢ) · m(Eᵢ)

  This file proves that the NIST representative abundances obey the fundamental
  normalization constraint Σᵢ x(Eᵢ) = 1, that the weighted average lies strictly
  between the lightest and heaviest isotope masses, and that mass is conserved
  in four balanced chemical reactions.

  Sections
  ────────
  1.  Standard atomic weights (NIST 2016)
  2.  Isotopic abundances and normalization
  3.  Atomic weight as abundance-weighted average
  4.  Law of conservation of mass
  5.  Molecular masses and ordering

  Proof status
  ────────────
  All 20 theorems have complete machine-checked proofs.
  No `sorry` placeholders remain.

  Limitations
  ───────────
  • Atomic weight values are the NIST 2016 standard atomic weights represented
    as exact rational fractions; the underlying CODATA 2018 physical measurements
    lie outside the scope of formal mathematics.
  • Isotopic abundances are the NIST 2016 "representative isotopic composition"
    values rounded to 6 significant figures, stored as exact rationals.
  • Molecular masses use the standard atomic weights; individual isotopologue
    masses are not tracked.
  • Avogadro's number (mole conversion) is not formalized here; molar mass
    in g/mol numerically equals the molecular mass in u.
-/

import SpeedOfLight

open Real

noncomputable section

-- ════════════════════════════════════════════════════════════════════════════
-- Section 1 — Standard Atomic Weights (NIST 2016)
-- All values in unified atomic mass units (u = Da).
-- Source: NIST Atomic Weights and Isotopic Compositions, 2016 revision.
-- Ref: https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses
-- ════════════════════════════════════════════════════════════════════════════

/-- Standard atomic weight of hydrogen: Ar(H) = 1.008 u (NIST 2016).

    The conventional atomic weight of hydrogen reflects the natural mixture of
    protium (H-1, 99.9885 %) and deuterium (H-2, 0.0115 %), rounded to 4 s.f.
    Value: 1008/1000 = 1.008 u. -/
noncomputable def aw_H  : ℝ := 1008  / 1000

/-- Standard atomic weight of helium: Ar(He) = 4.0026 u (NIST 2016).

    Helium in the atmosphere is almost entirely He-4 (99.999866 %); the tiny
    trace of primordial He-3 (0.000134 %) shifts the average fractionally above
    4 u.  Value: 40026/10000 = 4.0026 u. -/
noncomputable def aw_He : ℝ := 40026 / 10000

/-- Standard atomic weight of carbon: Ar(C) = 12.011 u (NIST 2016).

    Carbon is 98.93 % C-12 (the mass-unit anchor, m = 12 u exactly) and
    1.07 % C-13 (m ≈ 13.003 u), giving a weighted mean just above 12 u.
    Value: 12011/1000 = 12.011 u. -/
noncomputable def aw_C  : ℝ := 12011 / 1000

/-- Standard atomic weight of nitrogen: Ar(N) = 14.007 u (NIST 2016).

    Nitrogen is 99.632 % N-14 and 0.368 % N-15.  The abundance-weighted
    average sits just above 14 u.  Value: 14007/1000 = 14.007 u. -/
noncomputable def aw_N  : ℝ := 14007 / 1000

/-- Standard atomic weight of oxygen: Ar(O) = 15.999 u (NIST 2016).

    Oxygen is dominated by O-16 (99.757 %) with small fractions of O-17
    (0.038 %) and O-18 (0.205 %).  The weighted mean sits just below 16 u.
    Value: 15999/1000 = 15.999 u. -/
noncomputable def aw_O  : ℝ := 15999 / 1000

/-- The standard atomic weight of hydrogen is strictly positive. -/
theorem aw_H_pos : 0 < aw_H := by unfold aw_H; norm_num

/-- The standard atomic weight of carbon is strictly positive. -/
theorem aw_C_pos : 0 < aw_C := by unfold aw_C; norm_num

/-- The standard atomic weight of oxygen is strictly positive. -/
theorem aw_O_pos : 0 < aw_O := by unfold aw_O; norm_num

/-- The five selected elements follow strict periodic-order by atomic weight:
    Ar(H) < Ar(He) < Ar(C) < Ar(N) < Ar(O).

    Numerical verification:
      1.008 < 4.0026 < 12.011 < 14.007 < 15.999  ✓

    This is the standard ordering of the first 8 elements by atomic number /
    standard atomic weight (with the well-known inversion between Ar/K absent
    here since we stop at O). -/
theorem aw_periodic_order :
    aw_H < aw_He ∧ aw_He < aw_C ∧ aw_C < aw_N ∧ aw_N < aw_O := by
  unfold aw_H aw_He aw_C aw_N aw_O; norm_num

-- ════════════════════════════════════════════════════════════════════════════
-- Section 2 — Isotopic Abundances and Normalization
-- NIST 2016 "representative isotopic composition" values.
-- The fundamental constraint is that all isotopic abundances of one element
-- must sum to exactly 1 (they form a probability distribution over isotopes).
-- ════════════════════════════════════════════════════════════════════════════

/-- Isotopic abundance of H-1 (protium): x(H-1) = 0.999885 (NIST 2016).

    Protium is by far the dominant hydrogen isotope.  Its abundance of 99.9885 %
    is stored as the exact rational 999885/1000000. -/
noncomputable def ab_H1  : ℝ := 999885 / 1000000

/-- Isotopic abundance of H-2 (deuterium): x(H-2) = 0.000115 (NIST 2016).

    Deuterium constitutes 0.0115 % of natural hydrogen.  Its abundance is stored
    as the exact rational 115/1000000. -/
noncomputable def ab_H2  : ℝ :=    115 / 1000000

/-- Isotopic abundance of C-12: x(C-12) = 0.9893 (NIST 2016).

    C-12 defines the atomic mass unit (m(C-12) = 12 u exactly) and accounts
    for 98.93 % of natural carbon.  Stored as 9893/10000. -/
noncomputable def ab_C12 : ℝ :=   9893 / 10000

/-- Isotopic abundance of C-13: x(C-13) = 0.0107 (NIST 2016).

    C-13 (1.07 % of natural carbon) is the only stable heavy carbon isotope
    and is widely used in NMR spectroscopy.  Stored as 107/10000. -/
noncomputable def ab_C13 : ℝ :=    107 / 10000

/-- Isotopic abundance of O-16: x(O-16) = 0.99757 (NIST 2016).

    O-16 is the overwhelmingly dominant oxygen isotope (99.757 %).
    Stored as 99757/100000. -/
noncomputable def ab_O16 : ℝ :=  99757 / 100000

/-- Isotopic abundance of O-17: x(O-17) = 0.00038 (NIST 2016).

    O-17 is the only stable odd-neutron oxygen isotope (0.038 %).
    Stored as 38/100000. -/
noncomputable def ab_O17 : ℝ :=     38 / 100000

/-- Isotopic abundance of O-18: x(O-18) = 0.00205 (NIST 2016).

    O-18 (0.205 %) is used as a tracer in geochemistry and metabolism studies.
    Stored as 205/100000. -/
noncomputable def ab_O18 : ℝ :=    205 / 100000

/-- Both hydrogen isotopic abundances are strictly positive.

    Positivity is required for the weighted-average theorems in Section 3:
    a zero abundance would make the corresponding isotope irrelevant, but
    NIST lists both H-1 and H-2 with nonzero representative abundances. -/
theorem hydrogen_abundances_pos : 0 < ab_H1 ∧ 0 < ab_H2 := by
  unfold ab_H1 ab_H2; norm_num

/-- **Hydrogen normalization**: the NIST isotopic abundances of H-1 and H-2
    sum to exactly 1.

        x(H-1) + x(H-2) = 999885/1000000 + 115/1000000
                        = 1000000/1000000 = 1  ✓

    This is the fundamental constraint that isotopic abundances form a
    probability distribution over the stable isotopes of an element. -/
theorem hydrogen_abundances_sum_one : ab_H1 + ab_H2 = 1 := by
  unfold ab_H1 ab_H2; norm_num

/-- **Carbon normalization**: the NIST isotopic abundances of C-12 and C-13
    sum to exactly 1.

        x(C-12) + x(C-13) = 9893/10000 + 107/10000
                          = 10000/10000 = 1  ✓  -/
theorem carbon_abundances_sum_one : ab_C12 + ab_C13 = 1 := by
  unfold ab_C12 ab_C13; norm_num

/-- **Oxygen normalization**: the NIST isotopic abundances of O-16, O-17, and
    O-18 sum to exactly 1.

        x(O-16) + x(O-17) + x(O-18)
          = 99757/100000 + 38/100000 + 205/100000
          = 100000/100000 = 1  ✓

    Oxygen has three stable isotopes; their NIST representative abundances
    satisfy the normalization constraint exactly. -/
theorem oxygen_abundances_sum_one : ab_O16 + ab_O17 + ab_O18 = 1 := by
  unfold ab_O16 ab_O17 ab_O18; norm_num

/-- Protium is the overwhelmingly dominant hydrogen isotope: x(H-1) > x(H-2).

    Numerically: 999885/1000000 > 115/1000000 (protium exceeds deuterium
    by a factor of 8695).  This reflects the primordial nucleosynthesis
    abundance: the Big Bang produced roughly one deuterium per 25 hydrogen. -/
theorem protium_dominant : ab_H2 < ab_H1 := by
  unfold ab_H1 ab_H2; norm_num

-- ════════════════════════════════════════════════════════════════════════════
-- Section 3 — Atomic Weight as Abundance-Weighted Average
-- The IUPAC 2016 definition: Ar(E) = Σᵢ x(Eᵢ)·m(Eᵢ).
-- We prove structural properties of this weighted average for hydrogen:
-- positivity, and the strict-between-bounds property that guarantees the
-- standard atomic weight lies strictly between the lightest and heaviest
-- isotope masses.
-- ════════════════════════════════════════════════════════════════════════════

/-- The abundance-weighted average isotope mass for hydrogen with abstract
    isotope masses m1 (for H-1) and m2 (for H-2):

        aw_H_weighted(m₁, m₂) = x(H-1) · m₁ + x(H-2) · m₂

    With the NIST 2016 masses m(H-1) ≈ 1.00782503207 u and
    m(H-2) ≈ 2.01410177812 u, this formula reproduces the NIST standard
    atomic weight Ar(H) = 1.008 u. -/
noncomputable def aw_H_weighted (m1 m2 : ℝ) : ℝ := ab_H1 * m1 + ab_H2 * m2

/-- The abundance-weighted average is strictly positive whenever both isotope
    masses are positive.

    Proof: both terms x(H-1)·m₁ and x(H-2)·m₂ are products of positive
    numbers (the abundances are positive by `hydrogen_abundances_pos`), so
    their sum is positive. -/
theorem isotope_average_pos (m1 m2 : ℝ) (h1 : 0 < m1) (h2 : 0 < m2) :
    0 < aw_H_weighted m1 m2 := by
  unfold aw_H_weighted ab_H1 ab_H2
  nlinarith [mul_pos (by norm_num : (0 : ℝ) < 999885 / 1000000) h1,
             mul_pos (by norm_num : (0 : ℝ) < 115 / 1000000) h2]

/-- **Lower bound**: when m₁ < m₂, the weighted average strictly exceeds m₁.

    Proof outline:
      aw_H_weighted(m₁,m₂) − m₁
        = x(H-1)·m₁ + x(H-2)·m₂ − 1·m₁
        = x(H-1)·m₁ + x(H-2)·m₂ − (x(H-1)+x(H-2))·m₁   [sum-to-1]
        = x(H-2)·(m₂ − m₁) > 0                           [x(H-2),m₂−m₁ > 0]

    This ensures the standard atomic weight is strictly heavier than the
    lightest isotope mass, consistent with the presence of heavier isotopes. -/
theorem isotope_average_lower_bound (m1 m2 : ℝ) (hm1 : 0 < m1) (hlt : m1 < m2) :
    m1 < aw_H_weighted m1 m2 := by
  unfold aw_H_weighted ab_H1 ab_H2
  nlinarith [mul_pos (by norm_num : (0 : ℝ) < 115 / 1000000)
               (show (0 : ℝ) < m2 - m1 by linarith)]

/-- **Upper bound**: when m₁ < m₂, the weighted average is strictly less than m₂.

    Proof outline:
      m₂ − aw_H_weighted(m₁,m₂)
        = m₂ − x(H-1)·m₁ − x(H-2)·m₂
        = (1−x(H-2))·m₂ − x(H-1)·m₁
        = x(H-1)·m₂ − x(H-1)·m₁                         [sum-to-1 ⇒ 1−x(H-2)=x(H-1)]
        = x(H-1)·(m₂ − m₁) > 0                           [x(H-1),m₂−m₁ > 0]

    Together with `isotope_average_lower_bound`, this proves that the standard
    atomic weight lies strictly between the lightest and heaviest isotope masses,
    which is the expected physical behaviour. -/
theorem isotope_average_upper_bound (m1 m2 : ℝ) (hm1 : 0 < m1) (hlt : m1 < m2) :
    aw_H_weighted m1 m2 < m2 := by
  unfold aw_H_weighted ab_H1 ab_H2
  nlinarith [mul_pos (by norm_num : (0 : ℝ) < 999885 / 1000000)
               (show (0 : ℝ) < m2 - m1 by linarith)]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 4 — Law of Conservation of Mass
-- In a balanced chemical equation the total atomic count of each element is
-- equal on both sides.  Because mass is additive and atom counts balance,
-- the total mass of all reactants equals the total mass of all products.
-- The proofs below are purely algebraic (ring equalities) that follow from
-- the balanced stoichiometry; they hold for any positive values of the
-- atomic weight parameters.
-- ════════════════════════════════════════════════════════════════════════════

/-- **Water synthesis**: mass is conserved in 2 H₂ + O₂ → 2 H₂O.

    Atom balance: 4 H + 2 O = 4 H + 2 O  ✓
    Mass balance (in terms of aw_H, aw_O):
        reactants: 2·(2·aw_H) + 2·aw_O
        products:  2·(2·aw_H + aw_O)
    Equality follows by ring. -/
theorem water_synthesis_mass_conservation :
    2 * (2 * aw_H) + 2 * aw_O = 2 * (2 * aw_H + aw_O) := by ring

/-- **Methane combustion**: mass is conserved in CH₄ + 2 O₂ → CO₂ + 2 H₂O.

    Atom balance: C:1=1, H:4=4, O:4=4  ✓
    Mass balance:
        reactants: (aw_C + 4·aw_H) + 2·(2·aw_O)
        products:  (aw_C + 2·aw_O) + 2·(2·aw_H + aw_O)
    Equality follows by ring. -/
theorem methane_combustion_mass_conservation :
    (aw_C + 4 * aw_H) + 2 * (2 * aw_O) =
    (aw_C + 2 * aw_O) + 2 * (2 * aw_H + aw_O) := by ring

/-- **Ammonia synthesis**: mass is conserved in N₂ + 3 H₂ → 2 NH₃.

    Atom balance: N:2=2, H:6=6  ✓
    Mass balance:
        reactants: 2·aw_N + 3·(2·aw_H)
        products:  2·(aw_N + 3·aw_H)
    Equality follows by ring. -/
theorem ammonia_synthesis_mass_conservation :
    2 * aw_N + 3 * (2 * aw_H) = 2 * (aw_N + 3 * aw_H) := by ring

/-- **Carbon monoxide oxidation**: mass is conserved in 2 CO + O₂ → 2 CO₂.

    Atom balance: C:2=2, O:4=4  ✓
    Mass balance:
        reactants: 2·(aw_C + aw_O) + 2·aw_O
        products:  2·(aw_C + 2·aw_O)
    Equality follows by ring. -/
theorem carbon_monoxide_oxidation_conservation :
    2 * (aw_C + aw_O) + 2 * aw_O = 2 * (aw_C + 2 * aw_O) := by ring

-- ════════════════════════════════════════════════════════════════════════════
-- Section 5 — Molecular Masses and Ordering
-- Molecular (formula) mass = sum of standard atomic weights of constituent
-- atoms, using NIST 2016 values from Section 1.
-- The four common molecules H₂O, CO₂, CH₄, NH₃ satisfy a strict mass
-- ordering: CH₄ < NH₃ < H₂O < CO₂.
-- ════════════════════════════════════════════════════════════════════════════

/-- Molecular mass of water (H₂O): M(H₂O) = 2·aw_H + aw_O.

    Using NIST 2016 values: 2·1.008 + 15.999 = 18.015 u. -/
noncomputable def mol_H2O : ℝ := 2 * aw_H + aw_O

/-- Molecular mass of carbon dioxide (CO₂): M(CO₂) = aw_C + 2·aw_O.

    Using NIST 2016 values: 12.011 + 2·15.999 = 44.009 u. -/
noncomputable def mol_CO2 : ℝ := aw_C + 2 * aw_O

/-- Molecular mass of methane (CH₄): M(CH₄) = aw_C + 4·aw_H.

    Using NIST 2016 values: 12.011 + 4·1.008 = 16.043 u. -/
noncomputable def mol_CH4 : ℝ := aw_C + 4 * aw_H

/-- Molecular mass of ammonia (NH₃): M(NH₃) = aw_N + 3·aw_H.

    Using NIST 2016 values: 14.007 + 3·1.008 = 17.031 u. -/
noncomputable def mol_NH3 : ℝ := aw_N + 3 * aw_H

/-- The molecular mass of water is strictly positive.

    Follows immediately from positivity of aw_H and aw_O (Section 1). -/
theorem mol_H2O_pos : 0 < mol_H2O := by
  unfold mol_H2O aw_H aw_O; norm_num

/-- Carbon dioxide is heavier than water: M(H₂O) < M(CO₂).

    Numerically: 18.015 < 44.009 u.  Verified by norm_num after unfolding
    the NIST 2016 rational definitions. -/
theorem mol_CO2_heavier_than_H2O : mol_H2O < mol_CO2 := by
  unfold mol_H2O mol_CO2 aw_H aw_C aw_O; norm_num

/-- Water is heavier than ammonia: M(NH₃) < M(H₂O).

    Numerically: 17.031 < 18.015 u.  Although nitrogen is heavier than
    oxygen, ammonia has only three hydrogen atoms versus the one oxygen in
    water, so the molecular mass of H₂O still exceeds that of NH₃. -/
theorem mol_H2O_heavier_than_NH3 : mol_NH3 < mol_H2O := by
  unfold mol_H2O mol_NH3 aw_H aw_N aw_O; norm_num

/-- Ammonia is heavier than methane: M(CH₄) < M(NH₃).

    Numerically: 16.043 < 17.031 u.  Methane carries four hydrogens while
    ammonia carries three, but nitrogen (14.007 u) is heavier than carbon
    (12.011 u), making ammonia the heavier molecule overall. -/
theorem mol_NH3_heavier_than_CH4 : mol_CH4 < mol_NH3 := by
  unfold mol_CH4 mol_NH3 aw_H aw_C aw_N; norm_num

end
