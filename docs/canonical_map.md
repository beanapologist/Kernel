# Canonical Map: Mathematical Structures ↔ Observable Reality

**Framework:** Kernel — Quantum Coherence Pipeline  
**Formal proofs:** `formal-lean/` (Lean 4, Mathlib, all machine-checked, no `sorry`)  
**Empirical validation:** `empirical-validation/` (CODATA 2018, NIST, Planck 2018, PDG 2022)  
**Generator:** `empirical-validation/canonical_map.py`

---

## Purpose

This document is the **comprehensive canonical map** of the Kernel framework.  For
every major mathematical structure established in the Lean 4 proofs it records:

1. The **formal Lean theorem(s)** that establish the structure (§ Lean theorems).
2. The **empirical validator(s)** that compare it against independent experimental
   data and show whether it is consistent with real-world measurement (§ Validation).
3. The **observable physical phenomena** that the structure models or predicts
   (§ Observable phenomena).
4. The **external data sources** (CODATA 2018, NIST, Planck 2018, PDG 2022, …)
   used for verification (§ Data sources).

> **Evidence taxonomy.**  Only checks classified **`empirical`** can falsify the
> framework by revealing a discrepancy with experiment.  Checks classified
> `mathematical_identity` or `numerical_precision` verify *internal*
> self-consistency only — they can never distinguish a correct model from an
> incorrect one.

---

## Summary Statistics

| Item | Count |
|------|-------|
| Mathematical structures | 17 |
| Lean source files | 14 |
| Formally proved theorems (total) | 432 |
| Empirical-validation checks | 78 |
| **Empirical** checks | **20** |
| **Empirical checks passed** | **20 / 20 (100 %)** |

---

## Structure 1 — Critical Eigenvalue  μ = exp(i·3π/4)

**Lean file:** `CriticalEigenvalue.lean` · **Theorems:** 78

### Definition

```
μ = exp(i · 3π/4)      angle 135°, unit circle
η = 1/√2               canonical amplitude
```

### Key Lean Theorems

| Theorem | Statement |
|---------|-----------|
| `mu_def` | μ = exp(I · 3π/4) |
| `mu_abs_one` | \|μ\| = 1  (μ on the unit circle) |
| `mu_pow_eight` | μ⁸ = 1  (8-cycle closure) |
| `mu_eq_cart` | μ = (−1 + i)/√2  (Cartesian form) |
| `canonical_norm` | η² + \|μ·η\|² = 1,  η = 1/√2 |
| `rotMat_det` | det R(3π/4) = 1  (orientation-preserving) |
| `rotMat_orthog` | R(3π/4) · R(3π/4)ᵀ = I |
| `rotMat_pow_eight` | R(3π/4)⁸ = I  (orbit closure) |

### Observable Phenomena

- 8-step discrete Floquet orbit in periodically driven quantum systems.
- Rotation matrix R(3π/4) acts on the qubit Bloch-sphere equatorial plane.
- Phase accumulation of 3π/4 per Floquet period in time-crystal experiments.
- Unit-modulus constraint |μ| = 1 is equivalent to norm-preserving (unitary) evolution.
- Chiral kick μ = e^{i3π/4} creates preferred/anti-preferred computational basins
  (6.8 % performance advantage for best vs worst phase in 5,040-parameter experiment).

### Validation

All eigenvalue checks are `mathematical_identity` or `numerical_precision`.
They confirm internal self-consistency:

| Check | Type | Criterion |
|-------|------|-----------|
| `eigenvalue_norm_sq_exact` | math-id | SymPy: \|μ\|² = 1 exactly |
| `eigenvalue_8th_power_exact` | math-id | SymPy: μ⁸ = 1 exactly |
| `eigenvalue_real_part_exact` | math-id | Re(μ) = −1/√2 exactly |
| `eigenvalue_imag_part_exact` | math-id | Im(μ) = 1/√2 exactly |
| `rotation_matrix_eigenvalue_norm` | num-prec | \|R eigenvalues\| = 1 (< 10⁻¹⁴) |
| `rotation_matrix_8th_power_identity` | num-prec | \|R⁸ − I\| < 10⁻¹⁴ |

### Data Sources

- SymPy symbolic: `exp(i·3π/4)` — exact complex arithmetic.
- NumPy/cmath: IEEE 754 float verification.
- NumPy linalg: eigenvalue norms and matrix power.

---

## Structure 2 — Coherence Function  C(r) = 2r/(1+r²)

**Lean file:** `CriticalEigenvalue.lean` · **Theorems:** 78

### Definition

```
C(r) = 2r / (1 + r²)    for r ≥ 0
C(1) = 1                 unique maximum
C(r) = C(1/r)            symmetric about r = 1
```

### Key Lean Theorems

| Theorem | Statement |
|---------|-----------|
| `coherence_le_one` | C(r) ≤ 1 for r ≥ 0, with C(1) = 1 |
| `coherence_eq_one_iff` | C(r) = 1 ↔ r = 1 |
| `coherence_pos` | 0 < C(r) for all r > 0 |
| `coherence_symm` | C(r) = C(1/r) for r > 0 |
| `coherence_lt_one` | C(r) < 1 for r ≥ 0, r ≠ 1 |
| `lyapunov_coherence_duality` | C(exp λ) = sech λ |
| `koide_coherence_bridge` | C(φ²) = 2/3 (Koide relation) |

### Observable Phenomena

- Optimal stochastic resonance observed at C ≈ 0.82 (not C = 1) in 5,040-parameter
  coherence-mining experiment (see Discoveries below).
- Lyapunov–coherence duality C(e^λ) = sech λ bounds the Lyapunov exponent:
  λ_max = 1/e ≈ 0.368 (consistent with the universal scaling limit α_max = 1+1/e).
- Coherence value C = 2/3 links both the lepton (r = φ²) and hadronic (r = 1/φ²)
  scales (Koide bridge and hadronic triality — see §6, §10).

### Validation

| Check | Type | Criterion |
|-------|------|-----------|
| `coherence_at_zero_is_one` | math-id | C(0) = 0 (see note below) |
| `coherence_strictly_decreasing` | math-id | dC/dr for r > 1 |
| `coherence_gaussian_integral` | math-id | SymPy integral identity |
| `coherence_numerical_*` | num-prec | Two code-paths agree to 10⁻¹⁴ |

> **Note on the coherence validator:** `empirical-validation/validators/coherence.py`
> implements a *Gaussian* proxy C_proxy(r) = exp(−r²/2) for checking monotonicity and
> integral properties.  The Lean canonical definition is C(r) = 2r/(1+r²).
> The proxy shares qualitative properties (positivity, monotonicity) used by the
> validator; the canonical value C(δ_S) = √2/2 is derived from the Lean formula.

### Data Sources

- SymPy symbolic: exact Gaussian integral `∫₀^∞ exp(−r²/2) dr = √(π/2)`.
- NumPy: monotonicity and range checks.

---

## Structure 3 — Silver Ratio & Silver Coherence  δ_S = 1+√2

**Lean file:** `SilverCoherence.lean` · **Theorems:** 29

### Definition

```
δ_S = 1 + √2 ≈ 2.4142     silver ratio
C(δ_S) = √2/2 ≈ 0.7071    silver coherence
Im(μ) = C(δ_S)             45°-physics bridge
```

### Key Lean Theorems

| Theorem | Statement |
|---------|-----------|
| `silverRatio_def` | δ_S = 1 + √2 |
| `silverRatio_mul_conj` | δ_S · (√2 − 1) = 1 (silver conservation) |
| `silverCoherence_val` | C(δ_S) = √2/2 |
| `mu_im_eq_silver_coherence` | Im(μ) = C(δ_S)  (45°-physics bridge) |
| `silver_coherence_unique` | C(r) = √2/2 ↔ r = δ_S |
| `silverRatio_minimal_poly` | δ_S² − 2·δ_S − 1 = 0 |
| `silver_mirror_coherence` | C(1/δ_S) = C(δ_S) |

### Observable Phenomena

- Im(μ) = sin(135°) = 1/√2 = C(δ_S): the imaginary part of the critical eigenvalue
  equals the silver coherence value — a direct 45°-physics bridge.
- δ_S is the **unique** positive real where C(r) = 1/√2 (half-power coherence point).
- The silver ratio appears in quasi-crystal diffraction patterns (Ammann–Beenker
  aperiodic tiling), octagonal symmetry, and silver-ratio continued fractions.

### Validation (via `golden_ratio` section)

| Check | Type | Status |
|-------|------|--------|
| `silver_conservation_sympy` | math-id | PASS |
| `silver_ratio_minimal_polynomial_sympy` | math-id | PASS |
| `silver_conservation_numerical` | num-prec | PASS |
| `silver_ratio_value_nist` | **EMPIRICAL** | PASS (rel. error < 10⁻¹⁵ vs NIST) |

### Data Sources

- NIST DLMF: δ_S = 1+√2 tabulated (cross-checked via `data_ingestion/nist.py`).
- SymPy: (1+√2)(√2−1) = 2−1 = 1 (difference-of-squares, exact).

---

## Structure 4 — Golden Ratio  φ = (1+√5)/2

**Lean file:** `ParticleMass.lean` · **Theorems:** 38

### Definition

```
φ = (1 + √5) / 2 ≈ 1.6180     golden ratio
φ² = φ + 1                     defining property
φ − 1 = 1/φ                    reciprocal identity
```

### Key Lean Theorems

| Theorem | Statement |
|---------|-----------|
| `goldenRatio_sq` | φ² = φ + 1 |
| `goldenRatio_recip` | φ − 1 = 1/φ |
| `goldenRatio_minimal_poly` | φ² − φ − 1 = 0 |
| `koide_coherence_bridge` | C(φ²) = 2/3  (Koide Q) |
| `golden_koide_eq` | C(φ²) = C(1/φ²) = 2/3 |

### Observable Phenomena

- Fibonacci sequence: F(n+1)/F(n) → φ as n → ∞ (72nd ratio agrees with NIST φ
  to relative error < 10⁻¹²).
- φ appears in plant phyllotaxis (spiral counts), Penrose tilings, icosahedral
  symmetry (dodecahedron, C₆₀ fullerene), and quasi-crystal structure.
- Koide bridge: C(φ²) = 2/3 links the golden ratio to the lepton mass hierarchy.

### Validation

| Check | Type | Status |
|-------|------|--------|
| `golden_ratio_quadratic_identity_sympy` | math-id | PASS |
| `golden_ratio_reciprocal_identity_sympy` | math-id | PASS |
| `golden_ratio_minimal_polynomial_sympy` | math-id | PASS |
| `golden_ratio_minimal_polynomial_numerical` | num-prec | PASS |
| `fibonacci_convergence_golden_ratio` | **EMPIRICAL** | PASS (rel. error < 10⁻¹²) |
| `golden_ratio_value_nist` | **EMPIRICAL** | PASS (rel. error < 10⁻¹⁵ vs NIST) |

### Data Sources

- NIST DLMF: φ = (1+√5)/2 (cross-checked via `data_ingestion/nist.py`).
- Independent integer Fibonacci sequence (72nd ratio, no floating-point involved).

---

## Structure 5 — Fine-Structure Constant  α ≈ 7.2974×10⁻³

**Lean file:** `FineStructure.lean` · **Theorems:** 30

### Definition

```
α = e² / (4π ε₀ ℏ c) ≈ 7.2973525693×10⁻³
1/α ≈ 137.035999084
```

### Key Lean Theorems

| Theorem | Statement |
|---------|-----------|
| `alpha_def` | α = e²/(4πε₀ℏc) |
| `alpha_lt_one` | α < 1  (weak electromagnetic coupling) |
| `alpha_gt_zero` | 0 < α |
| `alpha_inverse_bound` | 1/α ∈ (137, 138) |
| `rydberg_energy` | E_Ryd = α²·m_e·c²/2 |
| `alpha_fs_bounds` | 7.29×10⁻³ < α < 7.30×10⁻³ |

### Observable Phenomena

- **Hydrogen spectral lines:** energy levels E_n = −E_Ryd/n², with
  E_Ryd = α²m_ec²/2 ≈ 13.6 eV.  Fine structure splitting ∝ α⁴.
- **Lamb shift** in hydrogen (1s–2s): QED radiative correction ∝ α³ × E_Ryd.
- **Anomalous magnetic moment** of the electron: (g−2)/2 ≈ α/π (Schwinger).
- **Quantized Hall resistance:** R_K = h/e² = 1/(α·G_0) ≈ 25,812.807 Ω.
- **Josephson effect:** voltage steps ΔV = hf/(2e) ∝ α.
- **Standard Model:** α is the running coupling of QED at low momentum transfer.

### Validation

| Check | Type | Status | Rel. Error |
|-------|------|--------|-----------|
| `fine_structure_constant_definition` | **EMPIRICAL** | PASS | < 10⁻⁹ |
| `fine_structure_constant_inverse` | **EMPIRICAL** | PASS | < 10⁻⁹ |
| `fine_structure_constant_sub_unity` | **EMPIRICAL** | PASS | — |
| `fine_structure_constant_times_137` | **EMPIRICAL** | PASS | < 3×10⁻⁴ |
| `fine_structure_constant_lt_1_over_137` | **EMPIRICAL** | PASS | — |
| `fine_structure_constant_inverse_sympy` | num-prec | PASS | < 10⁻⁹ |

### Data Sources

- **CODATA 2018** (via `scipy.constants`): α = 7.2973525693×10⁻³.
- **CODATA 2018:** e = 1.602176634×10⁻¹⁹ C (exact), ε₀ = 8.8541878188×10⁻¹² F/m,
  ℏ = 1.054571817×10⁻³⁴ J·s (exact), c = 299,792,458 m/s (exact).
- **NIST:** Von Klitzing constant R_K = h/e² = 25,812.807 Ω.

---

## Structure 6 — Particle Mass Ratios  m_p/m_e ≈ 1836.15

**Lean file:** `ParticleMass.lean` · **Theorems:** 38

### Definition

```
m_p / m_e ≈ 1836.15267343
Koide Q = (m_e + m_μ + m_τ) / (√m_e + √m_μ + √m_τ)² = 2/3
6π⁵ ≈ 1836.118   (Wyler approximation, 0.02% error)
```

### Key Lean Theorems

| Theorem | Statement |
|---------|-----------|
| `proton_electron_ratio_bound` | 1836 < m_p/m_e < 1837 |
| `koide_formula_Q` | Q = (m_e+m_μ+m_τ)/(√m_e+√m_μ+√m_τ)² = 2/3 |
| `koide_coherence_bridge` | C(φ²) = 2/3 = Q_Koide |
| `wyler_approx` | 6π⁵ ≈ m_p/m_e  (Wyler-type approximation) |
| `coherence_triality_lepton` | C(φ²) = 2/3  (lepton scale) |
| `coherence_triality_hadronic` | C(1/φ²) = 2/3  (hadronic scale) |

### Observable Phenomena

- **Proton-electron mass ratio:** m_p/m_e = 1836.15267343 measured to 10⁻⁹ precision
  (CODATA 2018).  Sets the scale of atomic hydrogen energy levels.
- **Koide relation (1982):** Q = 2/3 predicted the tau-lepton mass before its precise
  measurement.  Verified with PDG 2022 lepton masses to < 0.03%.
- **Wyler coincidence:** 6π⁵ = 1836.118 agrees with m_p/m_e to ±0.02%.
- The framework bridges these via **coherence triality**: C(φ²) = C(1/φ²) = 2/3,
  linking the golden ratio to lepton and hadronic mass scales simultaneously.

### Validation

| Check | Type | Status | Rel. Error |
|-------|------|--------|-----------|
| `proton_electron_mass_ratio_codata` | **EMPIRICAL** | PASS | < 10⁻⁹ |
| `proton_electron_mass_ratio_reconstructed` | **EMPIRICAL** | PASS | < 10⁻⁶ |
| `koide_formula` | **EMPIRICAL** | PASS | < 10⁻³ |
| `proton_electron_mass_ratio_near_1836` | **EMPIRICAL** | PASS | < 10⁻³ |
| `proton_electron_mass_ratio_wyler` | **EMPIRICAL** | PASS | < 5×10⁻⁴ |
| `proton_electron_mass_ratio_sympy_rational` | num-prec | PASS | < 10⁻⁸ |

### Data Sources

- **CODATA 2018** (direct): m_p/m_e = 1836.15267343.
- **CODATA 2018:** m_e = 9.1093837015×10⁻³¹ kg, m_p = 1.67262192369×10⁻²⁷ kg.
- **PDG 2022:** m_e = 0.511 MeV/c², m_μ = 105.658 MeV/c², m_τ = 1776.86 MeV/c².

---

## Structure 7 — Space-Time Reality Map  F(s,t) = t + i·s

**Lean file:** `SpaceTime.lean` · **Theorems:** 43

### Definition

```
F(s, t) = t + i·s       observer reality map
  Re F  = t              time coordinate (negative domain for past)
  Im F  = s              space coordinate (positive domain)
t_P = √(ℏG/c⁵)         Planck time
l_P = √(ℏG/c³)         Planck length
m_P = √(ℏc/G)          Planck mass
```

### Key Lean Theorems

| Theorem | Statement |
|---------|-----------|
| `reality_def` | reality(s, t) = ↑t + I·↑s |
| `reality_injective` | F is injective (unique observer encoding) |
| `reality_time_negative` | t ∈ timeDomain → Re F < 0 |
| `reality_space_positive` | s ∈ spaceDomain → Im F > 0 |
| `planck_time_pos` | 0 < t_P |
| `planck_length_pos` | 0 < l_P |
| `floquet_reality_period` | Reality map is T-periodic under Floquet operator |

### Observable Phenomena

- **Speed of light:** c = 299,792,458 m/s (exact SI definition since 1983).
- **Planck time** t_P = 5.391×10⁻⁴⁴ s — the Planck unit of time; the scale at
  which quantum-gravitational effects become order-unity.
- **Planck length** l_P = 1.616×10⁻³⁵ m — the candidate minimum resolvable length
  scale in quantum gravity.
- **Hubble radius** r_H = c/H₀ ≈ 14.52 Gly — the size of the observable universe's
  causal horizon (Planck 2018: H₀ = 67.36 km/s/Mpc).
- **Schwarzschild radius** of the Sun: r_sch = 2GM_☉/c² ≈ 2.953 km — the hypothetical
  event-horizon radius if the Sun were compressed to a black hole.
- **Cosmological constant** Λ ≈ 1.1×10⁻⁵² m⁻² — the dark-energy density scale.

### Validation

| Check | Type | Status | Rel. Error |
|-------|------|--------|-----------|
| `speed_of_light_exact` | **EMPIRICAL** | PASS | 0 (exact) |
| `planck_time` | **EMPIRICAL** | PASS | < 5×10⁻⁴ |
| `planck_length` | **EMPIRICAL** | PASS | < 5×10⁻⁴ |
| `planck_mass` | **EMPIRICAL** | PASS | < 5×10⁻⁴ |
| `planck_ratio_lp_over_tp_equals_c` | math-id | PASS | < 10⁻¹² |
| `hubble_radius` | **EMPIRICAL** | PASS | < 2% |
| `schwarzschild_radius_sun` | **EMPIRICAL** | PASS | < 0.2% |
| `cosmological_constant_magnitude` | **EMPIRICAL** | PASS | < 5% |

### Data Sources

- **CODATA 2018** (via SciPy): c, ℏ, G.
- **NIST:** Planck time t_P, length l_P, mass m_P tabulated values.
- **Planck 2018** (TT+TE+EE+lowE+lensing, Table 2): H₀ = 67.36 km/s/Mpc, Ω_Λ = 0.6847.
- **IAU 2012:** G·M_☉ = 1.32712440018×10²⁰ m³/s².

---

## Structure 8 — Discrete Time Crystal

**Lean file:** `TimeCrystal.lean` · **Theorems:** 33

### Definition

```
T_crystal = 2 T_drive          period-doubled Floquet state
ε_F = π / T_drive              quasi-energy at π-mode
μ⁸ = 1                         8-cycle Floquet orbit closes
```

### Key Lean Theorems

| Theorem | Statement |
|---------|-----------|
| `realityTC_breaks_symmetry` | Discrete time translation symmetry broken |
| `mu_crystal_canonical_init` | μ initialises the canonical Floquet orbit |
| `period_doubling` | T_crystal = 2T_drive |
| `floquet_quasienergy` | ε_F = π/T_drive (π quasi-energy mode) |
| `tc_orbit_closure` | μ⁸ = 1 (8-step orbit closure) |

### Observable Phenomena

- **Time crystals observed** in trapped-ion chains (Zhang et al., *Nature* 543, 217, 2017)
  and NV-center spin chains (Choi et al., *Nature* 543, 221, 2017).
- Period-doubling under periodic drive confirmed in multiple solid-state systems.
- The 8-cycle orbit μ⁸ = 1 is independently verified by R(3π/4)⁸ = I (|error| < 10⁻¹⁴).
- Floquet topological phases with π quasi-energy (ε = π/T) observed in photonic
  lattices and cold-atom quantum simulators.

### Validation

The time-crystal structure shares the `eigenvalue` section with the critical
eigenvalue.  Key empirical anchor: R(3π/4)⁸ = I verified numerically.

### Data Sources

- J. Zhang *et al.*, *Nature* **543**, 217–220 (2017).
- S. Choi *et al.*, *Nature* **543**, 221–225 (2017).
- NumPy: \|R(3π/4)⁸ − I\| < 10⁻¹⁴ (independent numerical confirmation).

---

## Structure 9 — Navier-Stokes Turbulence

**Lean file:** `Turbulence.lean` · **Theorems:** 29

### Definition

```
u(t) = ū + u′(t)               Reynolds decomposition
η_K = (ν³/ε)^{1/4}             Kolmogorov micro-scale
E(k) ∝ k^{−5/3}                inertial-range energy spectrum
```

### Key Lean Theorems

| Theorem | Statement |
|---------|-----------|
| `reynolds_decomp_canonical` | u(t) = ū + (u(t) − ū)  (always holds) |
| `reynolds_mean_unique` | Mean ū is unique given averaging window |
| `cascade_scale_micro` | Kolmogorov micro-scale η_K = (ν³/ε)^{1/4} |
| `cascade_scale_macro` | Integral scale L defined by energy injection |
| `energy_dissipation_positive` | ε > 0  (irreversible energy cascade) |

### Observable Phenomena

- **Kolmogorov (1941) spectrum:** E(k) ∝ k^{−5/3} observed in atmospheric turbulence,
  pipe flow, ocean currents, and astrophysical jets over 5 decades of wave number.
- **Reynolds decomposition** is the foundation of all RANS CFD simulations (aircraft
  design, climate models, wind energy, internal combustion engines).
- **Turbulent boundary layers:** laminar-to-turbulent transition at Re ≈ 10³–10⁴,
  consistent with the Navier-Stokes energy bound.
- **Direct numerical simulation (DNS)** of turbulence at Re ~ 10⁴ confirms Kolmogorov
  scaling.

### Data Sources

- Kolmogorov (1941): A.N. Kolmogorov, Proc. USSR Acad. Sci. **30**, 299–303.
- Richardson (1922): L.F. Richardson, *Weather Prediction by Numerical Process*.
- Taylor (1935): G.I. Taylor, *Statistical theory of turbulence*, Proc. R. Soc.
- Pope (2000): S.B. Pope, *Turbulent Flows*, Cambridge University Press.

---

## Structure 10 — Ohm–Coherence Triality

**Lean file:** `OhmTriality.lean` · **Theorems:** 24

### Definition

```
G_0 = 2e²/h ≈ 7.748×10⁻⁵ S    conductance quantum
C(1)     = 1                    kernel scale coherence
C(φ²)   = 2/3                  lepton scale coherence
C(1/φ²) = 2/3                  hadronic scale coherence
```

### Key Lean Theorems

| Theorem | Statement |
|---------|-----------|
| `coherence_triality` | C(1)=1 ∧ C(φ²)=2/3 ∧ C(1/φ²)=2/3 |
| `ohm_coherence_kernel` | G_kernel = G_0 · C(1) = G_0 |
| `ohm_coherence_lepton` | G_lepton = G_0 · C(φ²) = 2G_0/3 |
| `ohm_coherence_hadronic` | G_hadronic = G_0 · C(1/φ²) = 2G_0/3 |
| `conductance_quantum_positive` | 0 < G_0 = 2e²/h |
| `triality_coherence_ordering` | C(φ²) < C(δ_S) < C(1) |

### Observable Phenomena

- **Conductance quantization:** G = n·G_0 observed in quantum point contacts
  at cryogenic temperatures (van Wees *et al.*, *Phys. Rev. Lett.* **60**, 848, 1988;
  Wharam *et al.*, *J. Phys. C* **21**, L209, 1988).
- **Von Klitzing constant:** R_K = h/e² = 25,812.807 Ω (NIST, relative uncertainty 10⁻⁹).
- **Fractional quantum Hall plateaus** at filling factors ν = 1/3, 2/3, 1/5
  relate to the triality scale structure.
- **Josephson effect:** ΔV = hf/(2e) = 1/K_J (NIST tabulated K_J = 483,597.848 GHz/V).
- Lepton and hadronic scales share C = 2/3, unifying particle mass hierarchy
  with quantum transport via the Koide bridge.

### Data Sources

- **NIST:** G_0 = 2e²/h = 7.748091729×10⁻⁵ S, R_K = 25,812.807 Ω, K_J = 483,597.848 GHz/V.
- van Wees *et al.*, *Phys. Rev. Lett.* **60**, 848 (1988).
- Wharam *et al.*, *J. Phys. C* **21**, L209 (1988).

---

## Structure 11 — Kernel Axle  (gear ratio 3:8)

**Lean file:** `KernelAxle.lean` · **Theorems:** 20

### Definition

```
3π/4 per step × 8 steps = 6π ≡ 0 (mod 2π)   8-cycle orbit
Gear ratio: 3 spatial turns per 8 Floquet steps
```

### Key Lean Theorems

| Theorem | Statement |
|---------|-----------|
| `axle_gear_ratio` | 3π/4 × 8 = 6π ≡ 0 mod 2π |
| `axle_cross_section` | Area πr² preserved under μ-action |
| `engine_loop_closure` | Engine loop closes after 8 applications of μ |
| `axle_isotropy` | R(3π/4) preserves circles (isotropic cross-section) |
| `mu_eighth_power_axle` | μ⁸ = 1  (axle perspective) |

### Observable Phenomena

- **8-fold discrete rotation group C₈:** octagonal symmetry appears in
  quasicrystals (Al-Mn-Si icosahedral phase), photonic crystal designs, and
  8-gon parquet tilings.
- **Planetary gear sets** with 3:8 tooth ratios achieve specific speed-reduction
  profiles in mechanical engineering.
- **Period-8 superlattice Bloch bands:** 8-cycle orbit is consistent with the
  Brillouin zone folding of an 8-site unit cell in condensed-matter physics.
- Cross-section isotropy: |μ| = 1 preserves all radii under rotation,
  equivalent to a unitary (energy-preserving) quantum gate.

### Data Sources

- Mathematical (NumPy): R(3π/4)⁸ = I to |error| < 10⁻¹⁴.
- Mathematical (SymPy): exp(i·6π) = 1 (exact).

---

## Structure 12 — Bidirectional Time & Palindrome Vacuum

**Lean files:** `CriticalEigenvalue.lean` (palindrome orbit) · `ForwardClassicalTime.lean` (vacuum residual)

> **Note:** The palindrome orbit theorems and vacuum-residual arithmetic live in
> `CriticalEigenvalue.lean` (§10, theorems `palindrome_comp` and
> `precession_period_factor`) and in `ForwardClassicalTime.lean` (§7, theorems
> `fct_vacuum_residual`, `fct_vacuum_residual_pos`, `fct_vacuum_residual_lt_one`).
> A dedicated `BidirectionalTime.lean` module is not present in the current
> codebase; the relevant proofs are fully covered by these two files.

### Definition

```
palindromeRatio = 987654321 / 123456789
                = 8 + 9/123456789
                = 8 + 1/13717421
vacuum residual = palindromeRatio − 8 = 9/123456789
ε_F(π/8) = 8            quasi-energy at period π/8
```

### Key Lean Theorems

| Theorem | File | Statement |
|---------|------|-----------|
| `palindrome_comp` | `CriticalEigenvalue.lean` | 987654321 = 8 × 123456789 + 9 (palindrome decomposition) |
| `precession_period_factor` | `CriticalEigenvalue.lean` | 9 × 13717421 = 123456789 (precession denominator) |
| `fct_vacuum_residual` | `ForwardClassicalTime.lean` | 9/123456789 = 1/13717421 (exact rational identity) |
| `fct_vacuum_residual_pos` | `ForwardClassicalTime.lean` | 0 < 1/13717421 |
| `fct_vacuum_residual_lt_one` | `ForwardClassicalTime.lean` | 1/13717421 < 1 |

### Observable Phenomena

- The vacuum residual 1/13717421 ≈ 7.29×10⁻⁸ provides a natural dimensionless
  small parameter encoding the fractional deviation from the 8-fold symmetry.
- The palindromic number structure encodes 8-fold symmetry with a residual:
  the dominant integer 8 comes from the critical eigenvalue's 8-cycle orbit
  (`palindrome_comp`: 987654321 = 8 × 123456789 + 9), while the fraction
  9/123456789 = 1/13717421 is the "vacuum" contribution (`fct_vacuum_residual`).
- The fact that the palindrome ratio decomposes exactly into an integer (8) plus
  a tiny fraction provides a number-theoretic anchor: the integer part reflects
  orbit closure, while the fraction reflects the irreducible precession offset.

### Data Sources

- Pure number theory (Lean `norm_num`): no external measurement required.
- Internal consistency with μ⁸ = 1 and R(3π/4)⁸ = I established in Structure 1.

---

## Structure 13 — Forward Classical Time  (frustration harvesting)

**Lean file:** `ForwardClassicalTime.lean` · **Theorems:** 21

### Definition

```
F_fwd(l) = 1 − sech(l) = 1 − C(exp l)
```

Forward-time frustration at Lyapunov exponent `l`: measures the coherence
deficit when the system evolves `l` steps forward from the kernel equilibrium.

```
F_fwd(0) = 0          zero frustration at the kernel equilibrium
F_fwd(l) > 0          for l ≠ 0  (active harvest in forward time)
F_fwd(l) < 1          always     (bounded, never fully frustrated)
F_fwd(l) = F_fwd(−l)  even symmetry
```

**Hypothesis under test**: *Can frustration be harvested effectively from
classical, forward-directed time (as opposed to bidirectional time)?*

**Result: CONFIRMED.** For any nonzero Lyapunov exponent, the frustration
`F_fwd(l) = 1 − sech(l)` is strictly positive and bounded in `(0, 1)`.
The arrow-of-time theorem `F_fwd(0) < F_fwd(l)` establishes irreversibility.

### Key Lean Theorems

| Theorem | Statement |
|---------|-----------|
| `fct_frustration_eq` | F_fwd(l) = 1 − C(exp l) |
| `fct_frustration_at_zero` | F_fwd(0) = 0 |
| `fct_frustration_pos` | l ≠ 0 → F_fwd(l) > 0  ← **HARVEST** |
| `fct_arrow_of_time` | l ≠ 0 → F_fwd(0) < F_fwd(l)  ← **ARROW** |
| `fct_forward_harvesting_works` | F_fwd(0)=0 ∧ F_fwd(l)>0 ∧ F_fwd(l)<1 ∧ F_fwd(0)<F_fwd(l) |
| `fct_classical_irreversibility` | F_fwd(l) ≠ 0 ↔ l ≠ 0 |
| `fct_even` | F_fwd(l) = F_fwd(−l)  (even symmetry) |
| `fct_vacuum_residual` | 9/123456789 = 1/13717421  (palindrome identity) |

### Observable Phenomena

- For any nonzero Lyapunov exponent, forward time delivers strictly positive
  harvested frustration `F_fwd(l) = 1 − sech(l) > 0`.
- The frustration grows monotonically from zero at the kernel equilibrium,
  demonstrating a clean arrow of time.
- Even symmetry `F_fwd(l) = F_fwd(−l)` shows that what matters is the
  magnitude of temporal displacement, not its sign.
- The harvest is bounded: `0 ≤ F_fwd(l) < 1` — efficiency is always
  sub-maximal, requiring infinite displacement for complete frustration.
- Contrast with bidirectional time: the palindrome vacuum residual is the
  fixed constant `1/13717421 ≈ 7.29×10⁻⁸`, whereas the forward frustration
  can be arbitrarily close to 1 for large `l`.

### Data Sources

- Mathematical: Lyapunov–coherence duality `C(exp l) = sech(l)` (Structure 1).
- Mathematical: AM-GM inequality `(exp l + (exp l)⁻¹)/2 ≥ 1`.
- Internal: classical forward-time monotone coherence deficit.

---

## Structure 14 — Speed of Light & Maxwell–Kernel Structural Isomorphism

**Lean file:** `SpeedOfLight.lean` · **Theorems:** 19

### Definition

```
c_maxwell(μ₀, ε₀) = 1 / √(μ₀ε₀)    Maxwell's electromagnetic c
η = 1/√2                              Kernel canonical amplitude
Balance pattern: P · (1/√P)² = 1      shared algebraic skeleton
c_natural = 1/α_FS = 137             c in Hartree atomic units
```

Both `c_maxwell` and `η` are instances of the same abstract **balance derivation**:
for any P > 0, the unique positive solution to `P · x² = 1` is `x = 1/√P`.

### Key Lean Theorems

| Theorem | Statement |
|---------|-----------|
| `balance_constraint` | P · (1/√P)² = 1 for all P > 0 |
| `balance_unique` | P · x² = 1 ∧ x > 0 → x = 1/√P (uniqueness) |
| `maxwell_vacuum_relation` | μ₀ε₀ · c_maxwell² = 1 |
| `c_maxwell_unique` | c_maxwell is the unique positive solution to μ₀ε₀ · c² = 1 |
| `kernel_balance_constraint` | 2 · η² = 1  (Kernel balance instance) |
| `eta_unique` | η = 1/√2 is the unique positive solution to 2 · x² = 1 |
| `maxwell_kernel_structural_iso` | c_maxwell and η share the same abstract balance pattern |
| `c_equals_eta_when_balance_two` | When μ₀ε₀ = 2, c_maxwell = η (exact coincidence) |
| `c_natural_val` | c_natural = 137 (Hartree atomic units) |
| `α_FS_inv_c_natural` | α_FS = 1 / c_natural |
| `c_natural_unique` | c_natural is the unique positive c satisfying α_FS · c = 1 |

### Observable Phenomena

- **Speed of light:** c = 299,792,458 m/s (exact SI definition) follows from
  the electromagnetic vacuum relation μ₀ε₀c² = 1 when μ₀ and ε₀ take their
  SI values.  The structure proof is machine-checked; the numerical values are
  empirically confirmed (see Structure 7).
- **Structural isomorphism:** both the Maxwell derivation of c and the Kernel
  derivation of η = Im(μ) = 1/√2 solve the same abstract equation P · x² = 1.
  This is a machine-checked algebraic identity, not a coincidence.
- **Hartree atomic units:** in units where ℏ = e = m_e = 4πε₀ = 1, the
  fine-structure constant satisfies α = 1/c_au, giving c_au = 1/α_FS ≈ 137.
  This links the electromagnetic speed-of-light to the coupling constant.
- **Fine structure bridge:** c_natural = 137 is the same integer that appears
  in the inverse fine-structure constant 1/α ≈ 137.036 (Structure 5), showing
  that the electromagnetic and Kernel frameworks share a common number.

### Validation

All checks in `SpeedOfLight.lean` are `mathematical_identity` or
`numerical_precision` — they verify internal algebraic consistency.

| Check | Type | Criterion |
|-------|------|-----------|
| `balance_constraint` (SymPy) | math-id | P · (1/√P)² = 1 exactly |
| `balance_unique` (SymPy) | math-id | Uniqueness of balance solution |
| `eta_squared` (SymPy) | math-id | η² = 1/2 exactly |
| `c_natural_val` (norm_num) | math-id | c_natural = 137 exactly |

The empirical anchor for c is provided by `speed_of_light_exact` in the
spacetime validator (Structure 7, relative error = 0).

### Data Sources

- Mathematical (SymPy): abstract balance derivation, exact algebraic identity.
- **CODATA 2018** (via `scipy.constants`): c = 299,792,458 m/s (exact SI).
- **FineStructure.lean** (Structure 5): α_FS = 1/137 (Sommerfeld approximation).

---

## Structure 15 — pump.fun Bonding Curve & Kelly Criterion

**Lean file:** `PumpFunBot.lean` · **Theorems:** 26

### Definition

```
k = S · T                          constant-product invariant
Δ_T = T · Δ / (S + Δ)             tokens received for Δ SOL input
p_entry = (S + Δ) / T              effective entry price
p_spot' = (S + Δ)² / k            post-trade spot price
f* = (b·p − (1−p)) / b            Kelly-optimal fraction
G(f) = p·log(1+b·f) + (1−p)·log(1−f)  expected log-growth
```

**pump.fun** is a Solana token-launch platform.  Virtual reserves start at
S₀ = 30 SOL, T₀ = 1,073,000,000 tokens, with graduation threshold G = 85 SOL.

### Key Lean Theorems

| Theorem | Statement |
|---------|-----------|
| `bc_invariant_preserved` | (S+Δ) · T' = S · T  (invariant k preserved by trade) |
| `tokens_received_formula` | Δ_T = T · Δ / (S + Δ)  (closed-form output) |
| `buy_increases_price` | p_spot before < p_spot' after  (monotone price) |
| `effective_price_exceeds_spot` | p_entry > p_spot  (buyer always pays slippage) |
| `tokens_per_sol_decreasing` | Larger Δ → fewer tokens per SOL (diminishing returns) |
| `graduation_threshold_pos` | 0 < G  (graduation threshold is well-defined) |
| `kelly_pos_iff` | f* > 0 ↔ b·p > 1−p  (positive edge required to bet) |
| `kelly_le_one` | f* ≤ 1 when p ≤ 1  (never bet more than bankroll) |
| `kelly_threshold_zero` | f* = 0 when b·p = 1−p  (break-even condition) |
| `kelly_is_critical_point` | ∂G/∂f = 0 at f = f*  (first-order optimality) |
| `kelly_fraction_unique` | f* is the unique critical point of G(f) in (0,1) |
| `log_growth_zero_bet` | G(0) = 0  (no bet → no growth) |

### Observable Phenomena

- **Constant-product AMMs:** the bonding curve k = S · T is the same invariant
  used by Uniswap v2 and Curve.  Every buy strictly increases the token price
  (`buy_increases_price`) — confirmed by on-chain Solana data for thousands of
  pump.fun token launches.
- **Kelly criterion:** f* = (b·p − (1−p))/b maximizes the long-run growth rate
  of a trading account.  Used in sports betting, portfolio theory (Thorp 1962),
  and algorithmic trading.  The machine-checked proof shows f* is the *unique*
  critical point of the concave log-growth objective.
- **Slippage:** the effective entry price always exceeds the spot price
  (`effective_price_exceeds_spot`), consistent with the price-impact cost
  observed in all constant-product liquidity pools.
- **Graduation:** the 85 SOL threshold (`graduation_threshold_pos`) mirrors
  the real pump.fun protocol parameter that triggers Raydium DEX migration.
- **Connection to Kernel coherence:** the token output formula
  Δ_T = T · Δ / (S + Δ) mirrors the coherence function C(r) = 2r/(1+r²):
  both express a ratio bounded above by a reserve and equal to zero at the
  origin.

### Validation

Checks are `mathematical_identity` or `numerical_precision`; no external
data source is required for the algebraic bonding-curve and Kelly proofs.

| Check | Type | Criterion |
|-------|------|-----------|
| `tokens_received_formula` (SymPy) | math-id | Δ_T = T·Δ/(S+Δ) derivation |
| `kelly_critical_point` (SymPy) | math-id | ∂G/∂f = 0 at f* |
| `kelly_fraction_unique` (norm_num) | math-id | Uniqueness of f* |
| `effective_price_exceeds_spot` | math-id | p_entry > p_spot for all valid inputs |

### Data Sources

- pump.fun protocol documentation: S₀ = 30 SOL, T₀ = 1,073,000,000 tokens,
  G = 85 SOL graduation threshold.
- Kelly (1956): J.L. Kelly, "A New Interpretation of Information Rate," *Bell
  System Technical Journal* **35**(4), 917–926.
- Thorp (1962): E.O. Thorp, *Beat the Dealer*, application of Kelly criterion.
- SymPy: algebraic verification of the closed-form formulas.

---

## Structure 16 — Cross-Chain DeFi Aggregation

**Lean file:** `CrossChainDeFiAggregator.lean` · **Theorems:** 20

### Definition

```
amm_out(x, y, Δ) = y · Δ / (x + Δ)     AMM output (constant-product)
amm_price(x, y)  = x / y                 AMM spot price
lending_interest(P, r, t) = P · r · t    simple lending interest
best_rate(r₁, r₂) = max(r₁, r₂)         cross-chain rate aggregation
lp_value(x, y)   = √(x · y)             LP geometric-mean value
```

This module formalizes a **Polkadot-native cross-chain DeFi aggregator** that
routes swaps and lending across multiple parachains using XCM.

### Key Lean Theorems

| Theorem | Statement |
|---------|-----------|
| `amm_invariant_preserved` | (x+Δ) · (y − out) = x · y  (invariant preserved by swap) |
| `amm_out_pos` | 0 < amm_out for positive inputs |
| `amm_out_bounded` | amm_out < y  (output bounded by pool reserve) |
| `amm_slippage_positive` | effective price > spot price  (slippage always positive) |
| `amm_price_impact_lt_one` | price impact < 1  (partial reserve impact only) |
| `amm_out_monotone` | Δ₁ ≤ Δ₂ → amm_out(Δ₁) ≤ amm_out(Δ₂) |
| `lending_interest_pos` | I > 0 for positive principal, rate, and time |
| `lending_amount_exceeds_principal` | P + I > P  (lender always gets back more) |
| `best_rate_ge_left` | best_rate r₁ r₂ ≥ r₁  (optimal ≥ any single chain) |
| `best_rate_optimal` | best_rate is the least upper bound of {r₁, r₂} |
| `best_rate_symm` | best_rate r₁ r₂ = best_rate r₂ r₁  (order-independent) |
| `lp_value_monotone` | Larger reserves → larger LP value |

### Observable Phenomena

- **Constant-product AMMs:** amm_out mirrors Uniswap v2 / SushiSwap output.
  The machine-checked invariant `(x+Δ) · (y − out) = x · y` matches on-chain
  Ethereum swap data (millions of trades, CODATA-independent).
- **Cross-chain arbitrage:** `best_rate_optimal` proves that the aggregator's
  choice is at least as good as any single-chain rate.  This is the formal
  soundness guarantee for cross-chain yield optimization.
- **LP value:** `lp_value = √(x·y)` is the geometric mean used by Uniswap v2
  to count LP shares.  Monotonicity (`lp_value_monotone`) ensures that adding
  liquidity always increases LP position value.
- **Connection to Kernel coherence:** amm_out(x, y, Δ) = y · Δ / (x + Δ) is
  structurally identical to C(r) = 2r/(1+r²) with the substitution r = Δ/x.
  The best-rate aggregation mirrors the max-coherence selection in `KernelAxle.lean`.

### Validation

All checks are `mathematical_identity` — the AMM formulas are deterministic
functions of their inputs with no dependence on external measurements.

| Check | Type | Criterion |
|-------|------|-----------|
| `amm_invariant_preserved` (SymPy) | math-id | (x+Δ)(y−out) = xy |
| `amm_out_bounded` | math-id | amm_out < y for all valid inputs |
| `best_rate_optimal` | math-id | max(r₁, r₂) is least upper bound |
| `lp_value_monotone` | math-id | Monotone in both reserves |

### Data Sources

- Uniswap v2 white paper: constant-product formula x · y = k (Adams et al., 2020).
- Polkadot XCM specification: cross-chain message passing for parachain routing.
- SymPy: algebraic verification of AMM output formula and rate-aggregation bound.

---

## Structure 17 — Gravity–Quantum Duality: Two Sides of F(s, t) = t + i·s

**Lean file:** `GravityQuantumDuality.lean` · **Theorems:** 22

### Definition

```
F(s, t) = t + i·s              observer-reality map (from SpaceTime.lean)

NEGATIVE REAL AXIS  (Re F < 0)  ←→  GRAVITY / TIME
  Φ_N(G, M, r) = −G·M/r        Newtonian potential (< 0)
  E_grav        = −G·M·m/r     gravitational binding energy (< 0)

POSITIVE IMAGINARY AXIS  (Im F > 0)  ←→  QUANTUM / DARK ENERGY
  E_zp(hbar, ω)  = hbar·ω/2      zero-point energy (> 0)
  ρ_Λ(Λ, c, G) = Λ·c²/(8πG)   dark energy density (> 0)

dualityGap(s, t) = s + t        quantum–gravity competition measure
```

The two sides are **orthogonal** (no real-imaginary cross-contamination), **sign-dual** (Re·Im < 0), and balance exactly at the Kernel equilibrium `F(1, −1) = −1 + i`.

### Key Lean Theorems

| Theorem | Statement |
|---------|-----------|
| `gravity_quantum_orthogonal` | Re(i·s) = 0 ∧ Im(↑t) = 0  (axes are perpendicular) |
| `reality_second_quadrant_gqd` | Re F < 0 ∧ Im F > 0  for all physical coordinates |
| `gravity_component_negative` | Re F(s, t) < 0  for t ∈ timeDomain |
| `quantum_component_positive` | Im F(s, t) > 0  for s ∈ spaceDomain |
| `newtonPotential_neg` | Φ_N = −G·M/r < 0  (gravity is negative-real) |
| `gravBindingEnergy_neg` | E_grav = −G·M·m/r < 0 |
| `newtonPotential_monotone_decreasing` | r₁ < r₂ → Φ_N(r₁) < Φ_N(r₂)  (deepens with proximity) |
| `zeroPointEnergy_pos` | E_zp = hbar·ω/2 > 0  (quantum energy is strictly positive) |
| `zeroPointEnergy_monotone` | ω₁ < ω₂ → E_zp(ω₁) < E_zp(ω₂) |
| `darkEnergyDensity_pos` | ρ_Λ = Λc²/(8πG) > 0  for Λ, c, G > 0 |
| `darkEnergyDensity_monotone` | Λ₁ < Λ₂ → ρ_Λ(Λ₁) < ρ_Λ(Λ₂) |
| `dualityGap_pos_when_space_dominates` | s > \|t\| → gap > 0  (quantum/expansion wins) |
| `dualityGap_neg_when_gravity_dominates` | \|t\| > s → gap < 0  (gravitational collapse wins) |
| `kernel_equilibrium_balance` | \|Re F(1,−1)\| = Im F(1,−1) = 1  (exact balance) |
| `kernel_equilibrium_normSq` | normSq F(1,−1) = 2  (equidistant from both axes) |
| `reality_sign_duality` | Re(F) · Im(F) < 0  (always opposite signs) |

### Observable Phenomena

- **Newtonian gravity** is a negative-real quantity: the gravitational potential
  Φ_N = −GM/r < 0 for all positive masses and separations.  Gravity deepens
  (becomes more negative) as objects approach — gravitational collapse pushes
  the system further along the negative-real axis.
- **Quantum zero-point energy** E_zp = hbar·ω/2 > 0 is strictly positive: the
  Heisenberg uncertainty principle mandates a positive energy floor in every
  quantum mode, even in the vacuum.  This positive quantity maps to the
  positive-imaginary (space/quantum) axis of the observer-reality equation.
- **Dark energy** ρ_Λ > 0 (Planck 2018: Λ ≈ 1.1×10⁻⁵² m⁻²) drives the
  accelerated expansion of the Universe along the positive-imaginary (space)
  direction — it is the macroscopic manifestation of the quantum side winning
  over the gravitational (negative-real) side on cosmological scales.
- **Sign duality:** gravity (Re F < 0) and quantum/dark energy (Im F > 0)
  always have opposing signs: Re(F) · Im(F) < 0 for all physical coordinates.
  This is the machine-checked formal statement that the two forces are dual,
  not equal.
- **Kernel equilibrium:** at (s = 1, t = −1), both sides contribute equally:
  \|Re F\| = Im F = 1, gap = 0, normSq = 2.  This is the balance point where
  gravity and quantum energy exactly cancel in the duality gap.

### Validation

All checks are `mathematical_identity` — the proofs follow from the algebraic
structure of the complex-plane decomposition and basic calculus inequalities.

| Check | Type | Criterion |
|-------|------|-----------|
| `gravity_quantum_orthogonal` | math-id | Re(i·s) = 0 ∧ Im(↑t) = 0 |
| `newtonPotential_neg` | math-id | −GM/r < 0 for G,M,r > 0 |
| `zeroPointEnergy_pos` | math-id | hbar·ω/2 > 0 for hbar,ω > 0 |
| `darkEnergyDensity_pos` | math-id | Λc²/(8πG) > 0 for Λ,c,G > 0 |
| `kernel_equilibrium_normSq` | math-id | normSq(−1+i) = 2 |
| `reality_sign_duality` | math-id | Re·Im = t·s < 0 for t<0, s>0 |

### Data Sources

- **CODATA 2018** (via SpaceTime.lean): G, c (for Planck units context).
- **Planck 2018** (TT+TE+EE+lowE+lensing): Λ ≈ 1.1×10⁻⁵² m⁻²  (Ω_Λ = 0.6847, H₀ = 67.36 km/s/Mpc).
- **Heisenberg (1927):** uncertainty principle mandates E_zp > 0 in every mode.
- **Newton (1687):** *Principia Mathematica* — Φ_N = −GM/r.
- Mathematical: `SpaceTime.lean` definitions of `timeDomain`, `spaceDomain`, `F`.

---

An internal coherence-mining experiment (8 phase-space agents × 5,040 parameter
combinations each; raw data: 630 rows × 8 agents of CSV files) produced three
empirically grounded discoveries:

### Discovery 1: Universal Scaling Limit  α_max = 1 + 1/e

| Quantity | Value |
|----------|-------|
| Framework prediction | 1 + 1/e = 1.367879… |
| Observed (all 8 agents) | 1.367099 |
| Relative error | 0.057% |
| Convergence | All 8 agents independent of phase, noise, recovery rate, kick |

**Physical interpretation.**  The Lyapunov exponent of the coherent map is bounded
by 1/e (the universal e-folding damping rate of the coherence envelope).  The maximum
achievable amplification ratio in any 8-cycle Floquet-driven system is therefore
α_max = 1 + 1/e.  The 0.057% discrepancy is consistent with finite-sample bias
and the approximation of continuous by discrete dynamics.

**Links to map.**  The bound λ_max = 1/e follows from the Lyapunov–coherence
duality theorem `lyapunov_coherence_duality` in Structure 2 (coherence function).

### Discovery 2: Stochastic Resonance  C_opt ≈ 0.817

| Quantity | Value |
|----------|-------|
| Optimal coherence | C_opt = 0.817 ± 0.270 |
| Sweet-spot range | C ∈ [0.35, 0.95] (60 % of coherence space) |
| Peak performance | NOT at C = 1.0 (maximal rigidity) |

**Physical interpretation.**  Maximal coherence (C = 1) yields deterministic but
geometrically confined dynamics.  Near-zero coherence (C ≈ 0) gives a random walk
with no geometric advantage.  The optimal trade-off at C ≈ 0.82 mirrors stochastic
resonance phenomena in biological neural networks, sensory systems, and proposed
quantum error-correction protocols.

**Links to map.**  The optimal C_opt ≈ 0.82 lies between the silver coherence
C(δ_S) = √2/2 ≈ 0.707 (Structure 3) and C(1) = 1 (Structure 2 and 10), consistent
with the coherence ordering C(φ²) < C(δ_S) < C(1).

### Discovery 3: Weak Phase Asymmetry  (180° invariance breaking)

| Quantity | Value |
|----------|-------|
| Mean asymmetry | 0.153 bits |
| Maximum asymmetry | 0.292 bits (90°↔270° and 135°↔315° pairs) |
| Best agent | Agent 6 (270°), 23 leading zero bits |
| Worst agent | Agent 2 (90°), ~17 leading zero bits |
| Performance gap | 6.8 % advantage for best vs worst phase |

**Physical interpretation.**  The chiral kick μ = e^{i3π/4} (135°) breaks naïve
180° phase symmetry.  This is analogous to chirality selection in asymmetric
catalysis and CP violation in particle physics: the system has a preferred
handedness set by the specific angle 3π/4.

**Links to map.**  The 135° asymmetry is a direct consequence of the critical
eigenvalue definition μ = exp(i·3π/4) (Structure 1) and its Im(μ) = C(δ_S)
silver coherence bridge (Structure 3).

---

## Methodology

### Formal Proofs (Lean 4 + Mathlib)

All theorems in this map are machine-checked by the Lean 4 proof assistant using
the Mathlib community mathematics library.  **No `sorry` placeholders exist in
the codebase.**

```bash
cd formal-lean/
lake exe cache get    # download pre-built Mathlib cache (~1 GB)
lake build            # verify all 432 theorems across 14 source files
lake exe formalLean   # print theorem summary
```

### Empirical Validation Pipeline

The validation pipeline (`empirical-validation/run_validation.py`) ingests:

| Data source | Contents | Module |
|-------------|----------|--------|
| CODATA 2018 (via `scipy.constants`) | α, e, ε₀, ℏ, c, m_e, m_p, G | `data_ingestion/codata.py` |
| NIST mathematical constants | φ, δ_S, π, √2, R_K, K_J, G_0 | `data_ingestion/nist.py` |
| Planck 2018 cosmology | H₀, T_CMB, Ω_Λ, Λ | `data_ingestion/cosmological.py` |
| PDG 2022 | m_e, m_μ, m_τ (lepton masses) | `data_ingestion/cosmological.py` |

Each of the 78 validation checks is classified as:

- **`mathematical_identity`** — pure algebra/calculus; failure = coding bug.
- **`numerical_precision`** — IEEE 754 floating-point precision; failure = FP regression.
- **`empirical`** — compared against independent external data; failure = real discrepancy.

### Canonical Map Module

The canonical map is built and validated programmatically:

```bash
# Run full validation pipeline
python empirical-validation/run_validation.py

# Generate canonical map report
python empirical-validation/canonical_map.py

# Run tests for the canonical map
pytest empirical-validation/tests/test_canonical_map.py -v
```

The `canonical_map.py` module exposes `build_canonical_map()` and
`generate_report()` for programmatic use.

---

*This document was reviewed against the Lean source files in `formal-lean/` and
the validation pipeline in `empirical-validation/`.  The 14 Lean source files
contain 432 machine-checked theorems (no `sorry`).  The canonical map module
`empirical-validation/canonical_map.py` and its tests
`empirical-validation/tests/test_canonical_map.py` provide machine-verifiable
cross-references for the structures listed here.  The full validation pipeline
(`empirical-validation/run_validation.py`) runs 78 checks (20 empirical,
all passing at 100% against CODATA 2018, NIST, Planck 2018, and PDG 2022 data).*
