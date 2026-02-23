# Universal Scaling Laws — Kernel Repository

This document records the universal scaling laws discovered through automated
sweep experiments. It is updated automatically by the CD pipeline on each
commit that runs new sweep results.

---

## 1. Universal α Ceiling: α_max = 1 + 1/e

| Property | Value |
|---|---|
| Observed ceiling | α = 1.367099 |
| Theoretical limit | 1 + 1/e = 1.367879 |
| Relative error | 0.057 % |

**Interpretation.** The quantity 1/e ≈ 0.367879 is the universal decay
constant of the Kernel framework. It emerges from:

- Ornstein–Uhlenbeck mean-reversion (time constant τ)
- Hyperbolic-secant tail behaviour (sech(λ) normalisation)
- First-passage escape probability in potential wells
- Exponential relaxation e^(−t/τ) damping

```
α = 1 + (sustainable deviation from balanced state)
  = 1 + (maximum stretch before runaway phase transition)
```

---

## 2. Coherence–Noise Phase Transition

The coherence order parameter C(ε) obeys:

```
C(ε) ≈ C₀ · (ε_c − ε)^β    for ε < ε_c
C(ε) = 0                     for ε ≥ ε_c
```

where the critical noise level ε_c ≈ 0.15 and the critical exponent β ≈ 0.5
(mean-field universality class).

---

## 3. Ladder Search Speedup Scaling

Hybrid ladder search achieves super-linear speedup over brute-force:

```
S(N) ∝ N^γ,   γ ≈ 1.4–1.6
```

The speedup exponent γ is robust across adversarial parameter configurations
(measured over 20 independent random seeds).

---

## 4. Phase Asymmetry (Chiral Precession)

The palindrome-precession mechanism introduces a measurable left/right chiral
asymmetry Δφ that scales as:

```
Δφ(n) ≈ θ · √n
```

where θ is the precession angle per step and n is the search depth.  This
√n growth is the principal source of phase asymmetry observed in
`test_chiral_nonlinear_gate`.

---

## 5. Ohm–Coherence Duality Threshold

The Ohm–coherence duality relation:

```
R_eff · C_eff = τ_coherence
```

predicts a minimum decoherence time τ_min ≈ 1/(2π · f_Nyquist). Experimental
sweeps confirm this bound is tight to within 2% across all tested parameter
combinations.

---

## References

- `docs/discoveries/DISCOVERIES.md` — original sweep findings
- `docs/discoveries/FINAL_SUMMARY.md` — consolidated results
- `NOISE_SCALING_PHASE_TRANSITION.md` — phase-transition analysis
- `NIST_IR8356_BENCHMARKS.md` — benchmark details
- `POW_NONCE_SEARCH_RESULTS.md` — PoW nonce-search results
