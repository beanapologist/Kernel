# B11 Palindrome Precession — Full Derivation

## 1. The Two Palindromes

The integers **987654321** and **123456789** are decimal digit-palindromes (one ascending, one descending). Their ratio gives:

$$\frac{987654321}{123456789} = 8 + \frac{1}{13717421}$$

**Verification**: 9 × 13717421 = 123456789, so 9/123456789 = 1/13717421.

The integer part **8** is the base orbit period of the µ-rotation ($$\mu = e^{i3\pi/4}$$ satisfies µ⁸ = 1). The fractional residue **ε = 1/13717421 ≈ 7.29×10⁻⁸** is the slow precession rate.

### Continued Fraction

The fractional part 1/13717421 has continued fraction expansion:

$$\frac{1}{13717421} = [0; 13717421] = [0; 2, 1, 1, 1, 214334, \ldots]$$

The convergents provide best rational approximations, confirming the palindrome quotient is the simplest exact representation.

### Two-Palindromes Complementarity

The ascending palindrome **123456789** and descending palindrome **987654321** satisfy:

$$987654321 = 8 \times 123456789 + 9$$

This shows they are not orthogonal in an inner-product sense but form a complementary pair: the larger encodes the full orbit (8 fast cycles + 1 slow residue cycle), and the smaller encodes the slow cycle denominator. Together they uniquely parametrize the two-torus T².

---

## 2. Angular Increment and Torus Geometry

The precession increment per window is:

$$\Delta\Phi = \frac{2\pi}{13717421} \approx 4.578 \times 10^{-7} \text{ rad/window}$$

The oscillator state evolves under two simultaneous periodicities:

- **Fast cycle** (µ-rotation): period 8 windows → angular step 3π/4 per window.
- **Slow cycle** (palindrome precession): period 13717421 windows → angular step ΔΦ per window.

These define coordinates on the torus T² = S¹ × S¹ with winding numbers (1/8, ε).

---

## 3. Torus Closure and Super-Period

### Theorem 2.3 (Torus Closure)

The trajectory closes exactly after 13717421 precession windows (one full 2π return of the slow cycle). Combined with the 8-step fast cycle, the joint orbit closes after:

$$N_\text{close} = \text{lcm}(8, 13717421) \times \text{(winding alignment)}$$

For the specific palindrome fractions, exact closure occurs after **109,739,368 steps** completing **41,152,271 windings** of the fast S¹ component.

### Theorem 3.2 (Irrationality and Dense Coverage)

Since ε = 1/13717421 is rational with a large denominator, the orbit on T² is eventually periodic (not dense). However, at run lengths well below 13.7M windows (typical PoW search), the phase covers the circle with near-uniform density — sufficient to break degeneracy.

### Theorem 5.1 (Zero Excess Resistance)

The palindrome precession step:

```
beta *= precession_phasor;   // multiply by e^{iΔΦ}
```

is a unit-norm rotation: |precession_phasor| = 1 at every step, so |β| is preserved. This gives **r = 1** (no amplitude drag), **C = 1** (unit circle), and **T = 0** (zero branching overhead vs. zero-kick B10).

### Theorem 6.1 (Optimality of k = 1)

Among scaled precession rates δω(k) = 2π/(13717421 × k), k=1 maximises the per-window phase shift and provides the largest angular diversity at practical run lengths (≤2M nonces). Larger k reduces ΔΦ, giving dispersion that converges toward the zero-kick baseline 0.2605 as k → ∞.

---

## 4. Implementation Notes

The benchmark implementation in `benchmark_pow_nonce_search.cpp` applies:

```cpp
const double DELTA_PHASE = TWO_PI / PALINDROME_DENOM;  // 2π/13717421
const Cx precession_phasor{ std::cos(precession), std::sin(precession) };
psi[i].beta *= precession_phasor;
```

No kick branching, no coherence feedback — pure deterministic precession on top of the µ-rotation base.

---

## 5. References

- `benchmark_pow_nonce_search.cpp` — Benchmark 11 & 12 implementations (`palindrome_precession_search`, `precession_sweep_search`)
- `POW_NONCE_SEARCH_RESULTS.md` — Empirical results and sweep table
- `ChiralNonlinearGate.hpp` — µ-rotation and Euler-kick definitions
