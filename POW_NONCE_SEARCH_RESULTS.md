# Palindrome Precession Theory Overview
- Summary of two palindromes
- Overview of torus T²
- Explanation of zero-overhead periodicity
- [Full documentation](docs/B11_palindrome_precession.md)

## 11. Introduction to B11
B11 implements deterministic angular precession ∆Φ = 2π/13717421 rad/window derived from the palindrome quotient 987654321/123456789 = 8 + 1/13717421 (integer 8 = base orbit period; ε ≈7.29e-8 = precession rate). Creates exact torus closure after 109,739,368 steps (41,152,271 windings) with zero excess resistance (r=1, C=1, T=0). See docs/B11_palindrome_precession.md for full derivation, proofs (Theorems 2.3, 3.2, 5.1, 6.1), continued fraction [0;2,1,1,1,214334,...], and two-palindromes orthogonality.

## 12. B12 Sweep
| k | δω (cycles/step) | Super-period | Nonce     | Time (ms) | Dispersion | Notes                  |
|---|------------------|--------------|-----------|-----------|------------|------------------------|
| 1 | ~7.29e-8        | ~110M       | 51371    | ~78.5    | ~0.22     | Optimal, fastest       |
| 2 | ~3.645e-8       | ~220M       | 136099   | ~210     | ~0.22     | Slower, different path |
| 4 | ~1.822e-8       | ~439M       | 51371    | ~78.5    | ~0.22     | Matches k=1            |
| 8 | ~9.11e-9        | ~878M       | 136099   | ~210     | ~0.22     | Reverts toward baseline|

**Takeaway**: Dispersion modulated across all k (breaks 0.2605 lock); k=1 maximizes perturbation and speedup.

$$ \Delta \Phi = \frac{2\pi}{13717421} $$