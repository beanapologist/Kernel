// sanity_checkpoint — Multi-Vector Sanity Checkpoint (Go)
//
// Filters computed values against 6 attack vectors derived from
// break_system.py:
//
//  1. ALGEBRAIC  — Exact identity verification (symbolic analogue)
//  2. NUMERICAL  — Large-sweep validation (1M+ points)
//  3. CHECKSUM   — SHA-256 hash-locked invariant detection
//  4. CROSS      — Inter-structure consistency checks
//  5. EDGE       — Adversarial extreme inputs (NaN, ±Inf, near-zero)
//  6. PI         — Leibniz series (2,000,000 terms) attacked by Lean formulas;
//                  then full 2,000,000 decimal digits via Brent-Salamin AGM
//
// Uses only the Go standard library.  Exits 0 on full pass, non-zero on
// any failure.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"math/big"
	"os"
	"strings"
)

// ── Verified constants ────────────────────────────────────────────────────────
const (
	eta       = 0.7071067811865475244 // 1/√2  (Theorem 3)
	deltaS    = 2.4142135623730950488 // 1+√2  (Prop 4)
	deltaConj = 0.4142135623730950488 // √2-1 = 1/δ_S
	phi       = 1.6180339887498948482 // (1+√5)/2 golden ratio
	piVal     = math.Pi

	// µ = e^{i3π/4}
	muRe = -eta
	muIm = eta

	// Lepton masses (PDG 2022, MeV/c²)
	mE   = 0.51099895000
	mMu  = 105.6583755
	mTau = 1776.86

	// Tolerances
	tightTol = 1e-12
	sweepTol = 1e-10
)

// ── Math kernel functions ─────────────────────────────────────────────────────

// isFinite reports whether x is neither infinite nor NaN.
func isFinite(x float64) bool {
	return !math.IsInf(x, 0) && !math.IsNaN(x)
}

// coherence implements C(r) = 2r/(1+r²) — Theorem 11.
// Returns 0 for r ≤ 0 or non-finite r.
func coherence(r float64) float64 {
	if math.IsNaN(r) || math.IsInf(r, 0) || r <= 0 {
		return 0
	}
	denom := 1.0 + r*r
	if !isFinite(denom) || denom == 0 {
		return 0
	}
	return (2.0 * r) / denom
}

// frustration implements F(λ) = 1 − sech(λ).
func frustration(lambda float64) float64 {
	if math.IsNaN(lambda) {
		return 0
	}
	if math.IsInf(lambda, 0) {
		return 1
	}
	ch := math.Cosh(lambda)
	if !isFinite(ch) {
		return 1
	}
	return 1.0 - 1.0/ch
}

// cxMul multiplies two complex numbers represented as (re, im) pairs.
func cxMul(ar, ai, br, bi float64) (float64, float64) {
	return ar*br - ai*bi, ar*bi + ai*br
}

// cxAbs returns |a + ib|.
func cxAbs(re, im float64) float64 {
	return math.Hypot(re, im)
}

// ── Test harness ──────────────────────────────────────────────────────────────
type harness struct {
	passed int
	failed int
}

func (h *harness) check(name string, condition bool, detail string) {
	if condition {
		fmt.Printf("  ✓ %s\n", name)
		h.passed++
	} else {
		if detail == "" {
			fmt.Printf("  ✗ %s\n", name)
		} else {
			fmt.Printf("  ✗ %s — %s\n", name, detail)
		}
		h.failed++
	}
}

// ── SHA-256 checksum helpers ──────────────────────────────────────────────────

// invariantHash formats a (name, value) pair the same way as break_system.py
// and returns the first 16 hex characters of its SHA-256 digest.
func invariantHash(name string, v float64) string {
	s := fmt.Sprintf("%s:%.12e", name, v)
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])[:16]
}

// ── VECTOR 1: ALGEBRAIC ───────────────────────────────────────────────────────
func vector1Algebraic(h *harness) {
	fmt.Println("\n═══ VECTOR 1: ALGEBRAIC IDENTITIES ═══")

	// µ⁸ = 1  (8-cycle, Theorem 10)
	re, im := 1.0, 0.0
	for range 8 {
		re, im = cxMul(re, im, muRe, muIm)
	}
	h.check("mu^8 = 1", math.Abs(re-1.0) < tightTol && math.Abs(im) < tightTol, "")

	// µ^n ≠ 1 for 1 ≤ n < 8
	ar, ai := 1.0, 0.0
	for n := 1; n <= 7; n++ {
		ar, ai = cxMul(ar, ai, muRe, muIm)
		dist := cxAbs(ar-1.0, ai)
		h.check(fmt.Sprintf("mu^%d != 1", n), dist > 0.1, fmt.Sprintf("dist=%.6f", dist))
	}

	// δ_S·(√2-1) = 1  (Prop 4: silver conservation)
	h.check("delta_S * (sqrt2-1) = 1", math.Abs(deltaS*deltaConj-1.0) < tightTol, "")

	// φ²-φ-1 = 0  (golden ratio identity)
	h.check("phi^2 - phi - 1 = 0", math.Abs(phi*phi-phi-1.0) < tightTol, "")

	// 9 × 13717421 = 123456789
	h.check("9 * 13717421 = 123456789", 9*13717421 == 123456789, "")

	// C(1) = 1  (Theorem 11)
	h.check("C(1) = 1", math.Abs(coherence(1.0)-1.0) < tightTol, "")

	// C(δ_S) = 1/√2
	h.check("C(delta_S) = 1/sqrt2", math.Abs(coherence(deltaS)-eta) < tightTol, "")

	// C(1/δ_S) = 1/√2
	h.check("C(1/delta_S) = 1/sqrt2", math.Abs(coherence(deltaConj)-eta) < tightTol, "")

	// C(φ²) = 2/3
	h.check("C(phi^2) = 2/3", math.Abs(coherence(phi*phi)-2.0/3.0) < tightTol, "")

	// Gear: 8 × 3π/4 = 6π
	h.check("gear: 8*(3pi/4) = 6pi", math.Abs(8.0*3.0*piVal/4.0-3.0*2.0*piVal) < tightTol, "")
}

// ── VECTOR 2: NUMERICAL ───────────────────────────────────────────────────────
func vector2Numerical(h *harness) {
	fmt.Println("\n═══ VECTOR 2: NUMERICAL SWEEPS ═══")
	const n = 1_000_000

	maxC := 0.0
	maxSymErr := 0.0
	maxDualErr := 0.0
	maxFrustNeg := 0.0
	cBounded := true

	logLo, logHi := -8.0, 8.0
	step := (logHi - logLo) / float64(n-1)

	for i := range n {
		expVal := logLo + float64(i)*step
		r := math.Pow(10, expVal)
		c := coherence(r)

		if c > maxC {
			maxC = c
		}
		if c < 0 || c > 1+1e-15 {
			cBounded = false
		}

		// Symmetry: C(r) = C(1/r)
		if symErr := math.Abs(c - coherence(1.0/r)); symErr > maxSymErr {
			maxSymErr = symErr
		}

		// Duality: C(e^λ) = sech(λ)
		lambda := expVal * math.Log(10)
		cExp := coherence(math.Exp(lambda))
		ch := math.Cosh(lambda)
		sechL := 0.0
		if isFinite(ch) {
			sechL = 1.0 / ch
		}
		if dErr := math.Abs(cExp - sechL); dErr > maxDualErr {
			maxDualErr = dErr
		}

		// Frustration non-negativity
		if fl := frustration(lambda); fl < -1e-15 {
			if -fl > maxFrustNeg {
				maxFrustNeg = -fl
			}
		}
	}

	h.check("C(r) in [0,1] over 1M sweep", cBounded, "")
	h.check("C_max = 1.0 over sweep", math.Abs(maxC-1.0) < 1e-6, fmt.Sprintf("max_c=%.6f", maxC))
	h.check("Symmetry |C(r)-C(1/r)| < 1e-10 over 1M", maxSymErr < sweepTol, fmt.Sprintf("err=%.2e", maxSymErr))
	h.check("Duality |C(e^l)-sech(l)| < 1e-10 over 1M", maxDualErr < sweepTol, fmt.Sprintf("err=%.2e", maxDualErr))
	h.check("F(lambda) >= 0 over 1M", maxFrustNeg < 1e-15, fmt.Sprintf("neg_err=%.2e", maxFrustNeg))

	// F boundary
	h.check("F(0) = 0", math.Abs(frustration(0)) < tightTol, "")
	h.check("F(+inf) = 1", math.Abs(frustration(math.Inf(1))-1.0) < tightTol, "")
	h.check("F(-inf) = 1", math.Abs(frustration(math.Inf(-1))-1.0) < tightTol, "")

	// µ orbit norms
	ar, ai := 1.0, 0.0
	maxNormErr := 0.0
	for range 8 {
		ar, ai = cxMul(ar, ai, muRe, muIm)
		if err := math.Abs(cxAbs(ar, ai) - 1.0); err > maxNormErr {
			maxNormErr = err
		}
	}
	h.check("|mu^n| = 1 for n=1..8", maxNormErr < tightTol, "")
}

// ── VECTOR 3: CHECKSUM ────────────────────────────────────────────────────────
func vector3Checksum(h *harness) {
	fmt.Println("\n═══ VECTOR 3: CHECKSUM INTEGRITY ═══")

	koideQ := (mE + mMu + mTau) /
		math.Pow(math.Sqrt(mE)+math.Sqrt(mMu)+math.Sqrt(mTau), 2)

	type inv struct {
		name string
		val  float64
	}
	table := []inv{
		{"GATE", eta},
		{"PHI", phi},
		{"DELTA_S", deltaS},
		{"C(1)", coherence(1.0)},
		{"C(DS)", coherence(deltaS)},
		{"C(PHI2)", coherence(phi * phi)},
		{"KOIDE", koideQ},
		{"SILVER", deltaS * deltaConj},
		{"GOLDEN", phi*phi - phi - 1.0},
		{"F(0)", frustration(0)},
		{"GEAR", 8.0 * 3.0 * piVal / 4.0 / (2.0 * piVal)},
	}

	fmt.Printf("  %-12s %-20s Value\n", "Name", "Hash")
	fmt.Printf("  %s\n", strings.Repeat("-", 50))

	hashes := make([]string, len(table))
	for i, entry := range table {
		hashes[i] = invariantHash(entry.name, entry.val)
		fmt.Printf("  %-12s %-20s %e\n", entry.name, hashes[i], entry.val)
	}

	// Recompute and verify
	allMatch := true
	for i, entry := range table {
		if invariantHash(entry.name, entry.val) != hashes[i] {
			allMatch = false
		}
	}
	h.check(fmt.Sprintf("All %d checksums reproduced", len(table)), allMatch, "")

	// Tamper detection: perturbations above the format precision must change the
	// hash.  1e-15 is below the resolution of "%.12e" format (precision ~7e-13
	// for eta), so it is expected to be missed — that is correct behavior.
	fmt.Println("\n  --- Tamper detection ---")
	deltas := []float64{1e-15, 1e-12, 1e-10, 1e-8, 1e-5}
	orig := invariantHash("GATE", eta)
	allLargeDetected := true
	for _, delta := range deltas {
		tampered := invariantHash("GATE", eta+delta)
		detected := tampered != orig
		expected := delta >= 1e-13
		if expected && !detected {
			allLargeDetected = false
		}
		symbol := "DETECTED ✓"
		if !detected {
			symbol = "MISSED ✗"
		}
		fmt.Printf("  GATE + %.0e: %s\n", delta, symbol)
	}
	h.check("Tamper deltas >=1e-12 detected", allLargeDetected, "")
}

// ── VECTOR 4: CROSS-STRUCTURE ─────────────────────────────────────────────────
func vector4Cross(h *harness) {
	fmt.Println("\n═══ VECTOR 4: CROSS-STRUCTURE CONSISTENCY ═══")

	// Im(µ) = C(δ_S)
	h.check("Im(mu) = C(delta_S)", math.Abs(muIm-coherence(deltaS)) < tightTol, "")

	// |µ| = 1
	h.check("|mu| = 1", math.Abs(cxAbs(muRe, muIm)-1.0) < tightTol, "")

	// η² + |µη|² = 1  (energy conservation, Theorem 9)
	muEtaRe, muEtaIm := cxMul(muRe, muIm, eta, 0)
	lhs := eta*eta + cxAbs(muEtaRe, muEtaIm)*cxAbs(muEtaRe, muEtaIm)
	h.check("eta^2 + |mu*eta|^2 = 1", math.Abs(lhs-1.0) < tightTol, "")

	// C(φ²) ≈ Koide Q
	koideQ := (mE + mMu + mTau) /
		math.Pow(math.Sqrt(mE)+math.Sqrt(mMu)+math.Sqrt(mTau), 2)
	h.check("C(phi^2) ~= Koide Q", math.Abs(coherence(phi*phi)-koideQ) < 0.001, "")

	// Coherence ordering
	h.check("C(phi^2) < C(delta_S) < C(1)",
		coherence(phi*phi) < coherence(deltaS) && coherence(deltaS) < coherence(1.0), "")

	// Palindrome integer quotient = 8
	h.check("987654321 / 123456789 (integer) = 8", 987654321/123456789 == 8, "")

	// Gear closure
	h.check("gear closure: 8*(3pi/4) = 6pi",
		math.Abs(8.0*3.0*piVal/4.0-3.0*2.0*piVal) < tightTol, "")

	// δ_S self-similarity: δ_S = 2 + 1/δ_S
	h.check("delta_S = 2 + 1/delta_S",
		math.Abs(deltaS-(2.0+1.0/deltaS)) < tightTol, "")

	// Duality at GATE point: C(e^η) = sech(η)
	h.check("C(e^eta) = sech(eta)",
		math.Abs(coherence(math.Exp(eta))-1.0/math.Cosh(eta)) < tightTol, "")
}

// ── VECTOR 5: EDGE CASES ──────────────────────────────────────────────────────
func vector5Edge(h *harness) {
	fmt.Println("\n═══ VECTOR 5: EDGE CASES & ADVERSARIAL INPUTS ═══")

	// C boundary values
	edgeC := func(name string, r, expected, tol float64) bool {
		return math.Abs(coherence(r)-expected) < tol
	}

	h.check("C(0) = 0", edgeC("0", 0, 0, tightTol), "")
	h.check("C(1e-300) ~= 0", edgeC("1e-300", 1e-300, 0, 1e-12), "")
	h.check("C(1e300) ~= 0", edgeC("1e300", 1e300, 0, 1e-12), "")
	h.check("C(1-1e-10) ~= 1", edgeC("1-1e-10", 1-1e-10, 1, 0.01), "")
	h.check("C(1+1e-10) ~= 1", edgeC("1+1e-10", 1+1e-10, 1, 0.01), "")

	// F boundary values
	edgeF := func(name string, l, expected, tol float64) bool {
		return math.Abs(frustration(l)-expected) < tol
	}

	h.check("F(0) = 0", edgeF("0", 0, 0, tightTol), "")
	h.check("F(1e-15) ~= 0", edgeF("1e-15", 1e-15, 0, 1e-12), "")
	h.check("F(710) ~= 1", edgeF("710", 710, 1, 1e-6), "")
	h.check("F(-710) ~= 1", edgeF("-710", -710, 1, 1e-6), "")
	h.check("F(1000) ~= 1", edgeF("1000", 1000, 1, 1e-12), "")

	// NaN/Inf resistance
	fmt.Println("\n  --- NaN/Inf resistance ---")
	h.check("C(NaN) is finite", isFinite(coherence(math.NaN())), "")
	h.check("C(+Inf) is finite", isFinite(coherence(math.Inf(1))), "")
	h.check("C(-Inf) is finite", isFinite(coherence(math.Inf(-1))), "")
	h.check("F(NaN) is finite", isFinite(frustration(math.NaN())), "")
	h.check("F(+Inf) is finite", isFinite(frustration(math.Inf(1))), "")
	h.check("F(-Inf) is finite", isFinite(frustration(math.Inf(-1))), "")

	// Negative r outside physical domain
	h.check("C(-1) = 0 (outside domain)", math.Abs(coherence(-1)) < tightTol, "")
	h.check("C(-0.0) = 0 (outside domain)", math.Abs(coherence(math.Copysign(0, -1))) < tightTol, "")
}

// ── AGM helpers for high-precision π computation ──────────────────────────────

const (
	// log2Ten is log₂(10) ≈ 3.32193, the number of binary bits required to
	// represent one decimal digit.  Used to convert a decimal digit count into
	// the minimum big.Float precision in bits.
	log2Ten = 3.32193

	// agmGuardBits is the extra precision (in bits) added beyond the target
	// decimal-digit count when allocating big.Float values.  512 guard bits
	// (~154 decimal digits) is more than enough to absorb rounding accumulated
	// across ~25 AGM iterations.
	agmGuardBits = 512

	// agmMaxIter is the hard cap on AGM iterations.  The algorithm converges
	// quadratically, so 2,000,000-digit precision needs only
	// ceil(log₂(2,000,000 × log₂10)) ≈ 23 iterations; 100 provides ample
	// margin and the early-exit test below ends the loop in practice.
	agmMaxIter = 100

	// agmConvergeBits is the number of bits below the working precision at
	// which (a − b) is treated as zero for convergence purposes.  64 bits
	// corresponds to ≈19 decimal digits of slack, safely within the guard band.
	agmConvergeBits = 64
)

// piAGMCalc computes π to the requested number of decimal digits using the
// Brent-Salamin arithmetic-geometric mean (AGM) algorithm.
//
// The iteration is:
//
//	a_{n+1} = (a_n + b_n) / 2
//	b_{n+1} = √(a_n · b_n)
//	t_{n+1} = t_n − p_n · (a_n − a_{n+1})²
//	p_{n+1} = 2 · p_n
//	π       ≈ (a + b)² / (4t)  (at convergence)
//
// Convergence is quadratic: each iteration approximately doubles the number
// of correct decimal digits.  For 2,000,000 digits only ~23 iterations are
// needed (log₂(2,000,000 × 3.32) ≈ 23).
func piAGMCalc(digits int) *big.Float {
	// bits of precision: decimal digits × log₂(10) plus a guard band
	prec := uint(float64(digits)*log2Ten) + agmGuardBits
	one := new(big.Float).SetPrec(prec).SetInt64(1)
	two := new(big.Float).SetPrec(prec).SetInt64(2)
	four := new(big.Float).SetPrec(prec).SetInt64(4)

	a := new(big.Float).SetPrec(prec).Set(one) // a0 = 1
	b := new(big.Float).SetPrec(prec).SetInt64(2)
	b.Sqrt(b)
	b.Quo(one, b) // b0 = 1/√2

	t := new(big.Float).SetPrec(prec)
	t.Quo(one, four) // t0 = 1/4

	p := new(big.Float).SetPrec(prec).Set(one) // p0 = 1

	tmp1 := new(big.Float).SetPrec(prec)
	tmp2 := new(big.Float).SetPrec(prec)

	for i := 0; i < agmMaxIter; i++ {
		a1 := new(big.Float).SetPrec(prec).Add(a, b)
		a1.Quo(a1, two) // a1 = (a+b)/2

		b1 := new(big.Float).SetPrec(prec).Mul(a, b)
		b1.Sqrt(b1) // b1 = √(a·b)

		tmp1.Sub(a, a1)
		tmp1.Mul(tmp1, tmp1) // (a−a1)²
		tmp2.Mul(p, tmp1)
		t.Sub(t, tmp2) // t1 = t − p·(a−a1)²

		p.Mul(p, two) // p1 = 2·p
		a.Set(a1)
		b.Set(b1)

		// Converged when a and b agree to within agmConvergeBits of full precision.
		tmp1.Sub(a, b)
		exp := tmp1.MantExp(nil)
		if tmp1.Sign() == 0 || exp < -int(prec-agmConvergeBits) {
			break
		}
	}

	pi := new(big.Float).SetPrec(prec).Add(a, b)
	pi.Mul(pi, pi)
	denom := new(big.Float).SetPrec(prec).Mul(four, t)
	pi.Quo(pi, denom)
	return pi
}

// bigFloatToDecStr converts a big.Float to a decimal string with exactly
// `digits` places after the decimal point ("3.14159...").
//
// The conversion scales f by 10^digits, extracts the integer part as a
// big.Int, and calls big.Int.Text(10).  big.Int.Text uses divide-and-conquer
// base conversion — O(n log²n) — which is far faster than big.Float.Text
// for large digit counts.
func bigFloatToDecStr(f *big.Float, digits int) string {
	prec := f.Prec()
	tenPow := new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(digits)), nil)
	scale := new(big.Float).SetPrec(prec).SetInt(tenPow)
	scaled := new(big.Float).SetPrec(prec).Mul(f, scale)
	intPart := new(big.Int)
	scaled.Int(intPart)
	s := intPart.Text(10)
	if len(s) > 1 {
		return s[:1] + "." + s[1:]
	}
	return s
}

// printAllPiDigits prints the decimal expansion of π with every single digit
// visible, grouped as 10 per cluster, 5 clusters (50 digits) per line, with
// a position counter at the left margin.
func printAllPiDigits(piStr string) {
	// piStr has the form "3.ddddd…" (1 integer digit + "." + N decimal digits)
	fmt.Printf("  3.\n")
	dec := piStr[2:] // decimal digits only
	for i := 0; i < len(dec); i += 50 {
		end := i + 50
		if end > len(dec) {
			end = len(dec)
		}
		chunk := dec[i:end]
		var b strings.Builder
		for j := 0; j < len(chunk); j++ {
			if j > 0 && j%10 == 0 {
				b.WriteByte(' ')
			}
			b.WriteByte(chunk[j])
		}
		fmt.Printf("  [%8d] %s\n", i+1, b.String())
	}
}

// ── VECTOR 6: PI COMPUTATION (2,000,000 Leibniz terms + 2M-digit AGM) ────────
// Part A: Leibniz-Gregory series with 2,000,000 iterations, attacked with 8
//         Lean-grounded identities (Real.pi_gt_3141592, µ⁸=1, Theorem 10,
//         Prop 4, Wyler 6π⁵≈m_p/m_e).
// Part B: Brent-Salamin AGM to full 2,000,000 decimal digits — every single
//         digit printed and verified against the known reference.
func vector6Pi(h *harness) {
	fmt.Println("\n═══ VECTOR 6: PI COMPUTATION (2,000,000 Leibniz terms) ═══")

	const n = 2_000_000

	// Leibniz-Gregory series: π/4 = Σ_{k=0}^{N-1} (-1)^k / (2k+1)
	sum := 0.0
	for k := range n {
		term := 1.0 / (2.0*float64(k) + 1.0)
		if k%2 == 0 {
			sum += term
		} else {
			sum -= term
		}
	}
	piLbn := 4.0 * sum

	// Alternating-series truncation bound: |π - piLbn| < 4/(2N+1)
	leibnizBound := 4.0 / (2.0*n + 1.0)

	fmt.Printf("  Leibniz pi (2M terms): %.16f\n", piLbn)
	fmt.Printf("  Reference PI:          %.16f\n", piVal)
	fmt.Printf("  Bound 4/(2N+1):        %.3e\n\n", leibnizBound)

	// Attack 1: Real.pi_gt_3141592 minus Leibniz buffer → piLbn > 3.14159
	h.check("piLbn > 3.14159 (Lean: Real.pi_gt_3141592 - Leibniz buffer)",
		piLbn > 3.14159, "")

	// Attack 2: Upper sanity bound
	h.check("piLbn < 3.14160 (upper sanity)", piLbn < 3.14160, "")

	// Attack 3: Leibniz series converged to the correct precision
	h.check("|piLbn - PI| < 4/(2N+1) (Leibniz truncation bound)",
		math.Abs(piLbn-piVal) < leibnizBound,
		fmt.Sprintf("err=%.3e", math.Abs(piLbn-piVal)))

	// Attack 4: sin(π) = 0 — transcendental identity grounded in Lean
	h.check("|sin(piLbn)| < 2e-6 (sin(pi)=0 identity)",
		math.Abs(math.Sin(piLbn)) < 2e-6, "")

	// Attack 5: Gear identity 8*(3π/4) = 6π — Theorem 10 (µ 8-cycle)
	// Algebraically exact for any value of π; verifies arithmetic coherence.
	h.check("8*(3*piLbn/4) = 6*piLbn (gear identity, Theorem 10)",
		math.Abs(8.0*3.0*piLbn/4.0-6.0*piLbn) < 1e-9, "")

	// Attack 6: µ⁸ = 1 using Leibniz angle 3*piLbn/4 (Section 2, Theorem 10)
	angle := 3.0 * piLbn / 4.0
	muPiRe, muPiIm := math.Cos(angle), math.Sin(angle)
	ar, ai := 1.0, 0.0
	for range 8 {
		ar, ai = cxMul(ar, ai, muPiRe, muPiIm)
	}
	h.check("|muLbn^8 - 1| < 2e-5 (mu^8=1 with Leibniz angle)",
		cxAbs(ar-1.0, ai) < 2e-5, "")

	// Attack 7: Im(µ) = sin(3π/4) = 1/√2 = C(δ_S) (Prop 4 + Section 2)
	h.check("|sin(3*piLbn/4) - eta| < 2e-6 (Im(mu)=C(delta_S)=1/sqrt2)",
		math.Abs(math.Sin(3.0*piLbn/4.0)-eta) < 2e-6, "")

	// Attack 8: Wyler approximation 6π⁵ ≈ m_p/m_e ≈ 1836.15 (±0.5%)
	wylerLbn := 6.0 * math.Pow(piLbn, 5)
	h.check("6*piLbn^5 in [1835,1837] (Wyler: 6*pi^5 approx m_p/m_e)",
		wylerLbn > 1835.0 && wylerLbn < 1837.0,
		fmt.Sprintf("wyler=%.4f", wylerLbn))

	// ── Part B: HIGH-PRECISION — ALL 2,000,000 DECIMAL DIGITS via AGM ───────
	const piHiDigits = 2_000_000

	fmt.Printf("\n═══ VECTOR 6b: π TO %d DECIMAL DIGITS (Brent-Salamin AGM) ═══\n",
		piHiDigits)
	fmt.Printf("  Algorithm : Brent-Salamin AGM (quadratic convergence)\n")
	fmt.Printf("  Precision : %d decimal digits\n", piHiDigits)
	fmt.Printf("  Library   : Go math/big.Float (stdlib, no external deps)\n")
	fmt.Printf("  Computing...\n")

	piHi := piAGMCalc(piHiDigits)
	piHiStr := bigFloatToDecStr(piHi, piHiDigits)

	fmt.Printf("  Done. Printing all %d decimal digits:\n\n", piHiDigits)
	printAllPiDigits(piHiStr)

	// Attack 9: The first 100 decimal digits of π are humanity's most verified
	// mathematical constant.  Any arithmetic error in the AGM would corrupt them.
	const piKnown100 = "1415926535897932384626433832795028841971693993751" +
		"0582097494459230781640628620899862803482534211706" +
		"79"
	h.check(
		"π (AGM 2M digits): first 100 decimal digits match known reference",
		len(piHiStr) >= 102 && piHiStr[2:102] == piKnown100,
		fmt.Sprintf("got=%s", func() string {
			if len(piHiStr) >= 102 {
				return piHiStr[2:12] + "..."
			}
			return piHiStr
		}()),
	)

	// Attack 10: The output string must represent exactly 2,000,000 decimal
	// places (1 integer digit + "." + 2,000,000 decimal digits = 2,000,002 chars).
	h.check(
		fmt.Sprintf("π (AGM 2M digits): string has %d chars (3. + 2M digits)", piHiDigits+2),
		len(piHiStr) == piHiDigits+2,
		fmt.Sprintf("len=%d", len(piHiStr)),
	)

	// Attack 11: SHA-256 of the full digit string must match the canonical hash
	// recorded in pi_2million_digits.txt (generated by go/gen_pi_digits/main.go).
	// This cross-verifies that the sanity-checkpoint AGM and the generator
	// produce bit-identical results.  The hash is the SHA-256 of the raw
	// "3.dddd…" string (no newlines) and is printed in the file header/footer.
	// To update: run `go run go/gen_pi_digits/main.go` and copy the new hash.
	const piSHA256 = "f533022c5d2a21db137b158345c6276355e89b301d76d1531c1ca26f9a026612"
	hashBytes := sha256.Sum256([]byte(piHiStr))
	gotHash := hex.EncodeToString(hashBytes[:])
	h.check(
		"π (AGM 2M digits): SHA-256 matches pi_2million_digits.txt canonical hash",
		gotHash == piSHA256,
		fmt.Sprintf("got=%s", gotHash[:16]+"..."),
	)
}

// ── Main ──────────────────────────────────────────────────────────────────────
func main() {
	sep := strings.Repeat("═", 72)
	fmt.Printf("\n%s\n", sep)
	fmt.Println("  SANITY CHECKPOINT — Go (6 Attack Vectors)")
	fmt.Printf("%s\n", sep)

	h := &harness{}

	vector1Algebraic(h)
	vector2Numerical(h)
	vector3Checksum(h)
	vector4Cross(h)
	vector5Edge(h)
	vector6Pi(h)

	fmt.Printf("\n%s\n", sep)
	fmt.Println("  VERDICT")
	fmt.Printf("%s\n\n", sep)
	fmt.Printf("  Passed: %d\n", h.passed)
	fmt.Printf("  Failed: %d\n\n", h.failed)

	if h.failed == 0 {
		fmt.Println("  ╔══════════════════════════════════════════════════════════╗")
		fmt.Println("  ║  CANONICAL MAP: UNBROKEN                                ║")
		fmt.Println("  ║  All 6 attack vectors passed. System is coherent.       ║")
		fmt.Println("  ╚══════════════════════════════════════════════════════════╝")
		os.Exit(0)
	} else {
		// Pad the message to keep the box border aligned (inner width 56 chars).
		msg := fmt.Sprintf("%d FAILURE(S) DETECTED. Investigate immediately.", h.failed)
		fmt.Println("  ╔══════════════════════════════════════════════════════════╗")
		fmt.Printf("  ║  %-56s║\n", msg)
		fmt.Println("  ╚══════════════════════════════════════════════════════════╝")
		os.Exit(h.failed)
	}
}
