// gen_pi_digits — Generate a standalone file of all 2,000,000 decimal digits
// of π using the Brent-Salamin AGM algorithm.
//
// Usage:
//
//	go run go/gen_pi_digits/main.go [output_file]
//
// If output_file is omitted, writes to pi_2million_digits.txt in the current
// directory.
//
// The output file contains:
//   - A header with algorithm, precision, and date.
//   - The full decimal expansion formatted as 50 digits per line in 10-digit
//     groups, with a position counter at the left margin.
//   - A SHA-256 fingerprint of the raw digit string at the foot of the file
//     for independent verification.
//
// Algorithm  : Brent-Salamin AGM (quadratic convergence, ~23 iterations)
// Library    : Go math/big.Float (stdlib, no external dependencies)
// Precision  : 2,000,000 decimal digits
package main

import (
	"crypto/sha256"
	"fmt"
	"math/big"
	"os"
	"strings"
	"time"
)

// ── AGM constants (same as go/sanity_checkpoint/main.go) ─────────────────────

const (
	// log2Ten is log₂(10) ≈ 3.32193 — bits per decimal digit.
	log2Ten = 3.32193

	// agmGuardBits is extra precision added beyond the target digit count.
	// 512 guard bits (~154 decimal digits) absorbs rounding over ~25 iterations.
	agmGuardBits = 512

	// agmMaxIter is the hard cap on AGM iterations (quadratic convergence means
	// ~23 iterations suffice for 2,000,000 digits; 100 provides ample margin).
	agmMaxIter = 100

	// agmConvergeBits: treat (a−b) as zero when its exponent is this many bits
	// below the working precision.  64 bits ≈ 19 decimal digits of slack.
	agmConvergeBits = 64
)

// piAGM computes π to the requested number of decimal digits using the
// Brent-Salamin AGM algorithm.
func piAGM(digits int) *big.Float {
	prec := uint(float64(digits)*log2Ten) + agmGuardBits
	one := new(big.Float).SetPrec(prec).SetInt64(1)
	two := new(big.Float).SetPrec(prec).SetInt64(2)
	four := new(big.Float).SetPrec(prec).SetInt64(4)

	a := new(big.Float).SetPrec(prec).Set(one)
	b := new(big.Float).SetPrec(prec).SetInt64(2)
	b.Sqrt(b)
	b.Quo(one, b) // b0 = 1/√2

	t := new(big.Float).SetPrec(prec)
	t.Quo(one, four) // t0 = 1/4

	p := new(big.Float).SetPrec(prec).Set(one)

	tmp1 := new(big.Float).SetPrec(prec)
	tmp2 := new(big.Float).SetPrec(prec)

	for i := 0; i < agmMaxIter; i++ {
		a1 := new(big.Float).SetPrec(prec).Add(a, b)
		a1.Quo(a1, two)

		b1 := new(big.Float).SetPrec(prec).Mul(a, b)
		b1.Sqrt(b1)

		tmp1.Sub(a, a1)
		tmp1.Mul(tmp1, tmp1)
		tmp2.Mul(p, tmp1)
		t.Sub(t, tmp2)

		p.Mul(p, two)
		a.Set(a1)
		b.Set(b1)

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
// `digits` places after the decimal point, using big.Int.Text(10) for
// fast O(n log²n) base conversion.
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

func main() {
	const digits = 2_000_000

	outPath := "pi_2million_digits.txt"
	if len(os.Args) > 1 {
		outPath = os.Args[1]
	}

	fmt.Printf("Computing %d decimal digits of π (Brent-Salamin AGM)...\n", digits)
	start := time.Now()
	pi := piAGM(digits)
	computeTime := time.Since(start)
	fmt.Printf("  AGM compute : %v\n", computeTime)

	fmt.Printf("Converting to decimal string...\n")
	start = time.Now()
	piStr := bigFloatToDecStr(pi, digits)
	convTime := time.Since(start)
	fmt.Printf("  Conversion  : %v\n", convTime)

	// SHA-256 of the raw digit string (including "3." prefix) for verification.
	hash := sha256.Sum256([]byte(piStr))
	hashHex := fmt.Sprintf("%x", hash)

	fmt.Printf("Writing %s ...\n", outPath)
	f, err := os.Create(outPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	// ── Header ───────────────────────────────────────────────────────────────
	fmt.Fprintf(f, "# π to %d decimal digits\n", digits)
	fmt.Fprintf(f, "# Algorithm  : Brent-Salamin AGM (quadratic convergence, ~23 iterations)\n")
	fmt.Fprintf(f, "# Library    : Go math/big.Float (stdlib, no external deps)\n")
	fmt.Fprintf(f, "# Generated  : %s\n", time.Now().UTC().Format("2006-01-02T15:04:05Z"))
	fmt.Fprintf(f, "# Digits     : %d decimal places\n", digits)
	fmt.Fprintf(f, "# SHA-256    : %s\n", hashHex)
	fmt.Fprintf(f, "#\n")
	fmt.Fprintf(f, "# Format: position index [1-based decimal place] followed by 50 digits in\n")
	fmt.Fprintf(f, "# groups of 10.  The integer part (3) is on the first line by itself.\n")
	fmt.Fprintf(f, "#\n")
	fmt.Fprintf(f, "3.\n")

	// ── Digit body ───────────────────────────────────────────────────────────
	dec := piStr[2:] // decimal digits only (after "3.")
	for i := 0; i < len(dec); i += 50 {
		end := i + 50
		if end > len(dec) {
			end = len(dec)
		}
		chunk := dec[i:end]
		var sb strings.Builder
		for j := 0; j < len(chunk); j++ {
			if j > 0 && j%10 == 0 {
				sb.WriteByte(' ')
			}
			sb.WriteByte(chunk[j])
		}
		fmt.Fprintf(f, "[%8d] %s\n", i+1, sb.String())
	}

	// ── Footer ───────────────────────────────────────────────────────────────
	fmt.Fprintf(f, "#\n")
	fmt.Fprintf(f, "# End of file — %d decimal digits of π\n", digits)
	fmt.Fprintf(f, "# SHA-256    : %s\n", hashHex)

	fmt.Printf("  Written     : %s\n", outPath)
	fmt.Printf("  SHA-256     : %s\n", hashHex)
	fmt.Printf("  Total time  : %v\n", computeTime+convTime)
}
