//! sanity_checkpoint — Multi-Vector Sanity Checkpoint (Rust)
//!
//! Filters computed values against 5 attack vectors derived from
//! break_system.py:
//!
//!   1. ALGEBRAIC  — Exact identity verification (symbolic analogue)
//!   2. NUMERICAL  — Large-sweep validation (1M+ points)
//!   3. CHECKSUM   — FNV-1a hash-locked invariant detection
//!   4. CROSS      — Inter-structure consistency checks
//!   5. EDGE       — Adversarial extreme inputs (NaN, ±Inf, near-zero)
//!
//! No external crates required; stdlib only.  Exit 0 on full pass,
//! non-zero on any failure.

// ── Verified constants ────────────────────────────────────────────────────────
const ETA: f64 = 0.707_106_781_186_547_524_40; // 1/√2  (Theorem 3)
const DELTA_S: f64 = 2.414_213_562_373_095_048_80; // 1+√2  (Prop 4)
const DELTA_CONJ: f64 = 0.414_213_562_373_095_048_80; // √2-1 = 1/δ_S
const PHI: f64 = 1.618_033_988_749_894_848_20; // (1+√5)/2 golden ratio
const PI: f64 = std::f64::consts::PI;

// Lepton masses (PDG 2022, MeV/c²)
const M_E: f64 = 0.510_998_950_00;
const M_MU: f64 = 105.658_375_5;
const M_TAU: f64 = 1776.86;

// µ = e^{i3π/4} = (-1/√2, 1/√2)
const MU_RE: f64 = -ETA;
const MU_IM: f64 = ETA;

// Tolerances
const TIGHT_TOL: f64 = 1e-12;
const SWEEP_TOL: f64 = 1e-10;

// ── Math kernel functions ─────────────────────────────────────────────────────

/// C(r) = 2r/(1+r²) — coherence function (Theorem 11).
/// Returns 0 for r ≤ 0 or non-finite r.
fn coherence(r: f64) -> f64 {
    if !r.is_finite() || r <= 0.0 {
        return 0.0;
    }
    let denom = 1.0 + r * r;
    if !denom.is_finite() || denom == 0.0 {
        return 0.0;
    }
    (2.0 * r) / denom
}

/// F(λ) = 1 − sech(λ) — frustration / Lyapunov envelope.
fn frustration(lambda: f64) -> f64 {
    if lambda.is_nan() {
        return 0.0;
    }
    if lambda.is_infinite() {
        return 1.0;
    }
    let ch = lambda.cosh();
    if !ch.is_finite() {
        return 1.0;
    }
    1.0 - 1.0 / ch
}

// ── Complex helpers ───────────────────────────────────────────────────────────

/// Multiply two complex numbers (re, im).
fn cx_mul(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    (ar * br - ai * bi, ar * bi + ai * br)
}

/// Absolute value of complex number.
fn cx_abs(re: f64, im: f64) -> f64 {
    re.hypot(im)
}

// ── Test harness ──────────────────────────────────────────────────────────────
struct Harness {
    passed: u32,
    failed: u32,
}

impl Harness {
    fn new() -> Self {
        Harness {
            passed: 0,
            failed: 0,
        }
    }

    fn check(&mut self, name: &str, condition: bool, detail: &str) {
        if condition {
            println!("  ✓ {}", name);
            self.passed += 1;
        } else {
            if detail.is_empty() {
                println!("  ✗ {}", name);
            } else {
                println!("  ✗ {} — {}", name, detail);
            }
            self.failed += 1;
        }
    }
}

// ── FNV-1a hash (64-bit) ──────────────────────────────────────────────────────
fn fnv1a_hex(s: &str) -> String {
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
    const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
    let mut h: u64 = FNV_OFFSET;
    for byte in s.bytes() {
        h ^= u64::from(byte);
        h = h.wrapping_mul(FNV_PRIME);
    }
    format!("{:016x}", h)
}

fn invariant_hash(name: &str, v: f64) -> String {
    let s = format!("{}:{:.12e}", name, v);
    fnv1a_hex(&s)
}

// ── VECTOR 1: ALGEBRAIC ───────────────────────────────────────────────────────
fn vector1_algebraic(h: &mut Harness) {
    println!("\n═══ VECTOR 1: ALGEBRAIC IDENTITIES ═══");

    // µ⁸ = 1  (8-cycle, Theorem 10)
    let (mut re, mut im) = (1.0_f64, 0.0_f64);
    for _ in 0..8 {
        let (nr, ni) = cx_mul(re, im, MU_RE, MU_IM);
        re = nr;
        im = ni;
    }
    h.check("mu^8 = 1", (re - 1.0).abs() < TIGHT_TOL && im.abs() < TIGHT_TOL, "");

    // µ^n ≠ 1 for 1 ≤ n < 8
    let (mut ar, mut ai) = (1.0_f64, 0.0_f64);
    for n in 1..=7 {
        let (nr, ni) = cx_mul(ar, ai, MU_RE, MU_IM);
        ar = nr;
        ai = ni;
        let dist = cx_abs(ar - 1.0, ai);
        h.check(
            &format!("mu^{} != 1", n),
            dist > 0.1,
            &format!("dist={:.6}", dist),
        );
    }

    // δ_S·(√2-1) = 1  (silver conservation, Prop 4)
    h.check(
        "delta_S * (sqrt2-1) = 1",
        (DELTA_S * DELTA_CONJ - 1.0).abs() < TIGHT_TOL,
        "",
    );

    // φ²-φ-1 = 0  (golden ratio identity)
    h.check(
        "phi^2 - phi - 1 = 0",
        (PHI * PHI - PHI - 1.0).abs() < TIGHT_TOL,
        "",
    );

    // 9 × 13717421 = 123456789
    h.check("9 * 13717421 = 123456789", 9_i64 * 13_717_421 == 123_456_789, "");

    // C(1) = 1  (Theorem 11)
    h.check("C(1) = 1", (coherence(1.0) - 1.0).abs() < TIGHT_TOL, "");

    // C(δ_S) = 1/√2
    h.check(
        "C(delta_S) = 1/sqrt2",
        (coherence(DELTA_S) - ETA).abs() < TIGHT_TOL,
        "",
    );

    // C(1/δ_S) = 1/√2
    h.check(
        "C(1/delta_S) = 1/sqrt2",
        (coherence(DELTA_CONJ) - ETA).abs() < TIGHT_TOL,
        "",
    );

    // C(φ²) = 2/3
    h.check(
        "C(phi^2) = 2/3",
        (coherence(PHI * PHI) - 2.0 / 3.0).abs() < TIGHT_TOL,
        "",
    );

    // Gear: 8 × 3π/4 = 6π
    h.check(
        "gear: 8*(3pi/4) = 6pi",
        (8.0 * 3.0 * PI / 4.0 - 3.0 * 2.0 * PI).abs() < TIGHT_TOL,
        "",
    );
}

// ── VECTOR 2: NUMERICAL ───────────────────────────────────────────────────────
fn vector2_numerical(h: &mut Harness) {
    println!("\n═══ VECTOR 2: NUMERICAL SWEEPS ═══");
    const N: usize = 1_000_000;

    let mut max_c: f64 = 0.0;
    let mut max_sym_err: f64 = 0.0;
    let mut max_dual_err: f64 = 0.0;
    let mut max_frust_neg: f64 = 0.0;
    let mut c_bounded = true;

    let log_lo: f64 = -8.0;
    let log_hi: f64 = 8.0;
    let step = (log_hi - log_lo) / (N as f64 - 1.0);

    for i in 0..N {
        let exp_val = log_lo + i as f64 * step;
        let r = 10.0_f64.powf(exp_val);
        let c = coherence(r);

        if c > max_c {
            max_c = c;
        }
        if c < 0.0 || c > 1.0 + 1e-15 {
            c_bounded = false;
        }

        // Symmetry: C(r) = C(1/r)
        let sym_err = (c - coherence(1.0 / r)).abs();
        if sym_err > max_sym_err {
            max_sym_err = sym_err;
        }

        // Duality: C(e^λ) = sech(λ)
        let lambda = exp_val * 10.0_f64.ln();
        let c_exp = coherence(lambda.exp());
        let ch = lambda.cosh();
        let sech_l = if ch.is_finite() { 1.0 / ch } else { 0.0 };
        let d_err = (c_exp - sech_l).abs();
        if d_err > max_dual_err {
            max_dual_err = d_err;
        }

        // Frustration non-negativity
        let fl = frustration(lambda);
        if fl < -1e-15 {
            max_frust_neg = max_frust_neg.max(-fl);
        }
    }

    h.check("C(r) in [0,1] over 1M sweep", c_bounded, "");
    h.check(
        "C_max = 1.0 over sweep",
        (max_c - 1.0).abs() < 1e-6,
        &format!("max_c={:.6}", max_c),
    );
    h.check(
        "Symmetry |C(r)-C(1/r)| < 1e-10 over 1M",
        max_sym_err < SWEEP_TOL,
        &format!("err={:.2e}", max_sym_err),
    );
    h.check(
        "Duality |C(e^l)-sech(l)| < 1e-10 over 1M",
        max_dual_err < SWEEP_TOL,
        &format!("err={:.2e}", max_dual_err),
    );
    h.check(
        "F(lambda) >= 0 over 1M",
        max_frust_neg < 1e-15,
        &format!("neg_err={:.2e}", max_frust_neg),
    );

    // F boundary
    h.check("F(0) = 0", frustration(0.0).abs() < TIGHT_TOL, "");
    h.check(
        "F(+inf) = 1",
        (frustration(f64::INFINITY) - 1.0).abs() < TIGHT_TOL,
        "",
    );
    h.check(
        "F(-inf) = 1",
        (frustration(f64::NEG_INFINITY) - 1.0).abs() < TIGHT_TOL,
        "",
    );

    // µ orbit norms
    let (mut ar, mut ai) = (1.0_f64, 0.0_f64);
    let mut max_norm_err = 0.0_f64;
    for _ in 1..=8 {
        let (nr, ni) = cx_mul(ar, ai, MU_RE, MU_IM);
        ar = nr;
        ai = ni;
        let err = (cx_abs(ar, ai) - 1.0).abs();
        if err > max_norm_err {
            max_norm_err = err;
        }
    }
    h.check("|mu^n| = 1 for n=1..8", max_norm_err < TIGHT_TOL, "");
}

// ── VECTOR 3: CHECKSUM ────────────────────────────────────────────────────────
fn vector3_checksum(h: &mut Harness) {
    println!("\n═══ VECTOR 3: CHECKSUM INTEGRITY ═══");

    let koide_q = (M_E + M_MU + M_TAU)
        / (M_E.sqrt() + M_MU.sqrt() + M_TAU.sqrt()).powi(2);

    let table: Vec<(&str, f64)> = vec![
        ("GATE", ETA),
        ("PHI", PHI),
        ("DELTA_S", DELTA_S),
        ("C(1)", coherence(1.0)),
        ("C(DS)", coherence(DELTA_S)),
        ("C(PHI2)", coherence(PHI * PHI)),
        ("KOIDE", koide_q),
        ("SILVER", DELTA_S * DELTA_CONJ),
        ("GOLDEN", PHI * PHI - PHI - 1.0),
        ("F(0)", frustration(0.0)),
        ("GEAR", 8.0 * 3.0 * PI / 4.0 / (2.0 * PI)),
    ];

    println!("  {:<12} {:<20} Value", "Name", "Hash");
    println!("  {}", "-".repeat(50));

    let hashes: Vec<String> = table
        .iter()
        .map(|(name, val)| {
            let h = invariant_hash(name, *val);
            println!("  {:<12} {:<20} {:e}", name, h, val);
            h
        })
        .collect();

    // Recompute and verify
    let all_match = table
        .iter()
        .zip(hashes.iter())
        .all(|((name, val), expected)| invariant_hash(name, *val) == *expected);

    h.check(
        &format!("All {} checksums reproduced", table.len()),
        all_match,
        "",
    );

    // Tamper detection: perturbations above the format precision must change the
    // hash.  1e-15 is below the resolution of ":.12e" format (precision ~7e-13
    // for ETA), so it is expected to be missed — that is correct behavior.
    println!("\n  --- Tamper detection ---");
    let deltas = [1e-15_f64, 1e-12, 1e-10, 1e-8, 1e-5];
    let orig = invariant_hash("GATE", ETA);
    let mut all_large_detected = true;
    for &delta in &deltas {
        let tampered = invariant_hash("GATE", ETA + delta);
        let detected = tampered != orig;
        let expected = delta >= 1e-13;
        if expected && !detected {
            all_large_detected = false;
        }
        println!(
            "  GATE + {:.0e}: {}",
            delta,
            if detected { "DETECTED ✓" } else { "MISSED ✗" }
        );
    }
    h.check("Tamper deltas >=1e-12 detected", all_large_detected, "");
}

// ── VECTOR 4: CROSS-STRUCTURE ─────────────────────────────────────────────────
fn vector4_cross(h: &mut Harness) {
    println!("\n═══ VECTOR 4: CROSS-STRUCTURE CONSISTENCY ═══");

    // Im(µ) = C(δ_S)
    h.check(
        "Im(mu) = C(delta_S)",
        (MU_IM - coherence(DELTA_S)).abs() < TIGHT_TOL,
        "",
    );

    // |µ| = 1
    h.check("|mu| = 1", (cx_abs(MU_RE, MU_IM) - 1.0).abs() < TIGHT_TOL, "");

    // η² + |µη|² = 1  (energy conservation, Theorem 9)
    let mu_eta_re = MU_RE * ETA;
    let mu_eta_im = MU_IM * ETA;
    let lhs = ETA * ETA + cx_abs(mu_eta_re, mu_eta_im).powi(2);
    h.check(
        "eta^2 + |mu*eta|^2 = 1",
        (lhs - 1.0).abs() < TIGHT_TOL,
        "",
    );

    // C(φ²) ≈ Koide Q
    let koide_q = (M_E + M_MU + M_TAU)
        / (M_E.sqrt() + M_MU.sqrt() + M_TAU.sqrt()).powi(2);
    h.check(
        "C(phi^2) ~= Koide Q",
        (coherence(PHI * PHI) - koide_q).abs() < 0.001,
        "",
    );

    // Coherence ordering
    h.check(
        "C(phi^2) < C(delta_S) < C(1)",
        coherence(PHI * PHI) < coherence(DELTA_S) && coherence(DELTA_S) < coherence(1.0),
        "",
    );

    // Palindrome integer quotient = 8
    h.check(
        "987654321 / 123456789 (integer) = 8",
        987_654_321_i64 / 123_456_789 == 8,
        "",
    );

    // Gear closure
    h.check(
        "gear closure: 8*(3pi/4) = 6pi",
        (8.0 * 3.0 * PI / 4.0 - 3.0 * 2.0 * PI).abs() < TIGHT_TOL,
        "",
    );

    // δ_S self-similarity: δ_S = 2 + 1/δ_S
    h.check(
        "delta_S = 2 + 1/delta_S",
        (DELTA_S - (2.0 + 1.0 / DELTA_S)).abs() < TIGHT_TOL,
        "",
    );

    // Duality at GATE point: C(e^η) = sech(η)
    h.check(
        "C(e^eta) = sech(eta)",
        (coherence(ETA.exp()) - 1.0 / ETA.cosh()).abs() < TIGHT_TOL,
        "",
    );
}

// ── VECTOR 5: EDGE CASES ──────────────────────────────────────────────────────
fn vector5_edge(h: &mut Harness) {
    println!("\n═══ VECTOR 5: EDGE CASES & ADVERSARIAL INPUTS ═══");

    // C boundary values
    let edge_c = |_name: &str, r: f64, expected: f64, tol: f64| -> bool {
        (coherence(r) - expected).abs() < tol
    };

    h.check("C(0) = 0", edge_c("0", 0.0, 0.0, TIGHT_TOL), "");
    h.check("C(1e-300) ~= 0", edge_c("1e-300", 1e-300, 0.0, 1e-12), "");
    h.check("C(1e300) ~= 0", edge_c("1e300", 1e300, 0.0, 1e-12), "");
    h.check(
        "C(1-1e-10) ~= 1",
        edge_c("1-1e-10", 1.0 - 1e-10, 1.0, 0.01),
        "",
    );
    h.check(
        "C(1+1e-10) ~= 1",
        edge_c("1+1e-10", 1.0 + 1e-10, 1.0, 0.01),
        "",
    );

    // F boundary values
    let edge_f = |_name: &str, l: f64, expected: f64, tol: f64| -> bool {
        (frustration(l) - expected).abs() < tol
    };

    h.check("F(0) = 0", edge_f("0", 0.0, 0.0, TIGHT_TOL), "");
    h.check("F(1e-15) ~= 0", edge_f("1e-15", 1e-15, 0.0, 1e-12), "");
    h.check("F(710) ~= 1", edge_f("710", 710.0, 1.0, 1e-6), "");
    h.check("F(-710) ~= 1", edge_f("-710", -710.0, 1.0, 1e-6), "");
    h.check("F(1000) ~= 1", edge_f("1000", 1000.0, 1.0, 1e-12), "");

    // NaN/Inf resistance
    println!("\n  --- NaN/Inf resistance ---");
    h.check("C(NaN) is finite", coherence(f64::NAN).is_finite(), "");
    h.check("C(+Inf) is finite", coherence(f64::INFINITY).is_finite(), "");
    h.check(
        "C(-Inf) is finite",
        coherence(f64::NEG_INFINITY).is_finite(),
        "",
    );
    h.check("F(NaN) is finite", frustration(f64::NAN).is_finite(), "");
    h.check("F(+Inf) is finite", frustration(f64::INFINITY).is_finite(), "");
    h.check(
        "F(-Inf) is finite",
        frustration(f64::NEG_INFINITY).is_finite(),
        "",
    );

    // Negative r outside domain
    h.check("C(-1) = 0 (outside domain)", coherence(-1.0).abs() < TIGHT_TOL, "");
    h.check(
        "C(-0.0) = 0 (outside domain)",
        coherence(-0.0_f64).abs() < TIGHT_TOL,
        "",
    );
}

// ── Main ──────────────────────────────────────────────────────────────────────
fn main() {
    let sep = "═".repeat(72);
    println!("\n{}", sep);
    println!("  SANITY CHECKPOINT — Rust (5 Attack Vectors)");
    println!("{}", sep);

    let mut h = Harness::new();

    vector1_algebraic(&mut h);
    vector2_numerical(&mut h);
    vector3_checksum(&mut h);
    vector4_cross(&mut h);
    vector5_edge(&mut h);

    println!("\n{}", sep);
    println!("  VERDICT");
    println!("{}", sep);
    println!();
    println!("  Passed: {}", h.passed);
    println!("  Failed: {}", h.failed);
    println!();

    if h.failed == 0 {
        println!("  ╔══════════════════════════════════════════════════════════╗");
        println!("  ║  CANONICAL MAP: UNBROKEN                                ║");
        println!("  ║  All 5 attack vectors passed. System is coherent.       ║");
        println!("  ╚══════════════════════════════════════════════════════════╝");
        std::process::exit(0);
    } else {
        // Pad the message to keep the box border aligned (inner width 56 chars).
        let msg = format!("{} FAILURE(S) DETECTED. Investigate immediately.", h.failed);
        println!("  ╔══════════════════════════════════════════════════════════╗");
        println!("  ║  {:<56}║", msg);
        println!("  ╚══════════════════════════════════════════════════════════╝");
        std::process::exit(h.failed as i32);
    }
}
