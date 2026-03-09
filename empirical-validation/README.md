# Empirical Validation

This folder contains the end-to-end empirical validation framework for the Kernel
mathematical constructs.  It ingests publicly available datasets from trusted scientific
repositories (CODATA 2018, NIST, Planck 2018, PDG 2022) and validates them against the
theoretical framework established in the formal Lean proofs using SymPy and NumPy.

---

## Structure

```
empirical-validation/
├── requirements.txt              # Python dependencies
├── run_validation.py             # Main entry point — runs the full pipeline
├── checksums.py                  # SHA-256 + absolute-error checksum utilities
├── data_ingestion/
│   ├── __init__.py
│   ├── codata.py                 # CODATA 2018 physical constants (via SciPy)
│   ├── nist.py                   # NIST mathematical & SI constants
│   └── cosmological.py           # Planck 2018 + IAU 2012 + PDG 2022 constants
├── validators/
│   ├── __init__.py
│   ├── eigenvalue.py             # |μ|² = 1, μ⁸ = 1, rotation matrix orbit
│   ├── fine_structure.py         # Fine-structure constant α (CODATA 2018)
│   ├── particle_mass.py          # Proton/electron mass ratio, Koide formula
│   ├── coherence.py              # Coherence function C(r) = exp(−r²/2)
│   ├── golden_ratio.py           # Golden ratio φ and silver ratio δ_S
│   └── spacetime.py              # Planck units, Hubble radius, Λ
├── tests/
│   └── test_validation.py        # 63 pytest tests covering all modules
├── models/
│   └── balance_of_payments.py    # Eigenvalue-driven BoP model
└── reports/                      # Generated output (created at runtime)
    ├── validation_report.md
    ├── relative_errors.png
    └── pass_fail_summary.png
```

---

## Quick Start

```bash
pip install -r empirical-validation/requirements.txt
python empirical-validation/run_validation.py
```

Run tests:

```bash
pytest empirical-validation/tests/ -v
```

---

## Data Sources

| Source | Description | Reference |
|--------|-------------|-----------|
| **CODATA 2018** | Physical constants (α, e, ℏ, c, m_e, m_p, …) | `scipy.constants` (CODATA 2018) |
| **NIST DLMF** | Mathematical constants (φ, δ_S, π, e, √2) | Computed to IEEE 754 double precision |
| **Planck 2018** | Cosmological parameters (H₀, Ω_Λ, T_CMB, …) | Planck Collaboration A&A 641, A6 (2020) |
| **PDG 2022** | Lepton masses for Koide formula | Particle Data Group, PTEP 2022 |
| **IAU 2012** | Astronomical constants (M_☉, R_☉, pc) | IAU 2012 System of Astronomical Constants |

---

## Validated Constructs

| Validator | Checks | Description |
|-----------|--------|-------------|
| `eigenvalue.py` | 11 | Critical eigenvalue μ = e^{i3π/4}: \|μ\|²=1, μ⁸=1, component values, rotation matrix orbit |
| `fine_structure.py` | 6 | Fine-structure constant α ≈ 7.297×10⁻³: definition, inverse 1/α≈137.036, sub-unity |
| `particle_mass.py` | 6 | m_p/m_e ≈ 1836.15 (CODATA), Koide formula Q≈2/3, Wyler approximation 6π⁵ |
| `coherence.py` | 8 | C(r)=exp(−r²/2): C(0)=1, monotone, integral=√(π/2), golden/silver ratio values |
| `golden_ratio.py` | 11 | φ²=φ+1, φ−1=1/φ, δ_S·(√2−1)=1, Fibonacci convergence, NIST values |
| `spacetime.py` | 8 | Planck units t_P, l_P, m_P; l_P/t_P=c; Hubble radius; Schwarzschild radius; Λ |

**Total: 50 checks — 50 passed (100%)** as of the latest run.

---

## Checksums

A `ChecksumRecord` is computed after each validator step:
- **n_pass / n_checks** — pass/fail count
- **abs_error_sum** — sum of |relative errors| (lower is better)
- **SHA-256** — hex digest of the serialised (name, modelled, observed) pairs for bit-exact reproducibility

The cumulative SHA-256 fingerprint can be compared across environments to verify that results are identical.

---

## Reproducibility

All computations use only deterministic arithmetic:
- SymPy for exact symbolic verification
- NumPy/cmath for IEEE 754 floating-point verification
- No random seeds, no network calls at runtime (data is hard-coded or read from `scipy.constants`)

To reproduce exactly:

```bash
pip install numpy==1.24 scipy==1.10 sympy==1.12 matplotlib==3.7 tabulate==0.9
python empirical-validation/run_validation.py
```
