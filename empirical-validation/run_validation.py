#!/usr/bin/env python3
"""
Empirical Validation Runner
============================
End-to-end pipeline that ingests publicly available scientific datasets
(CODATA 2018, NIST standards, Planck 2018 cosmological data) and validates
the mathematical constructs of the Kernel framework using SymPy and NumPy.

Usage
-----
    python run_validation.py [--output-dir REPORTS_DIR] [--no-plots]

The runner:
  1. Loads all datasets via the ``data_ingestion`` package.
  2. Runs six validator modules (eigenvalue, fine-structure, particle mass,
     coherence, golden ratio, space-time).
  3. Computes a cumulative checksum after each validator step.
  4. Prints a tabular summary to stdout.
  5. Saves a detailed Markdown report and (optionally) comparison plots to
     the ``reports/`` directory.

Exit code
---------
  0  All checks passed.
  1  One or more checks failed.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Ensure local packages are importable regardless of working directory ────
_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))

from data_ingestion import load_codata, load_cosmological, load_nist  # noqa: E402
from validators import (  # noqa: E402
    validate_coherence,
    validate_eigenvalue,
    validate_fine_structure,
    validate_golden_ratio,
    validate_particle_mass,
    validate_spacetime,
)
from checksums import compute as compute_checksum  # noqa: E402

try:
    from tabulate import tabulate  # type: ignore[import]
    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False

try:
    import matplotlib  # type: ignore[import]
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import]
    import numpy as np
    _HAS_PLOTS = True
except ImportError:
    _HAS_PLOTS = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "✓ PASS"
FAIL = "✗ FAIL"


def _status(r: dict[str, Any]) -> str:
    return PASS if r.get("passed", False) else FAIL


def _fmt_val(v: float) -> str:
    """Format a float for tabular display."""
    if v == 0.0:
        return "0"
    if abs(v) >= 0.001 and abs(v) < 1e7:
        return f"{v:.10g}"
    return f"{v:.6e}"


def _table_rows(results: list[dict[str, Any]]) -> list[list[str]]:
    rows = []
    for r in results:
        rows.append([
            r["name"],
            _fmt_val(r.get("modelled", 0.0)),
            _fmt_val(r.get("observed", 0.0)),
            f"{r.get('rel_error', 0.0):.3e}",
            _status(r),
        ])
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_markdown(
    all_results: list[dict[str, Any]],
    checksums: list[tuple[str, Any]],
    output_dir: Path,
) -> Path:
    """Write a detailed Markdown validation report."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    md_lines: list[str] = [
        f"# Empirical Validation Report",
        f"",
        f"**Generated:** {ts}  ",
        f"**Framework:** Kernel — Quantum Coherence Pipeline  ",
        f"**Data sources:** CODATA 2018 (via SciPy), NIST, Planck 2018, PDG 2022  ",
        f"",
        f"---",
        f"",
        f"## Summary",
        f"",
    ]

    total = len(all_results)
    n_pass = sum(1 for r in all_results if r.get("passed", False))
    n_fail = total - n_pass
    md_lines += [
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total checks | {total} |",
        f"| Passed | {n_pass} |",
        f"| Failed | {n_fail} |",
        f"| Pass rate | {100.0 * n_pass / max(total, 1):.1f}% |",
        f"",
        f"---",
        f"",
    ]

    # Per-section results
    sections = [
        ("Eigenvalue Dynamics (|μ|² = 1)", "eigenvalue"),
        ("Fine-Structure Constant (α)", "fine_structure"),
        ("Particle Mass Ratios", "particle_mass"),
        ("Coherence Function C(r)", "coherence"),
        ("Golden & Silver Ratios", "golden_ratio"),
        ("Space-Time Framework", "spacetime"),
    ]
    for title, key in sections:
        sec_results = [r for r in all_results if r.get("section") == key]
        if not sec_results:
            continue
        md_lines.append(f"## {title}")
        md_lines.append("")
        headers = ["Check", "Modelled", "Observed", "Rel. Error", "Status"]
        rows = _table_rows(sec_results)
        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            md_lines.append("| " + " | ".join(row) + " |")
        md_lines.append("")

    # Checksums
    md_lines += [
        "---",
        "",
        "## Validation Checksums",
        "",
        "Checksums are computed after each validator step to detect",
        "regressions between runs.",
        "",
        "| Step | Checks | Passed | Failed | ΣAbsErr | SHA-256 (first 16) |",
        "|------|--------|--------|--------|---------|-------------------|",
    ]
    for step_name, ck in checksums:
        md_lines.append(
            f"| {step_name} | {ck.n_checks} | {ck.n_pass} | {ck.n_fail} "
            f"| {ck.abs_error_sum:.4e} | `{ck.sha256[:16]}` |"
        )
    md_lines.append("")

    # Methodology
    md_lines += [
        "---",
        "",
        "## Methodology",
        "",
        textwrap.dedent("""\
        ### Data Ingestion
        - **CODATA 2018** physical constants loaded via `scipy.constants`
          (CODATA 2018 values); hard-coded fallback when SciPy is unavailable.
        - **NIST** mathematical constants (φ, δ_S, e, π, √2, √5) computed to
          IEEE 754 double precision and compared against NIST DLMF tabulations.
        - **Planck 2018** cosmological parameters (Table 2,
          TT,TE,EE+lowE+lensing) ingested from hard-coded best-fit values.
        - **PDG 2022** lepton masses used for the Koide formula check.

        ### Validation Strategy
        - Each mathematical construct is validated **twice**: once with SymPy
          for exact symbolic verification and once with NumPy for numerical
          floating-point confirmation.
        - Relative error `|modelled − observed| / |observed|` is the primary
          pass/fail metric; domain-appropriate tolerances are set per check.
        - A cumulative checksum (SHA-256 of the serialised (name, value) pairs
          plus the sum of absolute relative errors) is recorded after each
          validator step to ensure bit-exact reproducibility between runs.

        ### Reproducibility
        Run end-to-end with:
        ```bash
        pip install -r empirical-validation/requirements.txt
        python empirical-validation/run_validation.py
        ```
        """),
    ]

    output_path = output_dir / "validation_report.md"
    output_path.write_text("\n".join(md_lines), encoding="utf-8")
    return output_path


def _generate_plots(
    all_results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate comparison bar charts and error plots."""
    if not _HAS_PLOTS:
        return

    # Separate pass/fail results for visualisation
    names = [r["name"] for r in all_results]
    rel_errors = [abs(r.get("rel_error", 0.0)) for r in all_results]
    passed = [r.get("passed", False) for r in all_results]

    colors = ["#2ecc71" if p else "#e74c3c" for p in passed]

    # --- Plot 1: absolute relative errors per check (log scale) ------------
    fig, ax = plt.subplots(figsize=(14, max(5, 0.35 * len(names))))
    y_pos = np.arange(len(names))
    # Replace zero errors with a very small value for log scale;
    # 1e-17 is below IEEE 754 double machine epsilon (2.2e-16) so it does
    # not interfere with meaningful error bars while keeping the log axis valid.
    plot_errors = [max(e, 1e-17) for e in rel_errors]
    ax.barh(y_pos, plot_errors, color=colors, edgecolor="none", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Absolute Relative Error", fontsize=10)
    ax.set_title("Empirical Validation — Relative Errors per Check", fontsize=11)
    ax.set_xscale("log")
    ax.axvline(1e-9,  color="#3498db", linestyle="--", linewidth=0.8,
               label="1e-9 (high precision)")
    ax.axvline(1e-3,  color="#f39c12", linestyle="--", linewidth=0.8,
               label="1e-3 (0.1% tolerance)")
    ax.legend(fontsize=8)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="PASS"),
        Patch(facecolor="#e74c3c", label="FAIL"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_dir / "relative_errors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: pass/fail summary pie chart --------------------------------
    n_pass = sum(passed)
    n_fail = len(passed) - n_pass
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.pie(
        [n_pass, n_fail],
        labels=[f"PASS ({n_pass})", f"FAIL ({n_fail})"],
        colors=["#2ecc71", "#e74c3c"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 11},
    )
    ax2.set_title("Validation Summary", fontsize=13)
    fig2.savefig(output_dir / "pass_fail_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(output_dir: Path, no_plots: bool = False) -> int:
    """Execute the full validation pipeline.

    Returns
    -------
    int
        0 on full pass, 1 if any check failed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Kernel Empirical Validation Pipeline")
    print("=" * 72)

    # ── Step 0: Data ingestion ──────────────────────────────────────────────
    print("\n[1/7] Ingesting datasets …")
    codata = load_codata()
    nist   = load_nist()
    cosmo  = load_cosmological()
    data   = {**codata, **nist, **cosmo}
    print(f"      Loaded {len(codata)} CODATA constants, "
          f"{len(nist)} NIST constants, "
          f"{len(cosmo)} cosmological constants.")

    # ── Step 1–6: Validators ────────────────────────────────────────────────
    validator_steps = [
        ("Eigenvalue Dynamics",          "eigenvalue",     validate_eigenvalue),
        ("Fine-Structure Constant",      "fine_structure", validate_fine_structure),
        ("Particle Mass Ratios",         "particle_mass",  validate_particle_mass),
        ("Coherence Function C(r)",      "coherence",      validate_coherence),
        ("Golden & Silver Ratios",       "golden_ratio",   validate_golden_ratio),
        ("Space-Time Framework",         "spacetime",      validate_spacetime),
    ]

    all_results: list[dict[str, Any]] = []
    checksums: list[tuple[str, Any]] = []
    cumulative: list[dict[str, Any]] = []

    step_n = 2
    for label, section_key, validator_fn in validator_steps:
        print(f"\n[{step_n}/7] {label} …")
        results = validator_fn(data)
        # Tag each result with its section
        for r in results:
            r["section"] = section_key
        all_results.extend(results)
        cumulative.extend(results)

        # Print per-check results
        headers = ["Check", "Modelled", "Observed", "Rel.Err", "Status"]
        rows = _table_rows(results)
        if _HAS_TABULATE:
            print(tabulate(rows, headers=headers, tablefmt="simple"))
        else:
            for row in rows:
                print("  " + " | ".join(row))

        # Compute checksum for this step
        ck = compute_checksum(results)
        step_ck = compute_checksum(cumulative)
        checksums.append((label, ck))
        n_pass = sum(1 for r in results if r.get("passed", False))
        n_fail = len(results) - n_pass
        print(f"\n  → Checksum: {n_pass}/{len(results)} passed | "
              f"ΣAbsErr={ck.abs_error_sum:.4e} | SHA256={ck.sha256[:16]}…")
        step_n += 1

    # ── Final summary ────────────────────────────────────────────────────────
    total  = len(all_results)
    n_pass = sum(1 for r in all_results if r.get("passed", False))
    n_fail = total - n_pass

    print("\n" + "=" * 72)
    print("  FINAL SUMMARY")
    print("=" * 72)
    summary_rows = [
        ["Total checks", str(total)],
        ["Passed",       str(n_pass)],
        ["Failed",       str(n_fail)],
        ["Pass rate",    f"{100.0 * n_pass / max(total, 1):.1f}%"],
    ]
    if _HAS_TABULATE:
        print(tabulate(summary_rows, tablefmt="simple"))
    else:
        for row in summary_rows:
            print(f"  {row[0]:20s}: {row[1]}")

    # Cumulative checksum
    final_ck = compute_checksum(all_results)
    print(f"\nCumulative checksum:")
    print(f"  Checks  : {final_ck.n_checks}")
    print(f"  Passed  : {final_ck.n_pass}")
    print(f"  Failed  : {final_ck.n_fail}")
    print(f"  ΣAbsErr : {final_ck.abs_error_sum:.6e}")
    print(f"  SHA-256 : {final_ck.sha256}")

    # ── Reports ──────────────────────────────────────────────────────────────
    print(f"\n[7/7] Writing reports to {output_dir} …")
    md_path = _generate_markdown(all_results, checksums, output_dir)
    print(f"  Markdown report : {md_path}")
    if not no_plots:
        _generate_plots(all_results, output_dir)
        print(f"  Plots           : {output_dir}/relative_errors.png")
        print(f"                    {output_dir}/pass_fail_summary.png")

    print("\n" + "=" * 72)
    if n_fail == 0:
        print("  ALL CHECKS PASSED ✓")
    else:
        print(f"  {n_fail} CHECK(S) FAILED — see report for details")
    print("=" * 72)

    return 0 if n_fail == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kernel empirical validation pipeline"
    )
    parser.add_argument(
        "--output-dir",
        default=str(_HERE / "reports"),
        help="Directory for report output (default: empirical-validation/reports/)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation (useful in headless environments)",
    )
    args = parser.parse_args()

    exit_code = run(
        output_dir=Path(args.output_dir),
        no_plots=args.no_plots,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
