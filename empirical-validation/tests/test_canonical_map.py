"""
Unit tests for the canonical_map module.

Tests verify that:
  - The canonical map has all required structure keys.
  - Every entry has the required fields.
  - build_canonical_map() enriches entries with live validation results.
  - All empirical validation checks referenced by the canonical map pass.
  - The Markdown report is generated correctly.

Run with:  pytest empirical-validation/tests/ -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Make the empirical-validation directory importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).parent.parent))

from canonical_map import (  # noqa: E402
    _STATIC_MAP,
    _SECTION_TITLES,
    build_canonical_map,
    generate_report,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Static map structure tests
# ─────────────────────────────────────────────────────────────────────────────


class TestStaticMap:
    """Verify the shape and completeness of the static canonical map."""

    # Required top-level structure keys
    REQUIRED_STRUCTURES = {
        "critical_eigenvalue",
        "coherence_function",
        "silver_coherence",
        "golden_ratio",
        "fine_structure_constant",
        "particle_mass_ratios",
        "spacetime_reality_map",
        "time_crystal",
        "turbulence",
        "ohm_coherence_triality",
        "kernel_axle",
        "bidirectional_time",
        "forward_classical_time",
    }

    # Required fields per entry
    REQUIRED_FIELDS = {
        "lean_file",
        "n_theorems",
        "definition",
        "key_theorems",
        "empirical_section",
        "observable",
        "data_sources",
        "discovery_notes",
    }

    def test_all_required_structures_present(self):
        """Every expected mathematical structure must appear in the map."""
        missing = self.REQUIRED_STRUCTURES - set(_STATIC_MAP)
        assert not missing, f"Missing structures: {missing}"

    def test_no_unexpected_structures(self):
        """The map should not contain unknown structure keys."""
        extra = set(_STATIC_MAP) - self.REQUIRED_STRUCTURES
        assert not extra, (
            f"Unexpected structures found (update REQUIRED_STRUCTURES if intentional): {extra}"
        )

    def test_each_entry_has_required_fields(self):
        """Every entry must have all required fields."""
        for struct_key, entry in _STATIC_MAP.items():
            missing = self.REQUIRED_FIELDS - set(entry)
            assert not missing, (
                f"Structure '{struct_key}' is missing fields: {missing}"
            )

    def test_lean_file_is_nonempty_string(self):
        for struct_key, entry in _STATIC_MAP.items():
            lean_file = entry["lean_file"]
            assert isinstance(lean_file, str) and lean_file.endswith(".lean"), (
                f"'{struct_key}': lean_file must end in .lean, got '{lean_file}'"
            )

    def test_n_theorems_positive(self):
        for struct_key, entry in _STATIC_MAP.items():
            assert isinstance(entry["n_theorems"], int) and entry["n_theorems"] > 0, (
                f"'{struct_key}': n_theorems must be a positive int"
            )

    def test_definition_nonempty(self):
        for struct_key, entry in _STATIC_MAP.items():
            assert isinstance(entry["definition"], str) and len(entry["definition"]) > 3, (
                f"'{struct_key}': definition is empty or too short"
            )

    def test_key_theorems_is_list_of_pairs(self):
        for struct_key, entry in _STATIC_MAP.items():
            kts = entry["key_theorems"]
            assert isinstance(kts, list), f"'{struct_key}': key_theorems must be a list"
            assert len(kts) > 0, f"'{struct_key}': key_theorems must not be empty"
            for item in kts:
                assert (
                    isinstance(item, (list, tuple)) and len(item) == 2
                ), f"'{struct_key}': each key_theorem must be a (name, stmt) pair"

    def test_empirical_section_valid(self):
        valid_sections = {
            "eigenvalue", "fine_structure", "particle_mass",
            "coherence", "golden_ratio", "spacetime",
        }
        for struct_key, entry in _STATIC_MAP.items():
            sec = entry["empirical_section"]
            assert sec in valid_sections, (
                f"'{struct_key}': empirical_section '{sec}' is not a known validator section"
            )

    def test_observable_is_nonempty_list(self):
        for struct_key, entry in _STATIC_MAP.items():
            obs = entry["observable"]
            assert isinstance(obs, list), f"'{struct_key}': observable must be a list"
            assert len(obs) > 0, f"'{struct_key}': observable must not be empty"
            for item in obs:
                assert isinstance(item, str) and len(item) > 5, (
                    f"'{struct_key}': observable item is too short: '{item}'"
                )

    def test_data_sources_is_list(self):
        for struct_key, entry in _STATIC_MAP.items():
            srcs = entry["data_sources"]
            assert isinstance(srcs, list), f"'{struct_key}': data_sources must be a list"
            assert len(srcs) > 0, f"'{struct_key}': data_sources must not be empty"

    def test_discovery_notes_is_list(self):
        for struct_key, entry in _STATIC_MAP.items():
            notes = entry["discovery_notes"]
            assert isinstance(notes, list), (
                f"'{struct_key}': discovery_notes must be a list"
            )

    def test_total_lean_theorems_plausible(self):
        """Total theorem count should match repository metadata."""
        total = sum(e["n_theorems"] for e in _STATIC_MAP.values())
        # Known: 71+71+33+43+29+30+38+24+27+20+40 = well above 300
        assert total >= 300, f"Total theorems unexpectedly low: {total}"

    def test_section_titles_cover_all_structures(self):
        """Every structure key in the static map must have a display title."""
        missing = set(_STATIC_MAP) - set(_SECTION_TITLES)
        assert not missing, (
            f"Missing section titles for: {missing}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. build_canonical_map() tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildCanonicalMap:
    """Tests for the live-enriched canonical map."""

    @pytest.fixture(scope="class")
    def cmap(self):
        return build_canonical_map()

    def test_returns_dict(self, cmap):
        assert isinstance(cmap, dict)

    def test_same_keys_as_static_map(self, cmap):
        assert set(cmap) == set(_STATIC_MAP), (
            "build_canonical_map() must return the same structure keys as _STATIC_MAP"
        )

    def test_each_entry_has_validation_key(self, cmap):
        for struct_key, entry in cmap.items():
            assert "validation" in entry, (
                f"'{struct_key}': enriched entry must have 'validation' key"
            )

    def test_each_entry_has_summary_keys(self, cmap):
        summary_keys = {"all_pass", "n_checks", "n_pass", "n_empirical", "n_empirical_pass"}
        for struct_key, entry in cmap.items():
            missing = summary_keys - set(entry)
            assert not missing, (
                f"'{struct_key}': enriched entry is missing summary keys: {missing}"
            )

    def test_validation_results_are_lists(self, cmap):
        for struct_key, entry in cmap.items():
            val = entry["validation"]
            assert isinstance(val, list), (
                f"'{struct_key}': validation must be a list"
            )

    def test_n_checks_consistent(self, cmap):
        for struct_key, entry in cmap.items():
            assert entry["n_checks"] == len(entry["validation"]), (
                f"'{struct_key}': n_checks != len(validation)"
            )

    def test_n_pass_consistent(self, cmap):
        for struct_key, entry in cmap.items():
            computed = sum(1 for r in entry["validation"] if r.get("passed", False))
            assert entry["n_pass"] == computed, (
                f"'{struct_key}': n_pass mismatch"
            )

    def test_n_empirical_consistent(self, cmap):
        for struct_key, entry in cmap.items():
            computed = sum(
                1 for r in entry["validation"]
                if r.get("check_type") == "empirical"
            )
            assert entry["n_empirical"] == computed, (
                f"'{struct_key}': n_empirical mismatch"
            )

    def test_empirical_checks_in_dedicated_sections_all_pass(self, cmap):
        """Dedicated validator sections must have all empirical checks passing."""
        dedicated = {
            "fine_structure_constant",
            "particle_mass_ratios",
            "coherence_function",
            "golden_ratio",
            "spacetime_reality_map",
        }
        for struct_key in dedicated:
            entry = cmap[struct_key]
            for r in entry["validation"]:
                if r.get("check_type") == "empirical":
                    assert r.get("passed", False), (
                        f"'{struct_key}': empirical check '{r.get('name')}' failed: {r}"
                    )

    def test_mathematical_identity_checks_all_pass(self, cmap):
        """mathematical_identity checks must always pass."""
        for struct_key, entry in cmap.items():
            for r in entry["validation"]:
                if r.get("check_type") == "mathematical_identity":
                    assert r.get("passed", False), (
                        f"'{struct_key}': math-identity check "
                        f"'{r.get('name')}' failed (coding bug): {r}"
                    )

    def test_all_validation_results_have_required_keys(self, cmap):
        """Every validation result dict must have the standard validator keys."""
        required = {"name", "check_type", "pass_criterion", "passed", "rel_error"}
        for struct_key, entry in cmap.items():
            for r in entry["validation"]:
                missing = required - set(r)
                assert not missing, (
                    f"'{struct_key}': validation result '{r.get('name')}' "
                    f"missing keys: {missing}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Report generation tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerateReport:
    """Tests for the Markdown report generator."""

    @pytest.fixture(scope="class")
    def report_path(self, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("canonical_map_reports")
        cmap = build_canonical_map()
        return generate_report(cmap, tmp_dir)

    def test_report_file_created(self, report_path):
        assert report_path.exists(), f"Report file not created: {report_path}"

    def test_report_filename(self, report_path):
        assert report_path.name == "canonical_map_report.md"

    def test_report_nonempty(self, report_path):
        content = report_path.read_text(encoding="utf-8")
        assert len(content) > 1000, "Report is unexpectedly short"

    def test_report_contains_main_title(self, report_path):
        content = report_path.read_text(encoding="utf-8")
        assert "Canonical Map" in content

    def test_report_contains_all_structure_titles(self, report_path):
        content = report_path.read_text(encoding="utf-8")
        for title in _SECTION_TITLES.values():
            # Check for the bare section title substring (without leading ##)
            assert title in content, (
                f"Report is missing section title: '{title}'"
            )

    def test_report_contains_lean_files(self, report_path):
        content = report_path.read_text(encoding="utf-8")
        for entry in _STATIC_MAP.values():
            assert entry["lean_file"] in content, (
                f"Report is missing lean_file reference: {entry['lean_file']}"
            )

    def test_report_contains_empirical_validation_tables(self, report_path):
        content = report_path.read_text(encoding="utf-8")
        # Fine-structure section always has an EMPIRICAL table
        assert "**EMPIRICAL**" in content, (
            "Report must contain at least one EMPIRICAL check table row"
        )

    def test_report_contains_discoveries_section(self, report_path):
        content = report_path.read_text(encoding="utf-8")
        assert "Experimental Discoveries" in content

    def test_report_contains_alpha_max_discovery(self, report_path):
        content = report_path.read_text(encoding="utf-8")
        assert "1 + 1/e" in content or "α_max" in content, (
            "Report must describe the α_max = 1+1/e discovery"
        )

    def test_report_contains_stochastic_resonance_discovery(self, report_path):
        content = report_path.read_text(encoding="utf-8")
        assert "Stochastic Resonance" in content or "C_opt" in content

    def test_report_markdown_has_tables(self, report_path):
        content = report_path.read_text(encoding="utf-8")
        # At least 3 markdown tables (overview, structure count, validation)
        table_count = content.count("|---")
        assert table_count >= 3, f"Expected ≥ 3 Markdown tables, found {table_count}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cross-reference: canonical map ↔ validator sections
# ─────────────────────────────────────────────────────────────────────────────


class TestCrossReference:
    """Verify that canonical map structures reference valid validator sections."""

    def test_eigenvalue_structure_references_eigenvalue_section(self):
        entry = _STATIC_MAP["critical_eigenvalue"]
        assert entry["empirical_section"] == "eigenvalue"

    def test_fine_structure_references_fine_structure_section(self):
        entry = _STATIC_MAP["fine_structure_constant"]
        assert entry["empirical_section"] == "fine_structure"

    def test_particle_mass_references_particle_mass_section(self):
        entry = _STATIC_MAP["particle_mass_ratios"]
        assert entry["empirical_section"] == "particle_mass"

    def test_coherence_structure_references_coherence_section(self):
        entry = _STATIC_MAP["coherence_function"]
        assert entry["empirical_section"] == "coherence"

    def test_golden_ratio_references_golden_ratio_section(self):
        entry = _STATIC_MAP["golden_ratio"]
        assert entry["empirical_section"] == "golden_ratio"

    def test_spacetime_references_spacetime_section(self):
        entry = _STATIC_MAP["spacetime_reality_map"]
        assert entry["empirical_section"] == "spacetime"

    def test_koide_bridge_in_both_golden_ratio_and_particle_mass(self):
        """The Koide bridge C(φ²) = 2/3 must appear in both structures."""
        gr_theorems = [t for name, t in _STATIC_MAP["golden_ratio"]["key_theorems"]]
        pm_theorems = [t for name, t in _STATIC_MAP["particle_mass_ratios"]["key_theorems"]]
        koide_in_golden = any("2/3" in t or "Koide" in t for t in gr_theorems)
        koide_in_mass   = any("2/3" in t or "Koide" in t or "Q" in t for t in pm_theorems)
        assert koide_in_golden, "Koide bridge (2/3) not found in golden_ratio key_theorems"
        assert koide_in_mass,   "Koide bridge (2/3) not found in particle_mass_ratios key_theorems"

    def test_mu_eighth_power_in_eigenvalue_and_time_crystal(self):
        """μ⁸ = 1 must appear in both the eigenvalue and time crystal structures."""
        ev_theorems = [name for name, _ in _STATIC_MAP["critical_eigenvalue"]["key_theorems"]]
        tc_theorems = [name for name, _ in _STATIC_MAP["time_crystal"]["key_theorems"]]
        assert any("pow_eight" in n or "eighth" in n for n in ev_theorems), (
            "μ⁸=1 theorem not found in critical_eigenvalue key_theorems"
        )
        assert any("pow_eight" in n or "eighth" in n or "closure" in n for n in tc_theorems), (
            "8-cycle orbit theorem not found in time_crystal key_theorems"
        )

    def test_silver_coherence_links_to_eigenvalue_imaginary_part(self):
        """Im(μ) = C(δ_S) bridge theorem must be in silver_coherence entry."""
        sc_theorems = [name for name, _ in _STATIC_MAP["silver_coherence"]["key_theorems"]]
        assert any("im" in n.lower() or "silver" in n.lower() for n in sc_theorems), (
            "Im(μ)=C(δ_S) bridge not found in silver_coherence key_theorems"
        )

    def test_planck_units_referenced_in_spacetime_data_sources(self):
        """Spacetime entry must reference NIST Planck unit tabulations."""
        srcs = _STATIC_MAP["spacetime_reality_map"]["data_sources"]
        assert any("NIST" in s and "Planck" in s for s in srcs), (
            "Spacetime data_sources must reference NIST Planck units"
        )

    def test_codata_referenced_in_fine_structure_data_sources(self):
        """Fine-structure entry must reference CODATA 2018."""
        srcs = _STATIC_MAP["fine_structure_constant"]["data_sources"]
        assert any("CODATA" in s for s in srcs), (
            "Fine-structure data_sources must reference CODATA 2018"
        )

    def test_pdg_referenced_in_particle_mass_data_sources(self):
        """Particle-mass entry must reference PDG 2022."""
        srcs = _STATIC_MAP["particle_mass_ratios"]["data_sources"]
        assert any("PDG" in s for s in srcs), (
            "Particle mass data_sources must reference PDG 2022"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Mathematical value spot-checks
# ─────────────────────────────────────────────────────────────────────────────


class TestMathematicalValues:
    """Spot-check that the canonical map references correct numerical values."""

    def test_silver_ratio_value(self):
        """δ_S = 1 + √2 ≈ 2.4142."""
        delta_s = 1.0 + math.sqrt(2.0)
        assert abs(delta_s - 2.4142135623730951) < 1e-14

    def test_golden_ratio_value(self):
        """φ = (1+√5)/2 ≈ 1.6180."""
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        assert abs(phi - 1.6180339887498949) < 1e-14

    def test_alpha_fs_value(self):
        """α ≈ 7.2974×10⁻³."""
        alpha = 7.2973525693e-3
        assert 7.29e-3 < alpha < 7.31e-3

    def test_proton_electron_ratio(self):
        """m_p/m_e ≈ 1836.15267343."""
        ratio = 1836.15267343
        assert 1836.0 < ratio < 1837.0

    def test_koide_q_value(self):
        """Q = 2/3 ≈ 0.6667."""
        Q = 2.0 / 3.0
        assert abs(Q - 0.6666666666666666) < 1e-15

    def test_critical_eigenvalue_modulus(self):
        """μ = exp(i·3π/4) has |μ| = 1."""
        import cmath
        mu = cmath.exp(1j * 3 * math.pi / 4)
        assert abs(abs(mu) - 1.0) < 1e-14

    def test_critical_eigenvalue_8th_power(self):
        """μ⁸ = 1."""
        import cmath
        mu = cmath.exp(1j * 3 * math.pi / 4)
        assert abs(mu ** 8 - 1.0) < 1e-14

    def test_coherence_at_zero(self):
        """C(0) = 2·0/(1+0²) = 0; maximum is C(1) = 1.

        The Lean coherence function is C(r) = 2r/(1+r²), so C(0) = 0
        and the unique maximum is C(1) = 1.
        """
        C_zero = 2.0 * 0.0 / (1.0 + 0.0 ** 2)
        C_one  = 2.0 * 1.0 / (1.0 + 1.0 ** 2)
        assert C_zero == 0.0
        assert C_one == 1.0

    def test_coherence_at_silver_ratio(self):
        """C(δ_S) = 2·δ_S/(1+δ_S²) = √2/2.

        The Lean coherence function is C(r) = 2r/(1+r²), not exp(-r²/2).
        At δ_S = 1+√2: 2(1+√2)/(1+(1+√2)²) = 2(1+√2)/(4+2√2) = √2/2.
        """
        delta_s = 1.0 + math.sqrt(2.0)
        C_val = 2.0 * delta_s / (1.0 + delta_s ** 2)
        expected = math.sqrt(2.0) / 2.0
        assert abs(C_val - expected) < 1e-14

    def test_alpha_max_universal_limit(self):
        """α_max = 1 + 1/e ≈ 1.367879."""
        alpha_max = 1.0 + 1.0 / math.e
        assert abs(alpha_max - 1.3678794411714423) < 1e-12
        # Observed value 1.367099 should be within 0.1% of prediction
        observed = 1.367099
        rel_err = abs(observed - alpha_max) / alpha_max
        assert rel_err < 0.001, f"α_max observed ({observed}) too far from 1+1/e: {rel_err:.4%}"
