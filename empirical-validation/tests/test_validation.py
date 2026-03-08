"""
Unit tests for the empirical validation framework.

Run with:  pytest empirical-validation/tests/ -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the empirical-validation directory importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).parent.parent))

from checksums import ChecksumRecord, compute as compute_checksum
from data_ingestion import load_codata, load_cosmological, load_nist
from validators import (
    validate_coherence,
    validate_eigenvalue,
    validate_fine_structure,
    validate_golden_ratio,
    validate_particle_mass,
    validate_spacetime,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data ingestion
# ─────────────────────────────────────────────────────────────────────────────


class TestCodataIngestion:
    """CODATA data-ingestion module tests."""

    def test_load_returns_dict(self):
        data = load_codata()
        assert isinstance(data, dict)

    def test_required_keys_present(self):
        data = load_codata()
        required = [
            "fine_structure_constant",
            "electron_mass",
            "proton_mass",
            "proton_electron_mass_ratio",
            "speed_of_light",
        ]
        for key in required:
            assert key in data, f"Missing key: {key}"

    def test_each_entry_has_value_unit_source(self):
        data = load_codata()
        for key, entry in data.items():
            assert "value" in entry, f"{key}: missing 'value'"
            assert "unit" in entry, f"{key}: missing 'unit'"
            assert "source" in entry, f"{key}: missing 'source'"

    def test_fine_structure_constant_range(self):
        data = load_codata()
        alpha = data["fine_structure_constant"]["value"]
        assert 7.29e-3 < alpha < 7.31e-3, f"α out of expected range: {alpha}"

    def test_speed_of_light_exact(self):
        data = load_codata()
        c = data["speed_of_light"]["value"]
        assert c == pytest.approx(299_792_458.0, rel=1e-9)

    def test_proton_electron_mass_ratio_range(self):
        data = load_codata()
        ratio = data["proton_electron_mass_ratio"]["value"]
        assert 1836.0 < ratio < 1837.0, f"m_p/m_e out of range: {ratio}"


class TestNistIngestion:
    """NIST data-ingestion module tests."""

    def test_load_returns_dict(self):
        data = load_nist()
        assert isinstance(data, dict)

    def test_required_keys_present(self):
        data = load_nist()
        for key in ["golden_ratio", "silver_ratio", "pi", "euler_number", "sqrt2"]:
            assert key in data, f"Missing key: {key}"

    def test_golden_ratio_value(self):
        data = load_nist()
        phi = data["golden_ratio"]["value"]
        assert phi == pytest.approx((1 + math.sqrt(5)) / 2, rel=1e-15)

    def test_silver_ratio_value(self):
        data = load_nist()
        delta_s = data["silver_ratio"]["value"]
        assert delta_s == pytest.approx(1 + math.sqrt(2), rel=1e-15)

    def test_pi_value(self):
        data = load_nist()
        pi_val = data["pi"]["value"]
        assert pi_val == pytest.approx(math.pi, rel=1e-15)


class TestCosmologicalIngestion:
    """Cosmological data-ingestion module tests."""

    def test_load_returns_dict(self):
        data = load_cosmological()
        assert isinstance(data, dict)

    def test_hubble_constant_range(self):
        data = load_cosmological()
        H0 = data["hubble_constant"]["value"]
        assert 60.0 < H0 < 80.0, f"H₀ out of range: {H0} km/s/Mpc"

    def test_cmb_temperature_range(self):
        data = load_cosmological()
        T = data["cmb_temperature"]["value"]
        assert 2.7 < T < 2.8, f"T_CMB out of range: {T} K"

    def test_cosmological_constant_positive(self):
        data = load_cosmological()
        lam = data["cosmological_constant"]["value"]
        assert lam > 0, f"Λ must be positive, got {lam}"

    def test_cosmological_constant_order_of_magnitude(self):
        data = load_cosmological()
        lam = data["cosmological_constant"]["value"]
        # Λ ≈ 10⁻⁵² m⁻² from Planck 2018
        assert 1e-53 < lam < 1e-51, f"Λ order of magnitude wrong: {lam}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Checksum module
# ─────────────────────────────────────────────────────────────────────────────


class TestChecksum:
    """Tests for the checksums module."""

    def _make_results(self) -> list[dict]:
        return [
            {"name": "a", "modelled": 1.0, "observed": 1.0,
             "rel_error": 0.0, "passed": True},
            {"name": "b", "modelled": 1.01, "observed": 1.0,
             "rel_error": 0.01, "passed": False},
        ]

    def test_returns_checksum_record(self):
        ck = compute_checksum(self._make_results())
        assert isinstance(ck, ChecksumRecord)

    def test_n_checks_correct(self):
        ck = compute_checksum(self._make_results())
        assert ck.n_checks == 2

    def test_n_pass_n_fail(self):
        ck = compute_checksum(self._make_results())
        assert ck.n_pass == 1
        assert ck.n_fail == 1

    def test_abs_error_sum(self):
        ck = compute_checksum(self._make_results())
        assert ck.abs_error_sum == pytest.approx(0.01, rel=1e-9)

    def test_sha256_is_hex_string_of_length_64(self):
        ck = compute_checksum(self._make_results())
        assert isinstance(ck.sha256, str)
        assert len(ck.sha256) == 64
        int(ck.sha256, 16)   # must be valid hex

    def test_sha256_deterministic(self):
        results = self._make_results()
        ck1 = compute_checksum(results)
        ck2 = compute_checksum(results)
        assert ck1.sha256 == ck2.sha256

    def test_empty_results(self):
        ck = compute_checksum([])
        assert ck.n_checks == 0
        assert ck.n_pass == 0
        assert ck.n_fail == 0
        assert ck.abs_error_sum == 0.0

    def test_sha256_changes_with_different_values(self):
        r1 = [{"name": "x", "modelled": 1.0, "observed": 1.0,
                "rel_error": 0.0, "passed": True}]
        r2 = [{"name": "x", "modelled": 2.0, "observed": 1.0,
                "rel_error": 1.0, "passed": False}]
        assert compute_checksum(r1).sha256 != compute_checksum(r2).sha256


# ─────────────────────────────────────────────────────────────────────────────
# 3. Eigenvalue validator
# ─────────────────────────────────────────────────────────────────────────────


class TestEigenvalueValidator:
    """Tests for the eigenvalue validator."""

    @pytest.fixture(scope="class")
    def results(self):
        return validate_eigenvalue()

    def test_returns_list(self, results):
        assert isinstance(results, list)
        assert len(results) > 0

    def test_all_results_have_required_keys(self, results):
        for r in results:
            assert "name" in r
            assert "modelled" in r
            assert "observed" in r
            assert "passed" in r
            assert "rel_error" in r

    def test_norm_sq_passes(self, results):
        r = next(r for r in results if "norm_sq" in r["name"])
        assert r["passed"], f"norm_sq check failed: {r}"

    def test_8th_power_passes(self, results):
        r = next(r for r in results if "8th_power" in r["name"])
        assert r["passed"], f"8th_power check failed: {r}"

    def test_rotation_matrix_eigenvalue_norm(self, results):
        r = next(r for r in results if "rotation_matrix_eigenvalue_norm" in r["name"])
        assert r["passed"], f"rotation matrix check failed: {r}"

    def test_rotation_matrix_8th_power_identity(self, results):
        r = next(r for r in results if "rotation_matrix_8th_power_identity" in r["name"])
        assert r["passed"], f"R⁸=I check failed: {r}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Fine-structure constant validator
# ─────────────────────────────────────────────────────────────────────────────


class TestFineStructureValidator:
    """Tests for the fine-structure constant validator."""

    @pytest.fixture(scope="class")
    def results(self):
        codata = load_codata()
        return validate_fine_structure(codata)

    def test_returns_list(self, results):
        assert isinstance(results, list)
        assert len(results) > 0

    def test_definition_check_passes(self, results):
        r = next(r for r in results if "definition" in r["name"])
        assert r["passed"], f"α definition check failed: {r}"

    def test_inverse_check_passes(self, results):
        r = next(r for r in results if "inverse" in r["name"]
                 and "sympy" not in r["name"])
        assert r["passed"], f"1/α check failed: {r}"

    def test_sub_unity_passes(self, results):
        r = next(r for r in results if "sub_unity" in r["name"])
        assert r["passed"], f"α < 1 check failed: {r}"

    def test_alpha_value_in_correct_range(self, results):
        r = next(r for r in results if "definition" in r["name"])
        alpha = r["modelled"]
        assert 7.2e-3 < alpha < 7.4e-3, f"α = {alpha} out of range"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Particle mass validator
# ─────────────────────────────────────────────────────────────────────────────


class TestParticleMassValidator:
    """Tests for the particle mass ratio validator."""

    @pytest.fixture(scope="class")
    def results(self):
        codata = load_codata()
        return validate_particle_mass(codata)

    def test_returns_list(self, results):
        assert isinstance(results, list)

    def test_codata_ratio_passes(self, results):
        r = next(r for r in results if "codata" in r["name"])
        assert r["passed"], f"CODATA ratio check failed: {r}"

    def test_koide_formula_passes(self, results):
        r = next(r for r in results if "koide" in r["name"])
        assert r["passed"], f"Koide formula check failed: {r}"
        # Koide Q should be very close to 2/3
        assert abs(r["modelled"] - 2/3) < 0.005

    def test_wyler_approximation_passes(self, results):
        r = next(r for r in results if "wyler" in r["name"])
        assert r["passed"], f"Wyler approximation check failed: {r}"

    def test_proton_electron_ratio_reconstructed(self, results):
        r = next(r for r in results if "reconstructed" in r["name"])
        assert r["passed"], f"Reconstructed ratio check failed: {r}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Coherence function validator
# ─────────────────────────────────────────────────────────────────────────────


class TestCoherenceValidator:
    """Tests for the coherence function C(r) validator."""

    @pytest.fixture(scope="class")
    def results(self):
        return validate_coherence()

    def test_returns_list(self, results):
        assert isinstance(results, list)
        assert len(results) > 0

    def test_c_at_zero_is_1(self, results):
        r = next(r for r in results if "at_zero" in r["name"])
        assert r["passed"]
        assert r["modelled"] == pytest.approx(1.0, abs=1e-14)

    def test_c_strictly_decreasing(self, results):
        r = next(r for r in results if "decreasing" in r["name"])
        assert r["passed"]

    def test_c_range_check(self, results):
        r = next(r for r in results if "range" in r["name"])
        assert r["passed"]

    def test_integral_sqrt_pi_over_2(self, results):
        r = next(r for r in results if "integral" in r["name"])
        assert r["passed"]
        assert r["modelled"] == pytest.approx(math.sqrt(math.pi / 2), rel=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Golden ratio validator
# ─────────────────────────────────────────────────────────────────────────────


class TestGoldenRatioValidator:
    """Tests for the golden ratio and silver ratio validator."""

    @pytest.fixture(scope="class")
    def results(self):
        nist = load_nist()
        return validate_golden_ratio(nist)

    def test_returns_list(self, results):
        assert isinstance(results, list)
        assert len(results) > 0

    def test_quadratic_identity_sympy(self, results):
        r = next(r for r in results if "quadratic_identity" in r["name"])
        assert r["passed"], f"φ² = φ+1 (sympy) failed: {r}"

    def test_reciprocal_identity_sympy(self, results):
        r = next(r for r in results if "reciprocal_identity" in r["name"])
        assert r["passed"], f"φ−1 = 1/φ (sympy) failed: {r}"

    def test_minimal_polynomial_sympy(self, results):
        r = next(r for r in results if "golden_ratio_minimal_polynomial_sympy" in r["name"])
        assert r["passed"], f"φ²−φ−1=0 (sympy) failed: {r}"

    def test_silver_conservation_sympy(self, results):
        r = next(r for r in results if "silver_conservation_sympy" in r["name"])
        assert r["passed"], f"δ_S·(√2−1)=1 (sympy) failed: {r}"

    def test_fibonacci_convergence(self, results):
        r = next(r for r in results if "fibonacci" in r["name"])
        assert r["passed"], f"Fibonacci convergence to φ failed: {r}"

    def test_golden_ratio_value_matches_nist(self, results):
        r = next(r for r in results if "golden_ratio_value_nist" in r["name"])
        assert r["passed"], f"φ value vs NIST failed: {r}"

    def test_silver_ratio_value_matches_nist(self, results):
        r = next(r for r in results if "silver_ratio_value_nist" in r["name"])
        assert r["passed"], f"δ_S value vs NIST failed: {r}"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Space-time framework validator
# ─────────────────────────────────────────────────────────────────────────────


class TestSpacetimeValidator:
    """Tests for the space-time framework validator."""

    @pytest.fixture(scope="class")
    def results(self):
        codata = load_codata()
        cosmo  = load_cosmological()
        return validate_spacetime({**codata, **cosmo})

    def test_returns_list(self, results):
        assert isinstance(results, list)
        assert len(results) > 0

    def test_speed_of_light_exact(self, results):
        r = next(r for r in results if "speed_of_light" in r["name"])
        assert r["passed"]

    def test_planck_time_correct(self, results):
        r = next(r for r in results if "planck_time" in r["name"])
        assert r["passed"], f"Planck time check failed: {r}"
        assert r["modelled"] == pytest.approx(5.391247e-44, rel=5e-4)

    def test_planck_length_correct(self, results):
        r = next(r for r in results if "planck_length" in r["name"])
        assert r["passed"], f"Planck length check failed: {r}"
        assert r["modelled"] == pytest.approx(1.616255e-35, rel=5e-4)

    def test_planck_mass_correct(self, results):
        r = next(r for r in results if "planck_mass" in r["name"])
        assert r["passed"], f"Planck mass check failed: {r}"

    def test_lp_over_tp_equals_c(self, results):
        r = next(r for r in results if "lp_over_tp" in r["name"])
        assert r["passed"], f"l_P/t_P = c check failed: {r}"

    def test_schwarzschild_radius_sun(self, results):
        r = next(r for r in results if "schwarzschild" in r["name"])
        assert r["passed"], f"Schwarzschild radius check failed: {r}"


# ─────────────────────────────────────────────────────────────────────────────
# 9. End-to-end pipeline
# ─────────────────────────────────────────────────────────────────────────────


class TestEndToEndPipeline:
    """Integration test for the complete validation pipeline."""

    def test_all_validators_return_results(self):
        codata = load_codata()
        nist   = load_nist()
        cosmo  = load_cosmological()
        data   = {**codata, **nist, **cosmo}

        all_results = []
        for fn in [
            validate_eigenvalue,
            validate_fine_structure,
            validate_particle_mass,
            validate_coherence,
            validate_golden_ratio,
            validate_spacetime,
        ]:
            results = fn(data)
            assert isinstance(results, list)
            assert len(results) > 0
            all_results.extend(results)

        assert len(all_results) > 30, "Expected at least 30 checks total"

    def test_cumulative_checksum_deterministic(self):
        """Running the pipeline twice must produce the same SHA-256."""
        codata = load_codata()
        nist   = load_nist()
        cosmo  = load_cosmological()
        data   = {**codata, **nist, **cosmo}

        def _run():
            results = []
            for fn in [
                validate_eigenvalue,
                validate_fine_structure,
                validate_particle_mass,
                validate_coherence,
                validate_golden_ratio,
                validate_spacetime,
            ]:
                results.extend(fn(data))
            return compute_checksum(results).sha256

        sha1 = _run()
        sha2 = _run()
        assert sha1 == sha2, "Pipeline checksum is not deterministic"

    def test_majority_of_checks_pass(self):
        """At least 90% of all validation checks should pass."""
        codata = load_codata()
        nist   = load_nist()
        cosmo  = load_cosmological()
        data   = {**codata, **nist, **cosmo}

        all_results = []
        for fn in [
            validate_eigenvalue,
            validate_fine_structure,
            validate_particle_mass,
            validate_coherence,
            validate_golden_ratio,
            validate_spacetime,
        ]:
            all_results.extend(fn(data))

        n_pass = sum(1 for r in all_results if r.get("passed", False))
        pass_rate = n_pass / len(all_results)
        assert pass_rate >= 0.90, (
            f"Pass rate {pass_rate:.1%} < 90%; "
            f"failing checks: {[r['name'] for r in all_results if not r.get('passed')]}"
        )
