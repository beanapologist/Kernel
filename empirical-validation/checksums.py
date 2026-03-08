"""
Checksum utilities for validation accuracy tracking.
=====================================================
Provides deterministic, reproducible checksums at each validation step so
that discrepancies between modelled and observed values can be detected and
logged.  Two complementary checksums are used:

  1. **Absolute error checksum** — sum of absolute relative errors, rounded
     to a fixed number of decimal places.  Sensitive to the overall
     calibration quality of the model.

  2. **SHA-256 fingerprint** — a hex digest of the serialised (name, value)
     pairs in a validation pass.  Useful for bit-exact reproducibility checks.

Both are included in the ``ChecksumRecord`` returned by ``compute()``.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, NamedTuple


class ChecksumRecord(NamedTuple):
    """Immutable record produced by ``compute()``."""

    n_checks: int
    """Number of individual validation checks included."""

    n_pass: int
    """Number of checks that passed (relative error ≤ tolerance)."""

    n_fail: int
    """Number of checks that failed."""

    abs_error_sum: float
    """Sum of |relative_error| across all checks (lower is better)."""

    sha256: str
    """Hex-encoded SHA-256 digest of the serialised check data."""


def compute(results: list[dict[str, Any]]) -> ChecksumRecord:
    """Compute a ``ChecksumRecord`` from a list of validation result dicts.

    Each entry in *results* is expected to have at least the following keys:

    ``"name"``
        Human-readable name of the validated quantity.
    ``"modelled"``
        The value predicted by the theoretical framework.
    ``"observed"``
        The reference (empirical) value.
    ``"passed"``
        ``True`` if the check succeeded.
    ``"rel_error"``
        Relative error ``(modelled − observed) / observed``.

    Parameters
    ----------
    results:
        List of validation result dicts (as produced by each validator module).

    Returns
    -------
    ChecksumRecord
    """
    n_pass = sum(1 for r in results if r.get("passed", False))
    n_fail = len(results) - n_pass
    abs_error_sum = sum(abs(r.get("rel_error", 0.0)) for r in results)

    # Build a canonically sorted, deterministic JSON string for hashing.
    # Use only the stable fields (name, modelled, observed, passed) so that
    # the fingerprint is reproducible even if extra metadata is added later.
    canonical = json.dumps(
        [
            {
                "name": r["name"],
                "modelled": _round_sig(r.get("modelled", 0.0), 10),
                "observed": _round_sig(r.get("observed", 0.0), 10),
                "passed": r.get("passed", False),
            }
            for r in sorted(results, key=lambda x: x["name"])
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    sha = hashlib.sha256(canonical.encode()).hexdigest()

    return ChecksumRecord(
        n_checks=len(results),
        n_pass=n_pass,
        n_fail=n_fail,
        abs_error_sum=abs_error_sum,
        sha256=sha,
    )


def _round_sig(value: float, sig: int) -> float:
    """Round *value* to *sig* significant figures for stable hashing."""
    if not math.isfinite(value) or value == 0.0:
        return value
    magnitude = math.floor(math.log10(abs(value)))
    factor = 10 ** (sig - 1 - magnitude)
    return round(value * factor) / factor
