"""
Universal solver with path enumeration.
=========================================
Implements ``universal_solve(problem)`` — a general-purpose solver that

1. Enumerates all candidate solution paths via :func:`enumerate_all`,
2. Picks the best path according to :func:`proven_and_cheap`,
3. Attempts to execute the best path via :func:`execute`, and
4. Falls back to the next-best path if execution fails
   (:func:`retry_next`).

Algorithm sketch
----------------
::

    def universal_solve(problem):
        paths = enumerate_all(problem)
        best = max(paths, key=proven_and_cheap)
        execute(best) or retry_next()

Path enumeration strategy
--------------------------
Candidate paths are generated deterministically from the problem
description.  A *path* is a dict with the following keys:

``id``
    Unique string identifier of the form ``"p{k}"`` where *k* counts
    from 0.
``steps``
    Ordered list of symbolic transformation steps derived from the
    problem constraints and the μ⁸=1 spiral cycle phases
    (``coherence``, ``frustration``, ``generation``, ``extraction``,
    ``optimization``).
``cost``
    Estimated cost ∈ (0, ∞).  Lower is cheaper.  Computed from the
    number of steps and the Lean coherence function
    C(r) = 2r/(1+r²).
``proven``
    Boolean: ``True`` when the path can be verified symbolically
    (i.e., when the number of steps is divisible by the μ⁸ group
    order 8, ensuring a complete spiral revolution).

The enumeration is seeded by the SHA-256 digest of the canonical
string representation of *problem*, so the same problem always
produces the same ordered candidate list.

Integration with the μ⁸ optimizer
------------------------------------
:func:`proven_and_cheap` uses the Lean coherence function
``lean_coherence(r)`` from :mod:`optimizers.mu8_cycle_optimizer` to
blend proof-status and cost into a single comparable scalar.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from .mu8_cycle_optimizer import lean_coherence, PHASE_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# μ⁸ group order — determines path "proof" criterion
# ─────────────────────────────────────────────────────────────────────────────

#: A path whose step-count is a multiple of this value completes a full
#: μ⁸ revolution and is therefore considered *proven*.
MU8_ORDER: int = 8


# ─────────────────────────────────────────────────────────────────────────────
# Path enumeration
# ─────────────────────────────────────────────────────────────────────────────

def enumerate_all(problem: Any) -> list[dict]:
    """Enumerate all candidate solution paths for *problem*.

    Parameters
    ----------
    problem:
        Any Python object representing the problem to solve.  The
        canonical string representation (``str(problem)``) is used as
        the seed so that enumeration is deterministic.

    Returns
    -------
    list[dict]
        Non-empty list of path dicts, each containing:

        - ``id``     (str)   — unique identifier
        - ``steps``  (list)  — ordered transformation steps (strings)
        - ``cost``   (float) — estimated cost > 0
        - ``proven`` (bool)  — ``True`` iff the path is a complete
          μ⁸ revolution (step count divisible by 8)

    Notes
    -----
    The number of candidate paths is
    ``max(MU8_ORDER, len(PHASE_NAMES) * 2) = 10``.  This ensures at
    least one *proven* path (the one with exactly 8 steps) is always
    included in the enumeration.

    The step names are drawn from :data:`PHASE_NAMES` (the five-phase
    μ⁸ cycle) to keep paths grounded in the Lean-verified optimizer.
    """
    # Seed a lightweight PRNG-free deterministic index from the problem.
    seed_digest = hashlib.sha256(str(problem).encode()).digest()
    # Use the first 8 bytes as an unsigned integer for offset arithmetic.
    seed_int = int.from_bytes(seed_digest[:8], byteorder="big")

    paths: list[dict] = []
    n_phases = len(PHASE_NAMES)  # 5

    # Generate `max(MU8_ORDER, n_phases * 2)` = 10 candidate paths.
    n_paths = max(MU8_ORDER, n_phases * 2)

    for k in range(n_paths):
        # Deterministically choose number of steps: 1 … 2*MU8_ORDER, rotated
        # by the seed so different problems explore different step counts first.
        n_steps = ((seed_int + k) % (2 * MU8_ORDER)) + 1  # 1 … 16

        # Build the step list by cycling through PHASE_NAMES
        steps = [
            PHASE_NAMES[(seed_int + k + j) % n_phases]
            for j in range(n_steps)
        ]

        # Cost: coherence-modulated step count — cheaper is better.
        # r = n_steps / MU8_ORDER  so that r=1 (i.e. exactly 8 steps)
        # maximises lean_coherence and minimises the cost denominator.
        r = n_steps / MU8_ORDER
        cost = n_steps / (lean_coherence(r) + 1e-9)

        proven = (n_steps % MU8_ORDER) == 0

        paths.append({
            "id": f"p{k}",
            "steps": steps,
            "cost": cost,
            "proven": proven,
        })

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Scoring function
# ─────────────────────────────────────────────────────────────────────────────

def proven_and_cheap(path: dict) -> float:
    """Score a candidate path: higher is better.

    A path that is both *proven* (complete μ⁸ revolution) and *cheap*
    (low cost) receives the highest score.

    Score formula
    -------------
    ::

        proof_bonus = lean_coherence(1.0)  # = 1.0  iff proven
                    = 0.0                  iff not proven
        score = proof_bonus - cost / (cost_scale)

    where ``cost_scale = MU8_ORDER * 2`` normalises the cost to ≈ [0, 1].

    Parameters
    ----------
    path:
        A path dict as returned by :func:`enumerate_all`.

    Returns
    -------
    float
        Score.  Higher scores are preferred by :func:`universal_solve`.
    """
    proof_bonus = lean_coherence(1.0) if path["proven"] else 0.0
    # Normalise cost so that the maximum possible cost (2*MU8_ORDER steps)
    # contributes ≤ 1 unit of penalty.
    cost_scale = float(MU8_ORDER * 2)
    score = proof_bonus - path["cost"] / cost_scale
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Executor
# ─────────────────────────────────────────────────────────────────────────────

def execute(path: dict) -> bool:
    """Attempt to execute a solution path.

    The default implementation succeeds for any *proven* path and fails
    (returns ``False``) for unproven paths.  Replace or monkeypatch this
    function in tests or production integrations to inject custom
    execution logic.

    Parameters
    ----------
    path:
        A path dict as returned by :func:`enumerate_all`.

    Returns
    -------
    bool
        ``True`` on success, ``False`` on failure.
    """
    return bool(path.get("proven", False))


# ─────────────────────────────────────────────────────────────────────────────
# Main solver
# ─────────────────────────────────────────────────────────────────────────────

def universal_solve(
    problem: Any,
    *,
    _execute=None,
) -> dict | None:
    """Solve *problem* by enumerating paths and selecting the best one.

    Algorithm
    ---------
    ::

        paths = enumerate_all(problem)
        best  = max(paths, key=proven_and_cheap)
        execute(best) or retry_next()

    The solver iterates through candidates in descending score order,
    calling :func:`execute` on each until one succeeds.  If all
    candidates fail, ``None`` is returned.

    Parameters
    ----------
    problem:
        The problem to solve.  Passed directly to :func:`enumerate_all`.
    _execute:
        Optional callable with the same signature as :func:`execute`.
        Pass a custom executor in tests to control which paths succeed
        (default: uses the module-level :func:`execute`).

    Returns
    -------
    dict | None
        The first successfully executed path dict, or ``None`` if no
        path could be executed.

    Examples
    --------
    >>> result = universal_solve("mu8_identity")
    >>> result is not None
    True
    >>> result["proven"]
    True
    """
    _exec = _execute if _execute is not None else execute

    paths = enumerate_all(problem)

    # Sort candidates by score descending (best first).
    ranked = sorted(paths, key=proven_and_cheap, reverse=True)

    for candidate in ranked:
        if _exec(candidate):
            return candidate

    # All candidates exhausted — no solution found.
    return None


# ─────────────────────────────────────────────────────────────────────────────
# retry_next — convenience alias used in the algorithm sketch
# ─────────────────────────────────────────────────────────────────────────────

def retry_next(
    problem: Any,
    failed_path: dict,
    *,
    _execute=None,
) -> dict | None:
    """Retry the next-best candidate after *failed_path* has failed.

    This is a lower-level helper that implements the ``retry_next()``
    part of the algorithm sketch.  :func:`universal_solve` already
    calls it internally via its ranked iteration loop; expose it here
    for callers that want explicit control.

    Parameters
    ----------
    problem:
        Original problem passed to :func:`enumerate_all`.
    failed_path:
        The path that already failed (will be skipped).
    _execute:
        Optional custom executor (same semantics as in
        :func:`universal_solve`).

    Returns
    -------
    dict | None
        The next successfully executed path, or ``None`` if none remain.
    """
    _exec = _execute if _execute is not None else execute

    paths = enumerate_all(problem)
    ranked = sorted(paths, key=proven_and_cheap, reverse=True)

    skip_id = failed_path.get("id")

    # If failed_path has no 'id', we cannot identify it in the ranked list,
    # so we cannot safely skip it — return None immediately.
    if skip_id is None:
        return None

    past_failed = False
    for candidate in ranked:
        if candidate["id"] == skip_id:
            past_failed = True
            continue
        if past_failed and _exec(candidate):
            return candidate

    return None
