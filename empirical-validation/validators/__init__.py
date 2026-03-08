"""
Validators package for empirical validation.

Each sub-module exposes a ``validate(data)`` function that accepts the
combined data dict (CODATA + NIST + cosmological) and returns a list of
``ValidationResult`` named-tuples.
"""

from .eigenvalue import validate as validate_eigenvalue
from .fine_structure import validate as validate_fine_structure
from .particle_mass import validate as validate_particle_mass
from .coherence import validate as validate_coherence
from .golden_ratio import validate as validate_golden_ratio
from .spacetime import validate as validate_spacetime

__all__ = [
    "validate_eigenvalue",
    "validate_fine_structure",
    "validate_particle_mass",
    "validate_coherence",
    "validate_golden_ratio",
    "validate_spacetime",
]
