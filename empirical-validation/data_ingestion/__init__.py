"""
Data ingestion package for empirical validation.

Provides loaders for CODATA physical constants, NIST standards, and
publicly available cosmological/astrophysical datasets.  Each module
exposes a ``load()`` function that returns a plain ``dict`` of named
constants or records so that downstream validators remain decoupled
from the source format.
"""

from .codata import load as load_codata
from .nist import load as load_nist
from .cosmological import load as load_cosmological

__all__ = ["load_codata", "load_nist", "load_cosmological"]
