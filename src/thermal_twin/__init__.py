"""Thermal Twin core package."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("thermal-twin")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
