"""Material catalog + interpolation helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


class MaterialCatalog:
    """Loads material properties and provides temperature interpolation."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    @classmethod
    def from_file(cls, path: Path) -> "MaterialCatalog":
        return cls(yaml.safe_load(Path(path).read_text()))

    def get_property(self, material: str, field: str, temperature: float) -> float:
        entry = self._data["catalog"][material]
        temps = np.array(entry["T"], dtype=float)
        values = np.array(entry[field], dtype=float)
        return float(np.interp(temperature, temps, values))

    @property
    def metadata(self) -> dict[str, Any]:
        return {k: v for k, v in self._data.items() if k != "catalog"}
