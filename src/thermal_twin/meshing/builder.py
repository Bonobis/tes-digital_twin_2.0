"""gmsh-based mesh builder placeholders."""
from __future__ import annotations

from pathlib import Path
from typing import Any


class MeshBuilder:
    """Creates gmsh models and exports meshes for FiPy."""

    def __init__(self, geometry: Any, controls: dict[str, Any]):
        self.geometry = geometry
        self.controls = controls

    def build(self) -> dict[str, Any]:
        """Return mock mesh metadata until gmsh integration is added."""
        return {"elements": len(self.geometry.elements), "controls": self.controls}

    def write(self, path: Path) -> None:
        data = self.build()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(data))
