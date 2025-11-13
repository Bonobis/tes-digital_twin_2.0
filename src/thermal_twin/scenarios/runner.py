"""Scenario orchestration glue between configs, meshing, and solver."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..geometry import GeometrySpec
from ..meshing import MeshBuilder
from ..solver import HeatSolver, SolverSettings
from .schema import ScenarioConfig


class ScenarioRunner:
    """Loads configs, builds meshes, and executes the solver."""

    def __init__(self, config: ScenarioConfig):
        self.config = config

    def load_geometry(self) -> GeometrySpec:
        data = json.loads(Path(self.config.geometry_file).read_text())
        return GeometrySpec.from_dict(data)

    def run(self) -> dict[str, Any]:
        geometry = self.load_geometry()
        mesh_builder = MeshBuilder(geometry, controls=self.config.mesh.model_dump())
        mesh_info = mesh_builder.build()
        solver = HeatSolver(
            mesh_info,
            settings=SolverSettings(total_time_s=self.config.total_time_s, dt=self.config.dt_s),
        )
        result = solver.run()
        return {"geometry_elements": len(geometry.elements), **result}
