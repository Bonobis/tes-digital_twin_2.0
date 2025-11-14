
"""Scenario orchestration glue between configs, meshing, and solver."""
from __future__ import annotations

from typing import Any

from ..geometry import GeometrySpec
from ..materials import MaterialCatalog
from ..meshing import MeshBuilder
from ..solver import HeatSolver, SolverSettings
from .schema import ScenarioConfig


class ScenarioRunner:
    """Loads configs, builds meshes, and executes the solver."""

    def __init__(self, config: ScenarioConfig):
        self.config = config

    def load_geometry(self) -> GeometrySpec:
        return GeometrySpec.from_file(self.config.geometry_file)

    def run(self) -> dict[str, Any]:
        geometry = self.load_geometry()
        mesh_builder = MeshBuilder(geometry, controls=self.config.mesh.model_dump())
        mesh_result = mesh_builder.build()
        materials = MaterialCatalog.from_file(self.config.materials_file)
        solver = HeatSolver(
            mesh_result,
            geometry=geometry,
            catalog=materials,
            scenario=self.config,
            settings=SolverSettings(total_time_s=self.config.total_time_s, dt=self.config.dt_s),
        )
        result = solver.run()
        result.update(
            {
                "geometry_elements": len(geometry.elements),
                "mesh_path": str(mesh_result.mesh_path),
                "volume_groups": mesh_result.volume_groups,
                "surface_groups": mesh_result.surface_groups,
                "element_groups": mesh_result.element_groups,
            }
        )
        return result
