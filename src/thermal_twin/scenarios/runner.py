
"""Scenario orchestration glue between configs, meshing, and solver."""
from __future__ import annotations

from typing import Any, Callable

from ..geometry import GeometrySpec
from ..geometry.elements import UNIT_SCALES
from ..materials import MaterialCatalog
from ..meshing import MeshBuilder, MeshResult
from ..solver import HeatSolver, ProbeDefinition, SolverSettings
from .schema import ScenarioConfig


class ScenarioRunner:
    """Loads configs, builds meshes, and executes the solver."""

    def __init__(self, config: ScenarioConfig, telemetry: Callable[[dict[str, Any]], None] | None = None):
        self.config = config
        self.telemetry = telemetry

    def load_geometry(self) -> GeometrySpec:
        return GeometrySpec.from_file(self.config.geometry_file)

    def build_mesh(self) -> tuple[GeometrySpec, MeshResult]:
        geometry = self.load_geometry()
        mesh_builder = MeshBuilder(geometry, controls=self.config.mesh.model_dump())
        mesh_result = mesh_builder.build()
        return geometry, mesh_result

    def run(self) -> dict[str, Any]:
        geometry, mesh_result = self.build_mesh()
        materials = MaterialCatalog.from_file(self.config.materials_file)
        probes = self._build_probes(geometry)
        solver = HeatSolver(
            mesh_result,
            geometry=geometry,
            catalog=materials,
            scenario=self.config,
            settings=SolverSettings(total_time_s=self.config.total_time_s, dt=self.config.dt_s),
            probes=probes,
            telemetry=self.telemetry,
        )
        result = solver.run()
        result.update(
            {
                "geometry_elements": len(geometry.elements),
                "mesh_path": str(mesh_result.mesh_path),
                "volume_groups": mesh_result.volume_groups,
                "surface_groups": mesh_result.surface_groups,
                "element_groups": mesh_result.element_groups,
                "mesh_quality": mesh_result.quality,
            }
        )
        return result

    def _build_probes(self, geometry: GeometrySpec) -> list[ProbeDefinition]:
        if not self.config.measurements:
            return []
        default_units = geometry.units.lower()
        probes: list[ProbeDefinition] = []
        for measurement in self.config.measurements:
            units = (measurement.units or default_units).lower()
            if units not in UNIT_SCALES:
                raise ValueError(f"Unsupported measurement units '{units}'")
            scale = UNIT_SCALES[units]
            if len(measurement.position) != 3:
                raise ValueError(f"Measurement '{measurement.name}' must have exactly 3 coordinates")
            position_m: tuple[float, float, float] = (
                float(measurement.position[0]) * scale,
                float(measurement.position[1]) * scale,
                float(measurement.position[2]) * scale,
            )
            probes.append(ProbeDefinition(name=measurement.name, position=position_m))
        return probes
