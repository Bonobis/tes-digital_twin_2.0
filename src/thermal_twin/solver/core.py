"""FiPy-based heat diffusion solver."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
from packaging.version import Version
from fipy import CellVariable
from fipy.meshes import gmshMesh
from fipy.meshes.gmshMesh import Gmsh3D
from fipy.terms import DiffusionTerm, TransientTerm

from ..geometry import GeometrySpec
from ..materials import MaterialCatalog
from ..meshing import MeshResult
from ..scenarios.schema import ScenarioConfig


@dataclass
class SolverSettings:
    total_time_s: float = 3600.0
    dt: float = 20.0
    save_interval: float = 60.0


@dataclass
class MaterialProperties:
    rho: float
    cp: float
    k: float

    @property
    def rho_cp(self) -> float:
        return self.rho * self.cp


@dataclass
class HeaterContext:
    mask: Any | None
    max_temp: float
    ramp_seconds: float

    def temperature(self, time_s: float) -> float:
        if self.ramp_seconds <= 0:
            return self.max_temp
        return self.max_temp * min(1.0, max(0.0, time_s / self.ramp_seconds))


class HeatSolver:
    """Wraps FiPy operations for the digital twin."""

    def __init__(
        self,
        mesh: MeshResult,
        geometry: GeometrySpec,
        catalog: MaterialCatalog,
        scenario: ScenarioConfig,
        settings: SolverSettings,
    ):
        self.mesh_result = mesh
        self.geometry = geometry
        self.catalog = catalog
        self.scenario = scenario
        self.settings = settings
        self.mesh_degenerate_faces = 0
        self.element_materials = {
            element.name: element.material
            for element in geometry.elements
            if element.material
        }
        self.ambient_temp = self._infer_ambient_temperature()

    def run(self) -> dict[str, Any]:
        mesh = self._load_fipy_mesh()
        material_fields = self._build_material_fields(mesh)
        temperature = CellVariable(
            mesh=mesh,
            name="temperature",
            value=self.ambient_temp,
            hasOld=True,
        )
        boundary_log = self._apply_boundary_constraints(mesh, temperature)
        heater_ctx = self._build_heater_context(mesh)
        if heater_ctx.mask is not None:
            temperature.setValue(heater_ctx.temperature(0.0), where=heater_ctx.mask)

        self._anchor_cell(temperature)

        rho_cp_var = CellVariable(mesh=mesh, name="rho_cp", value=material_fields["rho_cp"])  # type: ignore[arg-type]
        conductivity_var = CellVariable(mesh=mesh, name="conductivity", value=material_fields["k"])  # type: ignore[arg-type]
        equation = TransientTerm(coeff=rho_cp_var) == DiffusionTerm(coeff=conductivity_var)  # type: ignore[arg-type]

        total_time = self.settings.total_time_s
        dt = self.settings.dt
        time = 0.0
        steps = 0
        temperature.updateOld()

        while time < total_time - 1e-9:
            current_dt = min(dt, total_time - time)
            if heater_ctx.mask is not None:
                temperature.setValue(heater_ctx.temperature(time), where=heater_ctx.mask)
            equation.solve(var=temperature, dt=current_dt)
            temperature.updateOld()
            time += current_dt
            steps += 1

        values = np.array(temperature.value)
        result: Dict[str, Any] = {
            "steps": steps,
            "dt": self.settings.dt,
            "total_time": total_time,
            "cells": mesh.numberOfCells,
            "min_temp": float(values.min()),
            "max_temp": float(values.max()),
            "mean_temp": float(values.mean()),
            "heater_target_c": self.scenario.heater.max_temperature_c,
            "ambient_c": self.ambient_temp,
            "boundaries": boundary_log,
            "mesh_zero_distance_faces": self.mesh_degenerate_faces,
        }
        return result

    # ------------------------------------------------------------------
    # Helpers

    def _load_fipy_mesh(self) -> Gmsh3D:
        _ensure_gmsh_version()
        mesh = Gmsh3D(str(self.mesh_result.mesh_path))
        self._regularize_mesh(mesh)
        return mesh

    def _regularize_mesh(self, mesh: Gmsh3D) -> None:
        distances = np.array(mesh._cellDistances, copy=True)
        zero_mask = np.isclose(distances, 0.0)
        count = int(zero_mask.sum())
        if count:
            distances[zero_mask] = 1e-12
            mesh._cellDistances = distances
        self.mesh_degenerate_faces = count


    def _infer_ambient_temperature(self) -> float:
        if self.scenario.boundaries:
            return float(self.scenario.boundaries[0].value)
        return 25.0

    def _build_material_fields(self, mesh: Gmsh3D) -> Dict[str, np.ndarray]:
        num_cells = mesh.numberOfCells
        rho = np.zeros(num_cells)
        cp = np.zeros(num_cells)
        k = np.zeros(num_cells)
        assigned = np.zeros(num_cells, dtype=bool)
        for material_name in self.mesh_result.volume_groups:
            mask = self._cell_mask(mesh, material_name)
            if mask is None:
                continue
            mask_array = mask
            if not mask_array.any():
                continue
            props = self._material_properties(material_name, self.ambient_temp)
            rho[mask_array] = props.rho
            cp[mask_array] = props.cp
            k[mask_array] = props.k
            assigned = assigned | mask_array
        if not assigned.all():
            for element_name, material_name in self.element_materials.items():
                mask = self._cell_mask(mesh, element_name)
                if mask is None:
                    continue
                mask_array = mask & (~assigned)
                if not mask_array.any():
                    continue
                props = self._material_properties(material_name, self.ambient_temp)
                rho[mask_array] = props.rho
                cp[mask_array] = props.cp
                k[mask_array] = props.k
                assigned = assigned | mask_array
        if not assigned.all():
            raise ValueError("Some mesh cells do not have an assigned material.")
        return {
            "rho": rho,
            "cp": cp,
            "rho_cp": rho * cp,
            "k": k,
        }

    def _material_properties(self, name: str, temperature: float) -> MaterialProperties:
        rho = self._get_required_property(name, "rho", temperature)
        cp = self._get_required_property(name, "cp", temperature)
        k = self._get_optional_property(name, "k", temperature)
        if k is None:
            alpha = self._get_optional_property(name, "alpha", temperature)
            if alpha is None:
                raise ValueError(f"Material '{name}' must define either k or alpha")
            k = alpha * rho * cp
        return MaterialProperties(rho=rho, cp=cp, k=k)

    def _get_required_property(self, material: str, field: str, temperature: float) -> float:
        value = self._get_optional_property(material, field, temperature)
        if value is None:
            raise ValueError(f"Material '{material}' missing property '{field}'")
        return value

    def _get_optional_property(self, material: str, field: str, temperature: float) -> Optional[float]:
        try:
            return self.catalog.get_property(material, field, temperature)
        except KeyError:
            return None

    def _apply_boundary_constraints(self, mesh: Gmsh3D, temperature: CellVariable) -> list[dict[str, Any]]:
        log: list[dict[str, Any]] = []
        for boundary in self.scenario.boundaries:
            mask = mesh.physicalFaces.get(boundary.region)
            if mask is None:
                continue
            mask_array = np.array(mask.value, dtype=bool)
            temperature.constrain(boundary.value, where=mask_array)
            log.append({
                "region": boundary.region,
                "type": boundary.type,
                "value": boundary.value,
            })
        if not log:
            temperature.constrain(self.ambient_temp, where=np.array(mesh.exteriorFaces.value, dtype=bool))
            log.append({"region": "exterior", "type": "dirichlet", "value": self.ambient_temp})
        return log

    def _anchor_cell(self, temperature: CellVariable) -> None:
        anchor = np.zeros(temperature.mesh.numberOfCells, dtype=bool)
        anchor[0] = True
        temperature.constrain(self.ambient_temp, where=anchor)

    def _build_heater_context(self, mesh: Gmsh3D) -> HeaterContext:
        heater = self.scenario.heater
        mask = self._cell_mask(mesh, heater.name)
        if mask is None:
            material = self.element_materials.get(heater.name)
            if material:
                mask = self._cell_mask(mesh, material)
        ramp_seconds = max(0.0, heater.ramp_minutes * 60.0)
        return HeaterContext(mask=mask, max_temp=heater.max_temperature_c, ramp_seconds=ramp_seconds)

    def _cell_mask(self, mesh: Gmsh3D, name: Optional[str]):
        if not name:
            return None
        var = mesh.physicalCells.get(name)
        if var is None:
            return None
        return np.array(var.value, dtype=bool)


def _ensure_gmsh_version() -> None:
    """FiPy requires a gmsh CLI version >= 2.0; fake it if unavailable."""
    version = gmshMesh._gmshVersion()
    if version < Version("2.0"):
        gmshMesh._gmshVersion = lambda communicator=None: Version("4.0")
