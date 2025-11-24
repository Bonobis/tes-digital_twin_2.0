"""FiPy-based heat diffusion solver."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
from packaging.version import Version
from fipy import CellVariable
from fipy.meshes import gmshMesh
from fipy.meshes.gmshMesh import Gmsh3D
from fipy.terms import DiffusionTerm, TransientTerm
from fipy.terms.explicitSourceTerm import _ExplicitSourceTerm
from fipy.terms.implicitSourceTerm import ImplicitSourceTerm
from fipy.solvers.scipy import LinearPCGSolver

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
    turn_off_s: Optional[float]
    ambient: float

    def temperature(self, time_s: float) -> float:
        if self.turn_off_s is not None and time_s >= self.turn_off_s:
            return self.ambient
        if self.ramp_seconds <= 0:
            return self.max_temp
        frac = min(1.0, max(0.0, time_s / self.ramp_seconds))
        return self.ambient + (self.max_temp - self.ambient) * frac


@dataclass
class ProbeDefinition:
    name: str
    position: tuple[float, float, float]


class HeatSolver:
    """Wraps FiPy operations for the digital twin."""

    def __init__(
        self,
        mesh: MeshResult,
        geometry: GeometrySpec,
        catalog: MaterialCatalog,
        scenario: ScenarioConfig,
        settings: SolverSettings,
        probes: Sequence[ProbeDefinition] | None = None,
        telemetry: Callable[[Dict[str, Any]], None] | None = None,
    ):
        self.mesh_result = mesh
        self.geometry = geometry
        self.catalog = catalog
        self.scenario = scenario
        self.settings = settings
        self.probes = list(probes or [])
        self.telemetry = telemetry
        self.mesh_degenerate_faces = 0
        self.element_materials = {
            element.name: element.material
            for element in geometry.elements
            if element.material
        }
        self.ambient_temp = self._infer_ambient_temperature()
        self._probe_cells: list[int] = []
        self._linear_solver = LinearPCGSolver(tolerance=1e-10, iterations=2000)

    def run(self) -> dict[str, Any]:
        mesh = self._load_fipy_mesh()
        material_fields = self._build_material_fields(mesh)
        temperature = CellVariable(
            mesh=mesh,
            name="temperature",
            value=self.ambient_temp,
            hasOld=True,
        )
        conductivity_var = CellVariable(
            mesh=mesh,
            name="conductivity",
            value=material_fields["k"],
        )  # type: ignore[arg-type]
        bc_diag, bc_src, boundary_log = self._apply_boundary_constraints(mesh, temperature, conductivity_var)
        heater_ctx = self._build_heater_context(mesh)
        if heater_ctx.mask is not None:
            temperature.constrain(heater_ctx.temperature(0.0), where=heater_ctx.mask)

        self._anchor_cell(temperature)

        rho_cp_var = CellVariable(
            mesh=mesh,
            name="rho_cp",
            value=material_fields["rho_cp"],
        )  # type: ignore[arg-type]
        equation = (
            TransientTerm(coeff=rho_cp_var)
            == DiffusionTerm(coeff=conductivity_var)  # type: ignore[arg-type]
            - ImplicitSourceTerm(coeff=bc_diag)
            + _ExplicitSourceTerm(coeff=bc_src)
        )

        total_time = self.settings.total_time_s
        dt = self.settings.dt
        time = 0.0
        steps = 0
        temperature.updateOld()
        probe_max: Dict[str, float] = {probe.name: self.ambient_temp for probe in self.probes}
        self._emit_telemetry(
            {
                "event": "start",
                "total_time": total_time,
                "dt": dt,
                "cells": mesh.numberOfCells,
                "probes": [
                    {"name": probe.name, "position_m": probe.position}
                    for probe in self.probes
                ],
            }
        )

        while time < total_time - 1e-9:
            current_dt = min(dt, total_time - time)
            if heater_ctx.mask is not None:
                heater_temp = heater_ctx.temperature(time)
                temperature.constrain(heater_temp, where=heater_ctx.mask)
            equation.solve(var=temperature, dt=current_dt, solver=self._linear_solver)
            temperature.updateOld()
            time += current_dt
            steps += 1
            step_probes = self._collect_probe_values_from_variable(temperature)
            self._emit_telemetry(
                {
                    "event": "step",
                    "step": steps,
                    "time": time,
                    "dt": current_dt,
                    "progress": min(1.0, time / total_time) if total_time else 1.0,
                    "heater_temp": heater_ctx.temperature(time),
                    "probes": step_probes,
                }
            )
            for name, val in step_probes.items():
                if val > probe_max.get(name, -np.inf):
                    probe_max[name] = val

        values = np.array(temperature.value)
        probe_final = self._collect_probe_values(values)
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
            "probe_final_temperatures": probe_final,
            "probe_max_temperatures": probe_max,
            "probe_definitions": [
                {"name": probe.name, "position_m": probe.position} for probe in self.probes
            ],
        }
        self._emit_telemetry({"event": "finish", "result": result})
        return result

    # ------------------------------------------------------------------
    # Helpers

    def _load_fipy_mesh(self) -> Gmsh3D:
        _ensure_gmsh_version()
        mesh = Gmsh3D(str(self.mesh_result.mesh_path))
        self._regularize_mesh(mesh)
        self._assign_probes(mesh)
        return mesh

    def _regularize_mesh(self, mesh: Gmsh3D) -> None:
        """Clamp degenerate geometry early to keep FiPy numerics stable."""
        eps = 1e-4
        distances = np.array(mesh._cellDistances, copy=True)
        zero_mask = np.isclose(distances, 0.0)
        count = int(zero_mask.sum())
        if count:
            distances[zero_mask] = eps
            mesh._cellDistances = distances
        if hasattr(mesh, "_faceToCellDistances"):
            ftc = np.array(mesh._faceToCellDistances, copy=True)
            ftc_zero = np.isclose(ftc, 0.0)
            if ftc_zero.any():
                ftc[ftc_zero] = eps
                mesh._faceToCellDistances = ftc
        if hasattr(mesh, "_scaledFaceAreas"):
            mesh._scaledFaceToCellDistances = mesh._scaledFaceAreas / mesh._cellDistances
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

    def _apply_boundary_constraints(
        self,
        mesh: Gmsh3D,
        temperature: CellVariable,
        conductivity: CellVariable,
    ) -> tuple[CellVariable, CellVariable, list[dict[str, Any]]]:
        """Build implicit boundary sinks and apply Dirichlet constraints."""
        boundaries = self.scenario.boundaries or []
        log: list[dict[str, Any]] = []
        num_cells = mesh.numberOfCells
        diag = np.zeros(num_cells, dtype=float)
        src = np.zeros(num_cells, dtype=float)
        k_faces = conductivity.arithmeticFaceValue
        k_values = np.asarray(k_faces.value)
        k_safe = np.where(k_values <= 0.0, 1e-12, k_values)
        face_areas = np.asarray(mesh._faceAreas)
        face_centers = np.asarray(mesh.faceCenters)
        cell_volumes = np.asarray(mesh.cellVolumes)
        face_cells = np.asarray(mesh.faceCellIDs)
        sigma = 5.670374419e-8
        for boundary in boundaries:
            mask_var = mesh.physicalFaces.get(boundary.region)
            if mask_var is None:
                continue
            base_mask = np.array(mask_var.value, dtype=bool)
            mask = self._filter_boundary_faces(boundary.region, base_mask, face_centers)
            if not mask.any():
                continue
            region_cells = self._region_cell_mask(mesh, boundary.region)
            btype = boundary.type.lower()
            emissivity_used: Optional[float] = None
            area_sum = float(face_areas[mask].sum())
            if btype in {"convection", "radiation"}:
                if boundary.h_coeff is None:
                    raise ValueError(f"Boundary '{boundary.region}' missing h_coeff")
                h_total = float(boundary.h_coeff)
                if btype == "radiation":
                    emissivity = getattr(boundary, "emissivity", None)
                    emissivity_used = float(emissivity) if emissivity is not None else 0.9
                    t_k = float(boundary.value) + 273.15
                    h_total += 4.0 * sigma * emissivity_used * (t_k ** 3)
                _ = k_safe  # keep reference to conductivity for clarity
                for face_idx in np.nonzero(mask)[0]:
                    cells = face_cells[:, face_idx]
                    target = self._select_boundary_cell(cells, region_cells)
                    area = float(face_areas[face_idx])
                    coeff = h_total * area / float(cell_volumes[target])
                    diag[target] += coeff
                    src[target] += coeff * float(boundary.value)
            else:
                if region_cells is not None:
                    temperature.constrain(float(boundary.value), where=region_cells)
            log.append(
                {
                    "region": boundary.region,
                    "type": boundary.type,
                    "value": boundary.value,
                    "h_coeff": getattr(boundary, "h_coeff", None),
                    "emissivity": emissivity_used,
                    "faces": int(mask.sum()),
                    "area": area_sum,
                }
            )
        if not log:
            log.append(
                {
                    "region": "none",
                    "type": "none",
                    "value": None,
                    "h_coeff": None,
                    "emissivity": None,
                    "faces": 0,
                    "area": 0.0,
                }
            )
        diag_var = CellVariable(mesh=mesh, value=diag)
        src_var = CellVariable(mesh=mesh, value=src)
        return diag_var, src_var, log

    def _filter_boundary_faces(
        self,
        region: str,
        mask: np.ndarray,
        face_centers: np.ndarray,
    ) -> np.ndarray:
        record = self.mesh_result.elements.get(region)
        if record and record.bbox:
            origin = np.array(record.bbox.origin) * self.geometry.unit_scale
            size = np.array(record.bbox.size) * self.geometry.unit_scale
            mins = origin
            maxs = origin + size
            tol = 1e-5
            boundary_mask = (
                np.isclose(face_centers[0], mins[0], atol=tol)
                | np.isclose(face_centers[0], maxs[0], atol=tol)
                | np.isclose(face_centers[1], mins[1], atol=tol)
                | np.isclose(face_centers[1], maxs[1], atol=tol)
                | np.isclose(face_centers[2], mins[2], atol=tol)
                | np.isclose(face_centers[2], maxs[2], atol=tol)
            )
            filtered = mask & boundary_mask
            if filtered.any():
                return filtered
        return mask

    def _region_cell_mask(self, mesh: Gmsh3D, region: str) -> Optional[np.ndarray]:
        mask = self._cell_mask(mesh, region)
        material = self.element_materials.get(region)
        if material:
            material_mask = self._cell_mask(mesh, material)
            if material_mask is not None:
                mask = material_mask if mask is None else (mask | material_mask)
        return mask

    def _select_boundary_cell(self, cells: np.ndarray, region_mask: Optional[np.ndarray]) -> int:
        if region_mask is not None:
            for cell in cells:
                if region_mask[int(cell)]:
                    return int(cell)
        return int(cells[0])

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
        ramp_seconds = max(0.0, heater.ramp_s)
        off_s = heater.turn_off_s
        return HeaterContext(
            mask=mask,
            max_temp=heater.max_temperature_c,
            ramp_seconds=ramp_seconds,
            turn_off_s=off_s,
            ambient=self.ambient_temp,
        )

    def _cell_mask(self, mesh: Gmsh3D, name: Optional[str]):
        if not name:
            return None
        var = mesh.physicalCells.get(name)
        if var is None:
            return None
        return np.array(var.value, dtype=bool)

    def _collect_probe_values_from_variable(self, temperature: CellVariable) -> Dict[str, float]:
        if not self.probes or not self._probe_cells:
            return {}
        values = np.asarray(temperature.value)
        return self._collect_probe_values(values)

    def _collect_probe_values(self, values: np.ndarray) -> Dict[str, float]:
        if not self.probes or not self._probe_cells:
            return {}
        data: Dict[str, float] = {}
        for probe, idx in zip(self.probes, self._probe_cells):
            data[probe.name] = float(values[idx])
        return data

    def _assign_probes(self, mesh: Gmsh3D) -> None:
        if not self.probes:
            self._probe_cells = []
            return
        centers = np.array(mesh.cellCenters.value.T)
        if centers.size == 0:
            self._probe_cells = []
            return
        indices: list[int] = []
        for probe in self.probes:
            target = np.array(probe.position)
            distances = np.sum((centers - target) ** 2, axis=1)
            idx = int(distances.argmin())
            indices.append(idx)
        self._probe_cells = indices

    def _emit_telemetry(self, payload: Dict[str, Any]) -> None:
        if self.telemetry is not None:
            self.telemetry(payload)


def _ensure_gmsh_version() -> None:
    """FiPy requires a gmsh CLI version >= 2.0; fake it if unavailable."""
    version = gmshMesh._gmshVersion()
    if version < Version("2.0"):
        gmshMesh._gmshVersion = lambda communicator=None: Version("4.0")
