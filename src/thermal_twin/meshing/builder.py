
"""gmsh-based mesh builder."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import gmsh
import meshio

from ..geometry import (
    BuildContext,
    ElementRecord,
    GeometryElement,
    GeometrySpec,
    _collect_points_from_dimtags,
)


@dataclass
class MeshResult:
    """Container for gmsh mesh artifacts."""

    mesh_path: Path
    mesh: meshio.Mesh
    volume_groups: Dict[str, int]
    surface_groups: Dict[str, int]
    element_groups: Dict[str, int]
    elements: Dict[str, ElementRecord]
    quality: Dict[str, float]


class MeshBuilder:
    """Creates gmsh models and exports meshes for FiPy."""

    def __init__(self, geometry: GeometrySpec, controls: dict[str, Any], output_dir: Path | None = None):
        self.geometry = geometry
        self.controls = controls
        self.output_dir = output_dir or Path("outputs/meshes")

    def build(self) -> MeshResult:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.model.add(self.geometry.name)
        try:
            ctx = BuildContext(self.geometry, gmsh.model.occ, self.geometry.unit_scale)
            for element in self.geometry.elements:
                element.build(ctx)
            self._fragment_elements(ctx)
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()
            volume_groups = self._register_volume_groups(ctx)
            surface_groups = self._register_surface_groups(ctx)
            element_groups = self._register_element_groups(ctx)
            self._apply_mesh_sizes(ctx)
            gmsh.model.mesh.generate(3)
            gmsh.model.mesh.optimize("Netgen")
            quality = self._evaluate_quality(float(self.controls.get("min_quality", 0.1)))
            self.output_dir.mkdir(parents=True, exist_ok=True)
            mesh_path = self.output_dir / f"{self.geometry.name}.msh"
            gmsh.write(str(mesh_path))
            mesh = meshio.read(mesh_path)
            return MeshResult(
                mesh_path=mesh_path,
                mesh=mesh,
                volume_groups=volume_groups,
                surface_groups=surface_groups,
                element_groups=element_groups,
                elements=ctx.elements,
                quality=quality,
            )
        finally:
            gmsh.finalize()

    def _register_volume_groups(self, ctx: BuildContext) -> Dict[str, int]:
        groups: Dict[str, int] = {}
        for material, tags in ctx.volumes_by_material.items():
            unique = sorted(set(tags))
            if not unique:
                continue
            tag = gmsh.model.addPhysicalGroup(3, unique)
            gmsh.model.setPhysicalName(3, tag, material)
            groups[material] = tag
        return groups

    def _register_surface_groups(self, ctx: BuildContext) -> Dict[str, int]:
        groups: Dict[str, int] = {}
        for name, record in ctx.elements.items():
            surfaces = []
            for dimtag in record.dimtags:
                surfaces.extend(gmsh.model.getBoundary([dimtag], oriented=False, combined=False))
            surface_tags = sorted({tag for dim, tag in surfaces if dim == 2})
            if not surface_tags:
                continue
            group = gmsh.model.addPhysicalGroup(2, surface_tags)
            gmsh.model.setPhysicalName(2, group, name)
            groups[name] = group
        return groups

    def _register_element_groups(self, ctx: BuildContext) -> Dict[str, int]:
        groups: Dict[str, int] = {}
        for name, record in ctx.elements.items():
            volumes = [tag for dim, tag in record.dimtags if dim == 3]
            if not volumes:
                continue
            group = gmsh.model.addPhysicalGroup(3, volumes)
            gmsh.model.setPhysicalName(3, group, name)
            groups[name] = group
        return groups

    def _element_priority(self, element: GeometryElement) -> int:
        role = element.params.get("role")
        if role == "heater":
            return 300
        if element.type == "block":
            return 200
        return 100

    def _apply_mesh_sizes(self, ctx: BuildContext) -> None:
        global_size_mm = float(self.controls.get("global_size_mm", 5.0))
        global_size = global_size_mm * 1e-3
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), global_size)
        rod_size_mm = self.controls.get("rod_refinement_mm")
        if rod_size_mm:
            rod_size = float(rod_size_mm) * 1e-3
            for record in ctx.elements.values():
                if record.element.type != "cylinder" and record.element.params.get("role") != "heater":
                    continue
                points = _collect_points_from_dimtags(record.dimtags)
                if points:
                    gmsh.model.mesh.setSize(points, rod_size)

    def _fragment_elements(self, ctx: BuildContext) -> None:
        volumes: list[tuple[int, int]] = []
        owners: list[str] = []
        priorities: Dict[str, int] = {}
        for name, record in ctx.elements.items():
            priorities[name] = self._element_priority(record.element)
            for dimtag in record.dimtags:
                if dimtag[0] == 3:
                    volumes.append(dimtag)
                    owners.append(name)
        if not volumes:
            return
        _, maps = gmsh.model.occ.fragment(volumes, [])
        fragments_by_owner: Dict[str, list[tuple[int, int]]] = {name: [] for name in ctx.elements}
        for owner, fragment_list in zip(owners, maps):
            fragments_by_owner.setdefault(owner, []).extend(fragment_list)
        assigned: set[tuple[int, int]] = set()
        for name in sorted(priorities, key=lambda item: priorities[item], reverse=True):
            record = ctx.elements[name]
            kept: list[tuple[int, int]] = []
            for fragment in fragments_by_owner.get(name, []):
                if fragment[0] != 3:
                    continue
                if fragment in assigned:
                    continue
                assigned.add(fragment)
                kept.append(fragment)
            non_volumes = [dimtag for dimtag in record.dimtags if dimtag[0] != 3]
            record.dimtags = kept + non_volumes
        ctx.rebuild_material_map()

    def _evaluate_quality(self, min_threshold: float) -> Dict[str, float]:
        """Compute mesh quality stats and enforce the threshold."""
        types, tag_lists, _ = gmsh.model.mesh.getElements(3)
        min_quality = float('inf')
        total = 0.0
        count = 0
        for tags in tag_lists:
            if len(tags) == 0:
                continue
            qualities = gmsh.model.mesh.getElementQualities(list(tags))
            if len(qualities) == 0:
                continue
            local_min = min(qualities)
            if local_min < min_quality:
                min_quality = local_min
            total += sum(qualities)
            count += len(qualities)
        if count == 0 or min_quality == float('inf'):
            raise RuntimeError('Mesh did not produce any 3D elements to evaluate quality.')
        mean_quality = total / count
        if min_quality < min_threshold:
            raise ValueError(
                f"Mesh minimum quality {min_quality:.3f} is below the required threshold {min_threshold:.3f}"
            )
        return {"min": float(min_quality), "mean": float(mean_quality), "count": int(count)}

