
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
        gmsh.model.add(self.geometry.name)
        try:
            ctx = BuildContext(self.geometry, gmsh.model.occ, self.geometry.unit_scale)
            for element in self.geometry.elements:
                element.build(ctx)
            self._fragment_elements(ctx)
            gmsh.model.occ.synchronize()
            volume_groups = self._register_volume_groups(ctx)
            surface_groups = self._register_surface_groups(ctx)
            element_groups = self._register_element_groups(ctx)
            self._apply_mesh_sizes(ctx)
            gmsh.model.mesh.generate(3)
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
        for name, record in ctx.elements.items():
            for dimtag in record.dimtags:
                if dimtag[0] == 3:
                    volumes.append(dimtag)
                    owners.append(name)
        if not volumes:
            return
        _, maps = gmsh.model.occ.fragment(volumes, [])
        owner_iter = iter(owners)
        updated: Dict[str, list[tuple[int, int]]] = {name: [] for name in ctx.elements}
        for name, fragment_list in zip(owners, maps):
            updated[name].extend(fragment_list)
        for name, record in ctx.elements.items():
            if not updated.get(name):
                continue
            other = [dimtag for dimtag in record.dimtags if dimtag[0] != 3]
            record.dimtags = updated[name] + other
        ctx.volumes_by_material.clear()
        for record in ctx.elements.values():
            if record.material:
                ctx.volumes_by_material.setdefault(record.material, [])
                ctx.volumes_by_material[record.material].extend(
                    tag for dim, tag in record.dimtags if dim == 3
                )
