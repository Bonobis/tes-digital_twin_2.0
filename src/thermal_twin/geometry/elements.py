"""Geometry element abstractions and gmsh builders."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, List, Optional, Tuple, cast

import gmsh


Vec3 = Tuple[float, float, float]

UNIT_SCALES: dict[str, float] = {
    "m": 1.0,
    "cm": 1e-2,
    "mm": 1e-3,
}


@dataclass
class GeometryElement:
    """Base element definition consumed by gmsh builders."""

    type: str
    name: str
    params: dict[str, Any]

    registry: ClassVar[dict[str, Callable[["GeometryElement", "BuildContext"], List[Tuple[int, int]]]]] = {}

    def build(self, ctx: "BuildContext") -> List[Tuple[int, int]]:
        builder = self.registry.get(self.type)
        if builder is None:
            raise ValueError(f"Unsupported geometry element type '{self.type}'")
        return builder(self, ctx)

    @classmethod
    def register(cls, element_type: str):
        def decorator(func: Callable[["GeometryElement", "BuildContext"], List[Tuple[int, int]]]):
            cls.registry[element_type] = func
            return func

        return decorator

    @property
    def material(self) -> Optional[str]:
        return self.params.get("material")


@dataclass
class BoundingBox:
    """Axis-aligned bounding box stored in config units."""

    origin: Vec3
    size: Vec3

    def expand(self, thickness: float, faces: Optional[Iterable[str]] = None) -> "BoundingBox":
        if faces is None:
            faces = {"+x", "-x", "+y", "-y", "+z", "-z"}
        else:
            faces = set(faces)
        origin = list(self.origin)
        size = list(self.size)
        axis_index = {"x": 0, "y": 1, "z": 2}
        for axis, idx in axis_index.items():
            if f"-{axis}" in faces:
                origin[idx] -= thickness
                size[idx] += thickness
            if f"+{axis}" in faces:
                size[idx] += thickness
        expanded_origin = _ensure_vec3(origin)
        expanded_size = _ensure_vec3(size)
        return BoundingBox(expanded_origin, expanded_size)


@dataclass
class ElementRecord:
    element: GeometryElement
    dimtags: List[Tuple[int, int]]
    material: Optional[str]
    bbox: Optional[BoundingBox] = None


@dataclass
class GeometrySpec:
    """Full geometry description with element list and metadata."""

    version: int = 1
    units: str = "m"
    name: str = "thermal_twin_geometry"
    elements: list[GeometryElement] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.units = self.units.lower()
        if self.units not in UNIT_SCALES:
            raise ValueError(f"Unsupported geometry unit '{self.units}'")

    @property
    def unit_scale(self) -> float:
        return UNIT_SCALES[self.units]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeometrySpec":
        elements: list[GeometryElement] = []
        for entry in data.get("elements", []):
            params = entry.get("params")
            if params is None:
                params = {k: v for k, v in entry.items() if k not in {"type", "name"}}
            elements.append(
                GeometryElement(
                    type=entry["type"],
                    name=entry["name"],
                    params=params,
                )
            )
        metadata = {k: v for k, v in data.items() if k not in {"version", "units", "name", "elements"}}
        return cls(
            version=data.get("version", 1),
            units=data.get("units", "m"),
            name=data.get("name", "thermal_twin_geometry"),
            elements=elements,
            metadata=metadata,
        )

    @classmethod
    def from_file(cls, path: Path) -> "GeometrySpec":
        import json

        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class BuildContext:
    """Holds gmsh OCC context and bookkeeping during construction."""

    spec: GeometrySpec
    occ: Any
    unit_scale: float
    elements: dict[str, ElementRecord] = field(default_factory=dict)
    bounding_boxes: dict[str, BoundingBox] = field(default_factory=dict)
    volumes_by_material: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))

    def register_element(
        self,
        element: GeometryElement,
        dimtags: List[Tuple[int, int]],
        bbox: Optional[BoundingBox] = None,
    ) -> None:
        record = ElementRecord(element=element, dimtags=dimtags, material=element.material, bbox=bbox)
        self.elements[element.name] = record
        if bbox:
            self.bounding_boxes[element.name] = bbox
        if element.material:
            self.volumes_by_material[element.material].extend(tag for dim, tag in dimtags if dim == 3)

    def get_bbox(self, name: str) -> BoundingBox:
        if name not in self.bounding_boxes:
            raise KeyError(f"Element '{name}' has no recorded bounding box")
        return self.bounding_boxes[name]

    def rebuild_material_map(self) -> None:
        self.volumes_by_material.clear()
        for record in self.elements.values():
            if record.material:
                self.volumes_by_material.setdefault(record.material, [])
                self.volumes_by_material[record.material].extend(
                    tag for dim, tag in record.dimtags if dim == 3
                )

def _vector(params: dict[str, Any], key: str, fallback: Optional[Iterable[float]] = None) -> List[float]:
    value = params.get(key)
    if value is None:
        if fallback is None:
            raise ValueError(f"Geometry element missing '{key}' parameter")
        value = fallback
    floats = [float(v) for v in value]
    if len(floats) != 3:
        raise ValueError(f"Parameter '{key}' must have length 3")
    return floats


def _scale_tuple(values: Iterable[float], scale: float) -> Vec3:
    vals = tuple(float(v) * scale for v in values)
    if len(vals) != 3:
        raise ValueError("Scaled tuple must have length 3")
    return cast(Vec3, vals)


def _axis_direction(axis: Any) -> Tuple[float, float, float]:
    if isinstance(axis, str):
        axis = axis.lower()
        if axis == "x":
            return (1.0, 0.0, 0.0)
        if axis == "y":
            return (0.0, 1.0, 0.0)
        if axis == "z":
            return (0.0, 0.0, 1.0)
        raise ValueError(f"Unsupported axis '{axis}'")
    if isinstance(axis, (list, tuple)):
        if len(axis) != 3:
            raise ValueError("Axis vector must have length 3")
        length = math.sqrt(sum(float(v) ** 2 for v in axis))
        if length == 0:
            raise ValueError("Axis vector must be non-zero")
        normalized = _ensure_vec3(float(v) / length for v in axis)
        return normalized
    raise ValueError("Axis must be string or length-3 vector")


def _resolve_block_origin(params: dict[str, Any], size: List[float]) -> List[float]:
    anchor = params.get("anchor")
    if anchor == "center" or ("center" in params and anchor is None):
        center = _vector(params, "center", fallback=params.get("origin", [0.0, 0.0, 0.0]))
        return [center[i] - size[i] / 2 for i in range(3)]
    if anchor == "bottom_center":
        base_center = _vector(params, "center", fallback=params.get("origin", [0.0, 0.0, 0.0]))
        return [
            base_center[0] - size[0] / 2,
            base_center[1] - size[1] / 2,
            base_center[2],
        ]
    origin = params.get("origin")
    if origin is None:
        return [-size[0] / 2, -size[1] / 2, -size[2] / 2]
    return [float(v) for v in origin]


def _collect_points_from_dimtags(dimtags: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    surfaces: List[Tuple[int, int]] = []
    curves: List[Tuple[int, int]] = []
    points: List[Tuple[int, int]] = []
    for dimtag in dimtags:
        surfaces.extend(gmsh.model.getBoundary([dimtag], oriented=False, combined=False))
    for surface in surfaces:
        curves.extend(gmsh.model.getBoundary([surface], oriented=False, combined=False))
    for curve in curves:
        points.extend(gmsh.model.getBoundary([curve], oriented=False, combined=False))
    unique = {(dim, tag) for dim, tag in points if dim == 0}
    return list(unique)


@GeometryElement.register("block")
def _build_block(element: GeometryElement, ctx: BuildContext) -> List[Tuple[int, int]]:
    params = element.params
    size = _vector(params, "size")
    origin = _resolve_block_origin(params, size)
    origin_m = _scale_tuple(origin, ctx.unit_scale)
    size_m = _scale_tuple(size, ctx.unit_scale)
    tag = gmsh.model.occ.addBox(*origin_m, *size_m)
    bbox = BoundingBox(_ensure_vec3(origin), _ensure_vec3(size))
    dimtags = [(3, tag)]
    ctx.register_element(element, dimtags, bbox=bbox)
    return dimtags


@GeometryElement.register("cylinder")
def _build_cylinder(element: GeometryElement, ctx: BuildContext) -> List[Tuple[int, int]]:
    params = element.params
    radius = float(params["radius"])
    height = float(params["height"])
    center = _vector(params, "center", fallback=params.get("origin", [0.0, 0.0, 0.0]))
    axis_param = params.get("axis", "z")
    direction = _axis_direction(axis_param)
    base = [center[i] - 0.5 * height * direction[i] for i in range(3)]
    base_m = _scale_tuple(base, ctx.unit_scale)
    axis_vec_m = _scale_tuple([direction[i] * height for i in range(3)], ctx.unit_scale)
    radius_m = radius * ctx.unit_scale
    tag = gmsh.model.occ.addCylinder(*base_m, *axis_vec_m, radius_m)
    if isinstance(axis_param, str) and axis_param.lower() in {"x", "y", "z"}:
        idx = {"x": 0, "y": 1, "z": 2}[axis_param.lower()]
        half = [radius, radius, radius]
        half[idx] = height / 2
    else:
        half = [max(radius, height / 2.0)] * 3
    bbox_origin = _ensure_vec3(center[i] - half[i] for i in range(3))
    bbox_size = _ensure_vec3(2 * half[i] for i in range(3))
    bbox = BoundingBox(origin=bbox_origin, size=bbox_size)
    dimtags = [(3, tag)]
    ctx.register_element(element, dimtags, bbox=bbox)
    return dimtags


@GeometryElement.register("shell")
def _build_shell(element: GeometryElement, ctx: BuildContext) -> List[Tuple[int, int]]:
    params = element.params
    target_name = params.get("target")
    if not target_name:
        raise ValueError("Shell element requires 'target' parameter")
    thickness = float(params["thickness"])
    gap = float(params.get("gap", 0.0))
    if gap < 0:
        raise ValueError("Shell gap must be non-negative")
    faces = params.get("apply_faces")
    target_bbox = ctx.get_bbox(target_name)
    outer_bbox = target_bbox.expand(thickness + gap, faces)
    outer_origin_m = _scale_tuple(outer_bbox.origin, ctx.unit_scale)
    outer_size_m = _scale_tuple(outer_bbox.size, ctx.unit_scale)
    outer_tag = gmsh.model.occ.addBox(*outer_origin_m, *outer_size_m)
    dimtags = [(3, outer_tag)]
    ctx.register_element(element, dimtags, bbox=outer_bbox)
    return dimtags


def _ensure_vec3(values: Iterable[float]) -> Vec3:
    seq = tuple(float(v) for v in values)
    if len(seq) != 3:
        raise ValueError("Expected exactly 3 values for Vec3")
    return cast(Vec3, seq)



__all__ = [
    "GeometryElement",
    "GeometrySpec",
    "BuildContext",
    "ElementRecord",
    "BoundingBox",
    "_collect_points_from_dimtags",
]
