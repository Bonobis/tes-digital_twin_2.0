"""Geometry element abstractions for modular design."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass
class GeometryElement:
    """Base element definition consumed by gmsh builders."""

    type: str
    name: str
    params: dict[str, Any]

    registry: ClassVar[dict[str, type["GeometryElement"]]] = {}

    def as_gmsh(self) -> dict[str, Any]:
        """Return a gmsh-friendly dictionary representation."""
        return {"type": self.type, "name": self.name, **self.params}

    @classmethod
    def register(cls, element_type: str):
        def decorator(subclass: type["GeometryElement"]):
            cls.registry[element_type] = subclass
            return subclass

        return decorator


@dataclass
class GeometrySpec:
    """Full geometry description with element list and metadata."""

    version: int
    elements: list[GeometryElement] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeometrySpec":
        elements = [GeometryElement(**entry) for entry in data.get("elements", [])]
        metadata = {k: v for k, v in data.items() if k not in {"version", "elements"}}
        return cls(version=data.get("version", 1), elements=elements, metadata=metadata)
