"""Pydantic models describing scenario configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class MeshControls(BaseModel):
    global_size_mm: float = Field(5.0, ge=0.1)
    rod_refinement_mm: Optional[float] = Field(None, ge=0.05)


class HeaterSchedule(BaseModel):
    name: str
    max_temperature_c: float
    ramp_minutes: float = 30.0


class BoundaryCondition(BaseModel):
    region: str
    type: str
    value: float
    h_coeff: Optional[float] = None


class ScenarioConfig(BaseModel):
    version: int = 1
    name: str
    description: str | None = None
    geometry_file: Path
    materials_file: Path = Path("materials.json")
    mesh: MeshControls = MeshControls()
    heater: HeaterSchedule
    boundaries: list[BoundaryCondition] = Field(default_factory=list)
    total_time_s: float = 7200.0
    dt_s: float = 20.0

    @field_validator("geometry_file", "materials_file", mode="after")
    @classmethod
    def _expand_path(cls, value: Path) -> Path:
        return value.expanduser()

    @classmethod
    def from_file(cls, path: Path) -> "ScenarioConfig":
        import yaml

        data = yaml.safe_load(Path(path).read_text())
        base = path.parent
        if "geometry_file" in data and not Path(data["geometry_file"]).is_absolute():
            data["geometry_file"] = base / data["geometry_file"]
        if "materials_file" in data and not Path(data["materials_file"]).is_absolute():
            data["materials_file"] = base / data["materials_file"]
        return cls(**data)
