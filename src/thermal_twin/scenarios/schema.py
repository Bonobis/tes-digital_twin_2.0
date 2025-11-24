"""Pydantic models describing scenario configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


def _default_mesh_controls() -> "MeshControls":
    return MeshControls()  # pyright: ignore[reportCallIssue]


class MeshControls(BaseModel):
    global_size_mm: float = Field(5.0, ge=0.1)
    rod_refinement_mm: Optional[float] = Field(None, ge=0.05)


class HeaterSchedule(BaseModel):
    name: str
    max_temperature_c: float
    ramp_minutes: float = 30.0
    ramp_seconds: Optional[float] = None
    turn_off_seconds: Optional[float] = None
    turn_off_minutes: Optional[float] = None
    turn_off_hours: Optional[float] = None

    @property
    def ramp_s(self) -> float:
        if self.ramp_seconds is not None:
            return float(self.ramp_seconds)
        return max(0.0, float(self.ramp_minutes) * 60.0)

    @property
    def turn_off_s(self) -> Optional[float]:
        if self.turn_off_seconds is not None:
            return float(self.turn_off_seconds)
        if self.turn_off_minutes is not None:
            return float(self.turn_off_minutes) * 60.0
        if self.turn_off_hours is not None:
            return float(self.turn_off_hours) * 3600.0
        return None


class BoundaryCondition(BaseModel):
    region: str
    type: str
    value: float
    h_coeff: Optional[float] = None


class MeasurementProbe(BaseModel):
    name: str
    position: tuple[float, float, float]
    units: str | None = None


class ScenarioConfig(BaseModel):
    version: int = 1
    name: str
    description: str | None = None
    geometry_file: Path
    materials_file: Path = Path("materials.json")
    mesh: MeshControls = Field(default_factory=_default_mesh_controls)
    heater: HeaterSchedule
    boundaries: list[BoundaryCondition] = Field(default_factory=list)
    measurements: list[MeasurementProbe] = Field(default_factory=list)
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
