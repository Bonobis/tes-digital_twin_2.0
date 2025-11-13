"""FiPy solver placeholder with configuration plumbing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SolverSettings:
    total_time_s: float = 3600.0
    dt: float = 20.0
    save_interval: float = 60.0


class HeatSolver:
    """Wraps FiPy operations (placeholder for now)."""

    def __init__(self, mesh: Any, settings: SolverSettings):
        self.mesh = mesh
        self.settings = settings

    def run(self) -> dict[str, Any]:
        steps = int(self.settings.total_time_s / self.settings.dt)
        return {"steps": steps, "dt": self.settings.dt}
