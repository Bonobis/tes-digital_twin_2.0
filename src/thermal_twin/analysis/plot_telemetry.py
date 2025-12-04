"""Quick telemetry reader/plotter for NDJSON outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

def load_steps(path: Path) -> Dict[str, List[float]]:
    """Load probe data and time from an NDJSON telemetry file."""
    times: List[float] = []
    probes: Dict[str, List[float]] = {}
    with path.open() as fh:
        for line in fh:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "step":
                continue
            t = float(event.get("time", 0.0))
            times.append(t)
            for name, temp in (event.get("probes") or {}).items():
                probes.setdefault(name, []).append(float(temp))
    probes["__time__"] = times
    return probes

def plot_file(path: Path, save: Path | None = None) -> None:
    data = load_steps(path)
    times = data.pop("__time__", [])
    if not times:
        print(f"No step data found in {path}")
        return
    plt.figure()
    for name, series in data.items():
        if len(series) != len(times):
            series = series[-len(times):]
        plt.plot(times, series, label=name)
    plt.xlabel("time (s)")
    plt.ylabel("temperature (degC)")
    plt.legend()
    plt.title(path.name)
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(save)
        print(f"Saved plot to {save}")
    else:
        plt.show()

def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Plot telemetry NDJSON file(s).")
    parser.add_argument("files", nargs="+", type=Path, help="Telemetry NDJSON paths.")
    parser.add_argument("--save", type=Path, help="Optional output image path (per file adds stem).")
    args = parser.parse_args()
    for ndjson_path in args.files:
        out = None
        if args.save:
            out = args.save
            if out.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                out = out.with_name(f"{ndjson_path.stem}{out.suffix}")
        plot_file(ndjson_path, out)

if __name__ == "__main__":  # pragma: no cover
    main()
