r"""Render a VTU/VTK mesh with stable colors per physical region.

This is meant for quickly generating consistent-looking figures from gmsh/meshio
exports, since default renderers can assign arbitrary colors per block.

Usage:
  .\.venv\Scripts\python scripts\color_preview.py outputs/meshes/4_blocks_naked.vtu --png publication/figures/4_blocks_naked_mesh.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import meshio
import numpy as np
import pyvista as pv

# Color by physical-group name (typically material names for tetra groups)
DEFAULT_COLORS: Dict[str, str] = {
    # Materials
    "dias_geopolymer": "#d97706",        # orange
    "geopolymer_uncured": "#facc15",    # bright yellow
    "brick_powder": "#8b5a2b",          # brown
    "nichrome": "#6b7280",              # gray
    "stainless_steel": "#9ca3af",       # light gray
    "silica_blanket": "#34d399",        # green
    "rockwool": "#22c55e",              # green
    "air_gap": "#38bdf8",               # light blue

    # Element names (fallback if tetra physical groups are element-tagged)
    "geopolymer_core": "#d97706",
    "heater_bedding": "#8b5a2b",
    "heater_rod": "#6b7280",
    "outer_shell": "#34d399",
    "rockwool_shell": "#22c55e",
}


def build_tag_name_map(mesh: meshio.Mesh) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for name, (idx, _dim) in mesh.field_data.items():
        mapping[int(idx)] = name
    return mapping


def meshio_to_pv(mesh: meshio.Mesh) -> pv.UnstructuredGrid:
    if "tetra" not in mesh.cells_dict:
        raise SystemExit("No tetra cells found in mesh")
    cells = mesh.cells_dict["tetra"]
    n_cells, npts = cells.shape
    if npts != 4:
        raise SystemExit("Expected tetra connectivity")
    # Build VTK-style cells array: [4, id0, id1, id2, id3, 4, ...]
    cell_conn = np.hstack([np.full((n_cells, 1), 4, dtype=np.int64), cells.astype(np.int64)]).ravel()
    celltypes = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
    return pv.UnstructuredGrid(cell_conn, celltypes, mesh.points)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path, help="Path to VTU/VTK/mesh file")
    parser.add_argument("--png", type=Path, help="Optional PNG output")
    parser.add_argument("--opacity", type=float, default=0.35, help="Mesh opacity (0-1)")
    args = parser.parse_args()

    m = meshio.read(args.mesh)
    tag_to_name = build_tag_name_map(m)
    phys = m.cell_data_dict.get("gmsh:physical", {}).get("tetra")
    if phys is None:
        raise SystemExit("gmsh:physical cell data not found")
    phys_arr = np.asarray(phys).reshape(-1)

    grid = meshio_to_pv(m)
    grid.cell_data["region_id"] = phys_arr

    unique_ids = sorted(set(int(x) for x in phys_arr))
    palette = ["#ef4444", "#3b82f6", "#22c55e", "#a855f7", "#f59e0b", "#06b6d4", "#94a3b8"]
    lut: Dict[int, str] = {}
    for idx, rid in enumerate(unique_ids):
        name = tag_to_name.get(rid, f"id_{rid}")
        color = DEFAULT_COLORS.get(name)
        if color is None:
            color = palette[idx % len(palette)]
        lut[rid] = color

    off_screen = args.png is not None
    pl = pv.Plotter(off_screen=off_screen, window_size=(1800, 1000))
    for rid in unique_ids:
        mask = grid.threshold(value=(rid - 0.5, rid + 0.5), scalars="region_id", invert=False)
        name = tag_to_name.get(rid, f"id_{rid}")
        pl.add_mesh(
            mask,
            color=lut[rid],
            show_scalar_bar=False,
            opacity=float(args.opacity),
            label=name,
            show_edges=False,
        )
    pl.add_legend(bcolor="white")
    pl.add_axes()
    pl.show_bounds(grid="front")

    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        pl.show(screenshot=str(args.png))
        print(f"Saved {args.png}")
    else:
        pl.show()


if __name__ == "__main__":
    main()
