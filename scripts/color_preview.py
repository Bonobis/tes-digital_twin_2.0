r"""Preview mesh with custom colors per region.

Usage:
  .\.venv\Scripts\python scripts\color_preview.py outputs/meshes/4_blocks_naked.vtu --png preview.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict

import meshio
import numpy as np
import pyvista as pv

DEFAULT_COLORS: Dict[str, str] = {
    "geopolymer_core": "#d97706",   # orange
    "heater_bedding": "#8b5a2b",    # brownish
    "heater_rod": "#6b7280",        # gray
    "gap1_air": "#38bdf8",          # light blue
    "gap2_air": "#0ea5e9",          # blue
    "gap3_air": "#0284c7",          # darker blue,
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
    grid = pv.UnstructuredGrid(cell_conn, celltypes, mesh.points)
    return grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path, help="Path to VTU/VTK/mesh file")
    parser.add_argument("--png", type=Path, help="Optional PNG output")
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

    pl = pv.Plotter()
    for rid in unique_ids:
        mask = grid.threshold(value=(rid - 0.5, rid + 0.5), scalars="region_id", invert=False)
        name = tag_to_name.get(rid, f"id_{rid}")
        color = lut[rid]
        pl.add_mesh(
            mask,
            color=color,
            show_scalar_bar=False,
            opacity=0.35,
            label=name,
            edge_color="black",
            show_edges=False,
        )
    pl.add_legend()
    pl.add_axes()
    pl.show_bounds(grid="front")

    if args.png:
        pl.show(screenshot=str(args.png))
    else:
        pl.show()


if __name__ == "__main__":
    main()
