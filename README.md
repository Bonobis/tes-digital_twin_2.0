# TES Digital Twin 2.0

Finite element digital twin for geopolymer heater experiments. The project couples gmsh-generated 3D geometries with FiPy heat diffusion solves, modular materials/heater catalogs, and future UI/ML tooling.

## Key Features
- Parametric geometry elements (blocks, heaters, insulation shells, future fluids) defined via JSON catalogs
- Material/property management sourced from `materials.json` with temperature-dependent interpolation
- gmsh → meshio pipeline for unstructured tetrahedral meshes tuned per region
- FiPy-based explicit solver with configurable heaters, boundary losses, and probe outputs
- Scenario configs (YAML) that bind geometry/material selections, solver controls, and experimental validation datasets
- Dash-based UI roadmap for geometry inspection, live simulation monitoring, and result management

## Repository Layout
```
configs/             # Sample geometry/scenario definitions and UI presets
src/thermal_twin/    # Installable package
  materials/         # Material catalogs & interpolators
  geometry/          # Element abstractions and transformations
  meshing/           # gmsh builders and mesh exporters
  solver/            # FiPy wrappers, sources, boundary models
  scenarios/         # Config schemas, runners, CLI bindings
  analysis/          # Data ingestion & validation utilities
  ui/                # Dash/Panel user interfaces
materials.json       # Provided material property curves
```

## Getting Started
1. `python -m venv .venv && .venv\Scripts\Activate.ps1`
2. `pip install -e .`
3. `thermal-twin simulate configs/scenarios/baseline_single_layer.yml`

Future work will incrementally fill in each module, add rigorous tests, and connect the CLI to a live dashboard.
