# Abstract (Draft)
We present an open-source digital twin for transient heat diffusion in composite heater–insulation assemblies, using FiPy finite volumes on gmsh-generated unstructured tetrahedral meshes. The framework couples Dirichlet heater schedules with Robin (convection + linearized radiation) boundaries, temperature-dependent material properties, and configurable low-conductivity contact gaps to emulate imperfect assembly. Experimental calibration targets include single- and double-insulated geopolymer blocks (400–700 °C ramps) and a 1.2 m bare multi-block stack. Telemetry is streamed during simulation (NDJSON) and post-processed for RMSE-based validation against measured probe data. The approach enables rapid parametric studies (heater efficiency, bedding conductivity, gap thickness, convective losses) and is intended to guide design and scale-up of field-deployable heated structures. [CITATION]

# Introduction (Draft outline)
- **Context and motivation**: Thermal management of heated composite blocks; need for predictive digital twin to reduce experimental iteration. [CITATION]
- **Related work**: Heat diffusion modeling in porous media, insulation performance, contact resistance effects, Dirichlet vs. power-driven heater models, FiPy/gmsh-based solvers. [CITATION]
- **Objective**: Deliver an extensible, validated simulation pipeline that matches experimental probe data (400–700 °C ramps) across geometries (single-layer, double-layer insulation, bare multi-block stack) and supports calibration of losses and heater efficiency.
- **Contributions**:
  - Modular scenario-driven solver (FiPy + gmsh) with Dirichlet heater ramps, Robin losses, temperature-dependent properties.
  - Explicit low-k gap modeling for imperfect block contacts; optional heater efficiency to capture unmodeled losses.
  - Validation workflow with streamed telemetry and RMSE metrics against lab datasets.
  - Open-source artifacts (meshes, scenarios, plotters) ready for replication and extension.
- **Paper structure**: Methods (geometry, materials, BCs, numerics), validation against experiments, sensitivity studies (h, bedding k, gap k, heater efficiency), discussion, conclusions and future work.
