# Digital Twin Thermal Simulator: Methodology and Assumptions

> Work-in-progress documentation for journal submission (Energies). Citation placeholders marked as `[CITATION]`.

## System Overview
- **Physics**: Transient heat conduction in heterogeneous solids with Dirichlet heater prescription, Robin (convection + linearized radiation) at exposed faces, optional internal low-k contact gaps.
- **Numerics**: Finite volume (FiPy) on unstructured tetrahedral meshes generated via gmsh; meshio for format conversion; Python tooling for scenarios and telemetry.
- **Geometries**:
  - Baseline single-layer: 10×10×25 cm geopolymer core, heater rod (r=0.8 cm), brick-dust bedding (r=1.0 cm), silica blanket shell (t=1.3 cm).
  - Double-layer (2L): same core, shell thickness 2.6 cm.
  - Four-block “naked” stack: 10×10×120 cm core, heater r=1.25 cm, bedding r=1.3 cm, three 2 mm low-k gaps.
  - Coordinates origin at (0,0,0); heater centered on core mid-plane unless otherwise noted.

## Governing Equations
- **Heat equation (per material region)**  
  \( \rho c_p \partial T/\partial t = \nabla\cdot(k\nabla T) + q \) where \(q=0\) (heater imposed via Dirichlet).

- **Dirichlet heater constraint**  
  \[
    T_h(t) = \begin{cases}
      T_0 + (T_{\max}-T_0)\,t/t_{\text{ramp}}, & t < t_{\text{ramp}}\\
      T_{\max}, & t \ge t_{\text{ramp}}
    \end{cases}
  \]
  Optional efficiency \(\eta_h \in (0,1]\): \(T_{\max,\text{eff}} = \eta_h T_{\max}\).

- **Robin boundary (convection + linearized radiation)** on surface \(\Gamma\):  
  \( -k_{\text{face}}\,\mathbf{n}\cdot\nabla T = h (T - T_\infty) + h_{\text{rad}} (T - T_\infty) \)  
  with \( h \) [W/m²·K], ambient \( T_\infty \), emissivity \( \varepsilon \), Stefan–Boltzmann \( \sigma \):  
  \( h_{\text{rad}} = 4\sigma\varepsilon T_\infty^3 \)  
  Effective total \( h_{\text{eff}} = h + h_{\text{rad}} \). Face conductivity \( k_{\text{face}} \) from adjacent cell material.

- **Contact gaps**: thin low-k solid layers (`air_gap`) giving added conduction resistance.

## Material Properties (current catalog)
- **Geopolymer**: \( \rho \approx 1960 \) kg/m³; \( c_p \approx 950–1050 \) J/kg·K; \( \alpha \approx 4.4–4.7\times10^{-7} \) m²/s.
- **Nichrome**: \( k \approx 14–16.5 \) W/m·K; \( c_p \approx 440–500 \) J/kg·K; \( \rho = 8400 \) kg/m³.
- **Silica blanket**: \( k = 0.04 \to 0.40 \) W/m·K (260–1200 °C); \( \rho = 128 \) kg/m³; \( c_p = 900–1150 \) J/kg·K.
- **Brick powder (bedding)**: tuned low, \( k = [0.28, 0.27, 0.26, 0.25, 0.24] \) W/m·K over 20–700 °C; \( \rho \approx 1560–1620 \) kg/m³; \( c_p \approx 880–990 \) J/kg·K (accounts for imperfect compaction/air).
- **Air gap (contact)**: \( k = [0.026, 0.030, 0.034, 0.038, 0.042] \) W/m·K over 20–700 °C; \( \rho = 1.2 \) kg/m³; \( c_p = 1000 \) J/kg·K.
- Emissivity default \( \varepsilon = 0.9 \).

## Boundary Conditions Used
- Baseline/2L: convection on `outer_shell` with \( h=17 \) W/m²·K; +z insertion face \( h=23 \) W/m²·K; \( T_\infty \approx 26\,°C \); \( \varepsilon=0.9 \).
- Naked 4-block: convection on `geopolymer_core`; \( h=18 \) W/m²·K; \( \varepsilon=0.9 \).

## Meshing and Quality
- gmsh tetra mesh; global size ~10 mm; rod refinement 2 mm; gaps 2 mm.
- Quality metric (radius ratio): accept min ≥0.2 (observed ~0.20–0.39), mean ~0.80–0.86. Cell distances floored to 1e-6 for FiPy stability.
- Exports: `.msh` (v2.2), `.vtu` preview.

## Solver Workflow
1. Parse scenario YAML (geometry, mesh controls, heater schedule, BCs, probes, time controls).
2. Build gmsh model, fragment, tag physical groups, mesh.
3. Convert to FiPy mesh; regularize distances.
4. Interpolate material fields (k, cp, rho) per cell.
5. Impose heater Dirichlet ramp; advance transient (implicit).
6. Apply Robin BCs with \( h_{\text{eff}} = h + 4\sigma\varepsilon T_\infty^3 \).
7. Stream telemetry (NDJSON), optional live plot; write summary JSON.

## Assumptions / Limitations
- Heater as uniform Dirichlet (no electrical/coil geometry); efficiency factor \( \eta_h \) can approximate extra losses.
- Contact resistance handled via explicit low-k gaps (no TCR model).
- Radiation linearized around ambient; nonlinear form not yet used at very high T.
- No phase change/dehydration; no fluid flow; convection coefficients uniform.
- Probes are point samples; no sensor lag/mass.

## Planned/Optional Extensions
- Nonlinear radiation BC; heater efficiency parameter in scenarios; true contact conductance model; data-driven heater power; calibration/optimization vs experimental NDJSON; adaptive refinement near heater/gaps.

## Validation Plan
- Compare ramps/plateaus vs experimental datasets (1L/2L, long-stack) with RMSE per probe and timing to milestones (e.g., 100, 200 °C).
- Sensitivity scans: vary \( h \), bedding k, gap k/thickness, \( \eta_h \).
- Stability checks with smaller dt (5–10 s) on stiff cases.

## Key Formula Summary
- Heat: \( \rho c_p \partial T/\partial t = \nabla\cdot(k\nabla T) \).
- Heater: Dirichlet ramp to \( \eta_h T_{\max} \).
- Robin: \( -k\nabla T\cdot n = (h + 4\sigma\varepsilon T_\infty^3)(T - T_\infty) \).
- Radiation linearization: \( h_{\text{rad}} = 4\sigma\varepsilon T_\infty^3 \).

## Citations
[CITATION] FiPy FVM solver
[CITATION] gmsh mesh generation
[CITATION] Mesh quality metrics (radius ratio)
[CITATION] Linearized radiation BC
[CITATION] Thermal properties of geopolymer/silica blanket/nichrome/air
[CITATION] Contact resistance in masonry/porous media
