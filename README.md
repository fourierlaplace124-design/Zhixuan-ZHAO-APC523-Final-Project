# Project Code Overview

## Code Architecture

The submission contains a finite-difference workflow for the backward-facing step flow problem.

- `mesh_generator.py`: Builds the structured Cartesian mesh, marks fluid/solid/inlet/outlet/wall regions, and stores velocity and pressure fields.
- `solver.py`: Implements the finite-difference incompressible Navier-Stokes solver using an explicit time step and projection method.
- `batch_experiment.py`: Sets up the backward-facing step geometry, grid sizes, Reynolds numbers, solver parameters, and batch execution.
- `mms_verification.py`: Runs the manufactured-solution check for the spatial discretization.
- `run_smooth_region_convergence.py`: Runs the smooth-region convergence analysis.
- `run_reynolds_corner_density.py`: Runs the Reynolds-number error-localization analysis.
- `demo_corner_rational_enrichment.py`: Runs the local corner enrichment demonstration.
- `plot_stability_region.py`: Generates the stability-region plots.
- `figures/`: Contains the figures used by the report and presentation.
- `final_report_academic.pdf`: Final written report.

## Environment

Required Python version:

```bash
Python 3.7+
```

Required packages:

```bash
numpy
scipy
matplotlib
```
