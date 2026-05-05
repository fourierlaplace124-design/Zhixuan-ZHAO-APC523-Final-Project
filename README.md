# Final Project
## Code Architecture

- `mesh_generator.py`: Builds the structured mesh and marks boundary/solid regions.
- `solver.py`: Finite-difference incompressible Navier-Stokes solver.
- `batch_experiment.py`: Shared setup for the backward-facing step experiments.
- `mms_verification.py`: Manufactured-solution verification.
- `plot_stability_region.py`: Stability-region plots.
- `run_smooth_region_convergence.py`: Smooth-region convergence analysis.
- `run_reynolds_corner_density.py`: Reynolds-number error-localization analysis.
- `demo_corner_rational_enrichment.py`: Corner enrichment analysis.
- `run_adroit.sh`: Bash entry point for running the programs.
- `adroit_job.slurm`: Optional Slurm job template.

## Environment

```bash
module purge
module load anaconda3/2025.6
conda create -y -n apc523-fdm -c conda-forge python=3.11 numpy scipy matplotlib
conda activate apc523-fdm
```

## Clone

```bash
git clone <repository-url>
cd <repository-name>/Submit
```

## Run Each Program

```bash
bash run_adroit.sh compile
bash run_adroit.sh mms
bash run_adroit.sh stability
bash run_adroit.sh smooth
bash run_adroit.sh reynolds
bash run_adroit.sh enrichment
bash run_adroit.sh full
```

Run `reynolds` before `enrichment` on a fresh clone.
