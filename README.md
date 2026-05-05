# Final Project - Program Files

## Project Information
- **Title**: Accuracy and Error Localization in a Finite Difference Solver for Backward-Facing Step Flow
- **Author**: Zhixuan Zhao (zz0487@princeton.edu)
- **Course**: APC523 Scientific Computing, Princeton University

## File List

This folder contains the minimal core program set required to reproduce the paper results.

### 1. Core Solver (3 files)

**solver.py** (24 KB)
- Finite difference Navier-Stokes solver
- Implements Chorin's projection method (fractional step method)
- Supports central difference and upwind schemes
- Explicit Euler time integration
- Includes convergence monitoring and drag coefficient calculation

**mesh_generator.py** (9.1 KB)
- Structured Cartesian mesh generator
- Supports backward-facing step geometry
- Mesh visualization functionality

**batch_experiment.py** (18 KB)
- Batch experiment manager
- Automated multi-case execution
- Grid refinement and Reynolds number sweeps

### 2. Verification Program (1 file)

**mms_verification.py** (6.6 KB)
- Method of Manufactured Solutions (MMS) verification
- Confirms second-order accuracy of discrete operators on smooth fields
- Generates convergence rate data for Table 1

### 3. Main Analysis Programs (4 files)

**run_smooth_region_convergence.py** (19 KB)
- Smooth-region grid convergence analysis
- Excludes corner and boundary effects
- Generates Figures 4 and 5
- Verifies interior second-order accuracy (p≈1.83)

**run_reynolds_corner_density.py** (9.9 KB)
- Reynolds number vs. corner error density analysis
- Regional error decomposition (full domain, smooth region, corner region)
- Generates Figure 6
- Studies error localization variation with Re (Re=8 to 200)

**demo_corner_rational_enrichment.py** (8.6 KB)
- Corner rational function enrichment demonstration
- Local singularity fitting inspired by Lightning Stokes
- Generates Table 3 and Figure 7
- Proof-of-concept: 35-87% error reduction

**plot_stability_region.py** (8.5 KB)
- Stability region visualization
- Generates Figures 2 and 3
- Complex plane stability diagram
- CFL-Fourier plane stability boundaries

### 4. Report

**final_report_academic.pdf** (1.1 MB)
- Complete academic report
- Contains all methods, results, and analysis

## Usage

### Environment Requirements
```bash
Python 3.7+
numpy
scipy
matplotlib
```

### Execution Order

1. **MMS Verification** (generates Table 1)
```bash
python mms_verification.py
```

2. **Smooth Region Convergence** (generates Figures 4, 5)
```bash
python run_smooth_region_convergence.py
```

3. **Reynolds Number Sweep** (generates Figure 6)
```bash
python run_reynolds_corner_density.py
```

4. **Corner Enrichment** (generates Table 3, Figure 7)
```bash
python demo_corner_rational_enrichment.py
```

5. **Stability Analysis** (generates Figures 2, 3)
```bash
python plot_stability_region.py
```

## Paper Results Correspondence

| Paper Content | Program File | Output |
|--------------|-------------|--------|
| Table 1: MMS convergence rates | mms_verification.py | Second-order accuracy verification |
| Table 2: Re=20 grid refinement | run_smooth_region_convergence.py | Global convergence rates |
| Table 3: Corner enrichment | demo_corner_rational_enrichment.py | Error reduction ratios |
| Figure 2: Complex plane stability | plot_stability_region.py | Stability region |
| Figure 3: CFL-Fo stability | plot_stability_region.py | Time step constraints |
| Figure 4: Grid error trends | run_smooth_region_convergence.py | Convergence curves |
| Figure 5: Smooth region convergence | run_smooth_region_convergence.py | p≈1.83 |
| Figure 6: Re-error density | run_reynolds_corner_density.py | Error localization |
| Figure 7: Corner error field | demo_corner_rational_enrichment.py | Enrichment effect |

## Main Findings

1. **MMS Verification**: Confirms second-order accuracy on smooth fields (p=2.016-2.043)
2. **Global Convergence**: At Re=20, global convergence rate p=1.31-1.64, degraded by corner singularity
3. **Smooth Region**: Excluding corner recovers convergence rate to p=1.83, approaching theoretical value
4. **Error Localization**: 
   - Re≤40: Corner error density lower than smooth region
   - Re≥80: Corner becomes high-density error region
   - Re=100-200: Strongest error localization
5. **Rational Function Enrichment**: Local corner error reduced by 35-87%

## Technical Details

### Numerical Method
- **Spatial discretization**: Central difference (Re=20) or upwind (Re=2000)
- **Time integration**: Explicit Euler
- **Pressure solver**: Projection method (Chorin's fractional step)
- **Boundary conditions**: Fixed velocity inlet, zero-gradient outlet, no-slip walls

### Stability Constraints
- Re=20: Δt ~ h² (diffusion-limited)
- Re=2000: Δt ~ h (convection-limited)

### Grid Setup
- 7 systematically refined grids: 80×32 to 320×128
- Backward-facing step geometry: expansion ratio 1:2
- Domain size: 10h × 4h

## File Statistics

- Core solver: 3 files
- Verification program: 1 file
- Analysis programs: 4 files
- Report: 1 PDF
- **Total: 9 files, 1.3 MB**

## Citation

If using this code, please cite:
```
Zhao, Z. (2026). Accuracy and Error Localization in a Finite 
Difference Solver for Backward-Facing Step Flow. APC523 Final 
Project, Princeton University.
```

## Contact

Zhixuan Zhao  
Email: zz0487@princeton.edu  
Princeton University

Last updated: May 4, 2026
