from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt


PROJECT_DIR = Path(__file__).resolve().parent
HOMEWORK_DIR = PROJECT_DIR.parent
SIM_DIR = PROJECT_DIR / "FlowSimulator"
OUT_DIR = HOMEWORK_DIR / "experiment_results_reynolds_corner_density"
FIELD_DIR = OUT_DIR / "fields"

sys.path.insert(0, str(SIM_DIR))

from batch_experiment import BatchExperimentManager  # noqa: E402
from solver import NavierStokesSolver  # noqa: E402


REYNOLDS_NUMBERS = [2, 8, 20, 80, 200, 800]
COARSE_GRID = (160, 64)
REFERENCE_GRID = (240, 96)
MAX_TIME = 30.0
SPATIAL_SCHEME = "central"

STEP_CORNER = np.array([4.0, 2.0])
CORNER_RADIUS = 0.75
BOUNDARY_MARGIN = 0.30


def ensure_dirs() -> None:
    FIELD_DIR.mkdir(parents=True, exist_ok=True)


def case_name(reynolds: int, nx: int, ny: int) -> str:
    return f"Re{reynolds}_N{nx}x{ny}"


def field_path(reynolds: int, nx: int, ny: int) -> Path:
    return FIELD_DIR / f"{case_name(reynolds, nx, ny)}.npz"


def run_case(manager: BatchExperimentManager, reynolds: int, nx: int, ny: int) -> Path:
    path = field_path(reynolds, nx, ny)
    if path.exists():
        print(f"Using cached field: {path}")
        return path

    print("\n" + "=" * 72)
    print(f"Running {case_name(reynolds, nx, ny)}")
    print("=" * 72)

    mesh = manager.create_mesh(nx, ny)
    nu = manager.get_viscosity(reynolds)
    dt = manager.get_time_step(nx, ny, reynolds)

    flow_params = {
        "reynolds": reynolds,
        "viscosity": nu,
        "dt": dt,
        "inlet_velocity": manager.inlet_velocity,
    }
    solver_params = {
        "spatial_scheme": SPATIAL_SCHEME,
        "temporal_scheme": "explicit",
        "convergence_tolerance": manager.convergence_tolerance,
        "cd_convergence_tolerance": manager.cd_convergence_tolerance,
        "cd_convergence_window_time": manager.cd_convergence_window_time,
        "max_iterations": 100000,
        "check_convergence_interval": 10,
    }

    solver = NavierStokesSolver(mesh, flow_params, solver_params)
    result = solver.run_simulation(
        total_time=MAX_TIME,
        output_interval=1000,
        convergence_check=True,
    )

    metadata = {
        "case_name": case_name(reynolds, nx, ny),
        "reynolds": reynolds,
        "nx": nx,
        "ny": ny,
        "dx": mesh.dx,
        "dy": mesh.dy,
        "h": min(mesh.dx, mesh.dy),
        "nu": nu,
        "dt": dt,
        "max_time": MAX_TIME,
        "spatial_scheme": SPATIAL_SCHEME,
        **result,
    }

    np.savez_compressed(
        path,
        x=mesh.x,
        y=mesh.y,
        cell_type=mesh.cell_type,
        u=mesh.u,
        v=mesh.v,
        p=mesh.p,
        metadata=json.dumps(metadata),
    )
    print(f"Saved field: {path}")
    return path


def load_field(path: Path) -> dict:
    with np.load(path) as data:
        field = {
            "x": data["x"],
            "y": data["y"],
            "cell_type": data["cell_type"],
            "u": data["u"],
            "v": data["v"],
            "p": data["p"],
            "metadata": json.loads(str(data["metadata"])),
        }
    field["X"], field["Y"] = np.meshgrid(field["x"], field["y"])
    return field


def make_interpolators(reference: dict) -> dict:
    axes = (reference["y"], reference["x"])
    return {
        "u": RegularGridInterpolator(axes, reference["u"], bounds_error=False, fill_value=np.nan),
        "v": RegularGridInterpolator(axes, reference["v"], bounds_error=False, fill_value=np.nan),
        "p": RegularGridInterpolator(axes, reference["p"], bounds_error=False, fill_value=np.nan),
        "cell_type": RegularGridInterpolator(
            axes,
            reference["cell_type"].astype(float),
            method="nearest",
            bounds_error=False,
            fill_value=1.0,
        ),
    }


def interpolate_reference(reference: dict, target: dict) -> dict:
    interpolators = make_interpolators(reference)
    points = np.column_stack([target["Y"].ravel(), target["X"].ravel()])
    shape = target["u"].shape
    return {
        key: interpolators[key](points).reshape(shape)
        for key in ("u", "v", "p", "cell_type")
    }


def region_masks(field: dict, ref_on_field: dict) -> dict:
    fluid = field["cell_type"] == 0
    ref_fluid = np.rint(ref_on_field["cell_type"]).astype(int) == 0
    finite_ref = np.isfinite(ref_on_field["u"]) & np.isfinite(ref_on_field["v"]) & np.isfinite(ref_on_field["p"])
    full = fluid & ref_fluid & finite_ref

    distance_to_boundary = distance_transform_edt(
        fluid,
        sampling=(field["metadata"]["dy"], field["metadata"]["dx"]),
    )
    distance_to_corner = np.sqrt(
        (field["X"] - STEP_CORNER[0]) ** 2
        + (field["Y"] - STEP_CORNER[1]) ** 2
    )

    corner = full & (distance_to_corner < CORNER_RADIUS)
    smooth = full & (distance_to_boundary >= BOUNDARY_MARGIN) & (distance_to_corner >= CORNER_RADIUS)
    boundary = full & (distance_to_boundary < BOUNDARY_MARGIN) & (distance_to_corner >= CORNER_RADIUS)
    other = full & ~(corner | smooth | boundary)
    return {"full": full, "smooth": smooth, "corner": corner, "boundary": boundary, "other": other}


def velocity_rms_error(field: dict, ref_on_field: dict, mask: np.ndarray) -> float:
    if not np.any(mask):
        return math.nan
    diff2 = (field["u"] - ref_on_field["u"]) ** 2 + (field["v"] - ref_on_field["v"]) ** 2
    return float(np.sqrt(np.mean(diff2[mask])))


def pressure_rms_error(field: dict, ref_on_field: dict, mask: np.ndarray, gauge_mask: np.ndarray) -> float:
    if not np.any(mask):
        return math.nan
    p = field["p"] - np.mean(field["p"][gauge_mask])
    pref = ref_on_field["p"] - np.mean(ref_on_field["p"][gauge_mask])
    return float(np.sqrt(np.mean((p[mask] - pref[mask]) ** 2)))


def velocity_relative_rms(field: dict, ref_on_field: dict, mask: np.ndarray) -> float:
    if not np.any(mask):
        return math.nan
    diff2 = (field["u"] - ref_on_field["u"]) ** 2 + (field["v"] - ref_on_field["v"]) ** 2
    ref2 = ref_on_field["u"] ** 2 + ref_on_field["v"] ** 2
    denom = float(np.mean(ref2[mask]))
    if denom <= 0.0:
        return math.nan
    return float(np.sqrt(np.mean(diff2[mask]) / denom))


def pressure_relative_rms(field: dict, ref_on_field: dict, mask: np.ndarray) -> float:
    if not np.any(mask):
        return math.nan
    p = field["p"][mask] - np.mean(field["p"][mask])
    pref = ref_on_field["p"][mask] - np.mean(ref_on_field["p"][mask])
    denom = float(np.mean(pref**2))
    if denom <= 0.0:
        return math.nan
    return float(np.sqrt(np.mean((p - pref) ** 2) / denom))


def analyze_reynolds(reynolds: int) -> dict:
    coarse = load_field(field_path(reynolds, *COARSE_GRID))
    reference = load_field(field_path(reynolds, *REFERENCE_GRID))
    ref_on_coarse = interpolate_reference(reference, coarse)
    masks = region_masks(coarse, ref_on_coarse)

    row = {
        "reynolds": reynolds,
        "coarse_grid": f"{COARSE_GRID[0]}x{COARSE_GRID[1]}",
        "reference_grid": f"{REFERENCE_GRID[0]}x{REFERENCE_GRID[1]}",
        "coarse_final_time": coarse["metadata"]["final_time"],
        "reference_final_time": reference["metadata"]["final_time"],
        "coarse_converged": coarse["metadata"]["converged"],
        "reference_converged": reference["metadata"]["converged"],
        "coarse_diverged": coarse["metadata"]["diverged"],
        "reference_diverged": reference["metadata"]["diverged"],
    }

    full = masks["full"]
    for region in ("full", "smooth", "corner", "boundary", "other"):
        mask = masks[region]
        row[f"n_{region}"] = int(np.sum(mask))
        row[f"velocity_rms_{region}"] = velocity_rms_error(coarse, ref_on_coarse, mask)
        row[f"pressure_rms_{region}"] = pressure_rms_error(coarse, ref_on_coarse, mask, full)
        row[f"velocity_rel_rms_{region}"] = velocity_relative_rms(coarse, ref_on_coarse, mask)
        row[f"pressure_rel_rms_{region}"] = pressure_relative_rms(coarse, ref_on_coarse, mask)

    row["velocity_corner_over_smooth"] = row["velocity_rms_corner"] / row["velocity_rms_smooth"]
    row["velocity_boundary_over_smooth"] = row["velocity_rms_boundary"] / row["velocity_rms_smooth"]
    row["pressure_corner_over_smooth"] = row["pressure_rms_corner"] / row["pressure_rms_smooth"]
    row["pressure_boundary_over_smooth"] = row["pressure_rms_boundary"] / row["pressure_rms_smooth"]
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ensure_dirs()
    manager = BatchExperimentManager(output_dir=str(OUT_DIR / "raw_solver_output"), enable_fvm_baseline=False)

    for reynolds in REYNOLDS_NUMBERS:
        for nx, ny in (COARSE_GRID, REFERENCE_GRID):
            run_case(manager, reynolds, nx, ny)

    rows = [analyze_reynolds(reynolds) for reynolds in REYNOLDS_NUMBERS]
    summary_path = OUT_DIR / "regional_error_density_by_reynolds.csv"
    write_csv(summary_path, rows)

    print("\nRegional error-density summary:")
    print(summary_path)
    print("Re, vel corner/smooth, vel boundary/smooth, p corner/smooth, p boundary/smooth")
    for row in rows:
        print(
            f"{row['reynolds']:>4}: "
            f"{row['velocity_corner_over_smooth']:.3f}, "
            f"{row['velocity_boundary_over_smooth']:.3f}, "
            f"{row['pressure_corner_over_smooth']:.3f}, "
            f"{row['pressure_boundary_over_smooth']:.3f}"
        )


if __name__ == "__main__":
    main()
