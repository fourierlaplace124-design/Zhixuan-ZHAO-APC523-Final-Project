from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt


PROJECT_DIR = Path(__file__).resolve().parent
HOMEWORK_DIR = PROJECT_DIR.parent
SIM_DIR = PROJECT_DIR / "FlowSimulator"
OUT_DIR = HOMEWORK_DIR / "experiment_results_step_smooth_analysis"
FIELD_DIR = OUT_DIR / "fields"
FIG_DIR = OUT_DIR / "figures"

sys.path.insert(0, str(SIM_DIR))

from batch_experiment import BatchExperimentManager  # noqa: E402
from solver import NavierStokesSolver  # noqa: E402


REYNOLDS_NUMBER = 20
MAX_TIME = 80.0
GRIDS = [
    (80, 32),
    (120, 48),
    (160, 64),
    (200, 80),
    (240, 96),
    (280, 112),
    (320, 128),
]

STEP_CORNER = np.array([4.0, 2.0])
BOUNDARY_MARGIN = 0.30
CORNER_RADIUS = 0.75


def ensure_dirs() -> None:
    FIELD_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_case(manager: BatchExperimentManager, nx: int, ny: int) -> Path:
    case_name = f"Re{REYNOLDS_NUMBER}_N{nx}x{ny}"
    field_file = FIELD_DIR / f"{case_name}.npz"
    if field_file.exists():
        print(f"Skipping existing field file: {field_file}")
        return field_file

    print("\n" + "=" * 70)
    print(f"Running smooth-region case: {case_name}")
    print("=" * 70)

    mesh = manager.create_mesh(nx, ny)
    nu = manager.get_viscosity(REYNOLDS_NUMBER)
    dt = manager.get_time_step(nx, ny, REYNOLDS_NUMBER)

    flow_params = {
        "reynolds": REYNOLDS_NUMBER,
        "viscosity": nu,
        "dt": dt,
        "inlet_velocity": manager.inlet_velocity,
    }
    solver_params = {
        "spatial_scheme": "central",
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
        output_interval=500,
        convergence_check=True,
    )

    metadata = {
        "case_name": case_name,
        "reynolds": REYNOLDS_NUMBER,
        "nx": nx,
        "ny": ny,
        "dx": mesh.dx,
        "dy": mesh.dy,
        "h": min(mesh.dx, mesh.dy),
        "nu": nu,
        "dt": dt,
        "max_time": MAX_TIME,
        **result,
    }

    np.savez_compressed(
        field_file,
        x=mesh.x,
        y=mesh.y,
        cell_type=mesh.cell_type,
        u=mesh.u,
        v=mesh.v,
        p=mesh.p,
        metadata=json.dumps(metadata),
    )
    print(f"Saved field file: {field_file}")
    return field_file


def load_field(field_file: Path) -> dict:
    with np.load(field_file) as data:
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


def interpolate_reference(interpolators: dict, target: dict) -> dict:
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

    distance_to_boundary = distance_transform_edt(fluid, sampling=(field["metadata"]["dy"], field["metadata"]["dx"]))
    distance_to_corner = np.sqrt(
        (field["X"] - STEP_CORNER[0]) ** 2
        + (field["Y"] - STEP_CORNER[1]) ** 2
    )

    smooth = full & (distance_to_boundary >= BOUNDARY_MARGIN) & (distance_to_corner >= CORNER_RADIUS)
    corner = full & (distance_to_corner < CORNER_RADIUS)
    return {"full": full, "smooth": smooth, "corner": corner}


def relative_velocity_l2(field: dict, ref_on_field: dict, mask: np.ndarray) -> float:
    if not np.any(mask):
        return math.nan
    diff2 = (field["u"] - ref_on_field["u"]) ** 2 + (field["v"] - ref_on_field["v"]) ** 2
    ref2 = ref_on_field["u"] ** 2 + ref_on_field["v"] ** 2
    denom = float(np.sum(ref2[mask]))
    if denom <= 0.0:
        return math.nan
    return float(np.sqrt(np.sum(diff2[mask]) / denom))


def relative_pressure_l2(field: dict, ref_on_field: dict, mask: np.ndarray) -> float:
    if not np.any(mask):
        return math.nan
    p = field["p"][mask]
    pref = ref_on_field["p"][mask]
    p = p - np.mean(p)
    pref = pref - np.mean(pref)
    denom = float(np.sum(pref**2))
    if denom <= 0.0:
        return math.nan
    return float(np.sqrt(np.sum((p - pref) ** 2) / denom))


def estimate_pairwise_orders(rows: list[dict], metric: str) -> list[dict]:
    orders = []
    valid = [row for row in rows if row["nx"] < max(item["nx"] for item in rows)]
    for coarse, fine in zip(valid[:-1], valid[1:]):
        e_coarse = coarse[metric]
        e_fine = fine[metric]
        if e_coarse > 0 and e_fine > 0:
            order = math.log(e_coarse / e_fine) / math.log(coarse["h"] / fine["h"])
        else:
            order = math.nan
        orders.append(
            {
                "metric": metric,
                "coarse_grid": f"{coarse['nx']}x{coarse['ny']}",
                "fine_grid": f"{fine['nx']}x{fine['ny']}",
                "order": order,
            }
        )
    return orders


def difference_velocity_l2(a: dict, b_on_a: dict, mask: np.ndarray) -> float:
    if not np.any(mask):
        return math.nan
    diff2 = (a["u"] - b_on_a["u"]) ** 2 + (a["v"] - b_on_a["v"]) ** 2
    return float(np.sqrt(np.mean(diff2[mask])))


def difference_pressure_l2(a: dict, b_on_a: dict, mask: np.ndarray) -> float:
    if not np.any(mask):
        return math.nan
    pa = a["p"][mask]
    pb = b_on_a["p"][mask]
    pa = pa - np.mean(pa)
    pb = pb - np.mean(pb)
    return float(np.sqrt(np.mean((pa - pb) ** 2)))


def solve_richardson_order(ratio: float, h3: float, h2: float, h1: float) -> float:
    if not np.isfinite(ratio) or ratio <= 0:
        return math.nan

    def model(p: float) -> float:
        denominator = h2**p - h1**p
        if denominator <= 0:
            return math.nan
        return (h3**p - h2**p) / denominator

    low = 1.0e-8
    high = 8.0
    f_low = model(low) - ratio
    f_high = model(high) - ratio
    if not np.isfinite(f_low) or not np.isfinite(f_high) or f_low * f_high > 0:
        return math.nan

    for _ in range(80):
        mid = 0.5 * (low + high)
        f_mid = model(mid) - ratio
        if abs(f_mid) < 1.0e-10:
            return mid
        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    return 0.5 * (low + high)


def estimate_richardson_triplets(fields: list[dict]) -> list[dict]:
    rows = []
    for coarse, middle, fine in zip(fields[:-2], fields[1:-1], fields[2:]):
        middle_on_coarse = interpolate_reference(make_interpolators(middle), coarse)
        fine_on_coarse = interpolate_reference(make_interpolators(fine), coarse)

        masks = region_masks(coarse, fine_on_coarse)
        middle_fluid = np.rint(middle_on_coarse["cell_type"]).astype(int) == 0
        finite_middle = (
            np.isfinite(middle_on_coarse["u"])
            & np.isfinite(middle_on_coarse["v"])
            & np.isfinite(middle_on_coarse["p"])
        )

        for region, base_mask in masks.items():
            mask = base_mask & middle_fluid & finite_middle
            d32_velocity = difference_velocity_l2(coarse, middle_on_coarse, mask)
            d21_velocity = difference_velocity_l2(
                {
                    **coarse,
                    "u": middle_on_coarse["u"],
                    "v": middle_on_coarse["v"],
                    "p": middle_on_coarse["p"],
                },
                fine_on_coarse,
                mask,
            )
            d32_pressure = difference_pressure_l2(coarse, middle_on_coarse, mask)
            d21_pressure = difference_pressure_l2(
                {
                    **coarse,
                    "u": middle_on_coarse["u"],
                    "v": middle_on_coarse["v"],
                    "p": middle_on_coarse["p"],
                },
                fine_on_coarse,
                mask,
            )

            h3 = coarse["metadata"]["h"]
            h2 = middle["metadata"]["h"]
            h1 = fine["metadata"]["h"]
            rows.append(
                {
                    "region": region,
                    "coarse_grid": f"{coarse['metadata']['nx']}x{coarse['metadata']['ny']}",
                    "middle_grid": f"{middle['metadata']['nx']}x{middle['metadata']['ny']}",
                    "fine_grid": f"{fine['metadata']['nx']}x{fine['metadata']['ny']}",
                    "n_points": int(np.sum(mask)),
                    "velocity_d32": d32_velocity,
                    "velocity_d21": d21_velocity,
                    "velocity_order": solve_richardson_order(d32_velocity / d21_velocity, h3, h2, h1),
                    "pressure_d32": d32_pressure,
                    "pressure_d21": d21_pressure,
                    "pressure_order": solve_richardson_order(d32_pressure / d21_pressure, h3, h2, h1),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(rows: list[dict]) -> Path:
    comparable = [row for row in rows if row["nx"] < max(item["nx"] for item in rows)]
    h = np.array([row["h"] for row in comparable])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)

    specs = [
        (axes[0], "velocity", "Velocity relative difference"),
        (axes[1], "pressure", "Pressure relative difference"),
    ]
    for ax, prefix, ylabel in specs:
        for region, style in [
            ("smooth", "o-"),
            ("full", "s--"),
            ("corner", "^:"),
        ]:
            values = np.array([row[f"{region}_{prefix}_rel_diff"] for row in comparable])
            ax.loglog(h, values, style, label=region)

        smooth_values = np.array([row[f"smooth_{prefix}_rel_diff"] for row in comparable])
        if np.all(np.isfinite(smooth_values)) and np.all(smooth_values > 0):
            ideal = smooth_values[0] * (h / h[0]) ** 2
            ax.loglog(h, ideal, "k-", linewidth=1.0, alpha=0.65, label=r"$O(h^2)$")

        ax.set_xlabel(r"$h=\min(\Delta x,\Delta y)$")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
        ax.invert_xaxis()
        ax.legend(frameon=False)

    figure_path = FIG_DIR / "smooth_region_convergence.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    return figure_path


def make_smooth_focus_plot(rows: list[dict]) -> Path:
    comparable = [row for row in rows if row["nx"] < max(item["nx"] for item in rows)]
    h = np.array([row["h"] for row in comparable])
    velocity = np.array([row["smooth_velocity_rel_diff"] for row in comparable])
    h_ref = min(item["h"] for item in rows)

    reference_model = np.abs(h**2 - h_ref**2)

    def fit_scale(values: np.ndarray) -> float:
        return float(np.exp(np.mean(np.log(values / reference_model))))

    velocity_scale = fit_scale(velocity)
    velocity_model = velocity_scale * reference_model

    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    ax.loglog(h, velocity, "o-", label="velocity, smooth region")
    ax.loglog(
        h,
        velocity_model,
        "k--",
        linewidth=1.0,
        alpha=0.8,
        label=r"$C_v |h^2-h_{\mathrm{ref}}^2|$",
    )

    ax.set_xlabel(r"$h=\min(\Delta x,\Delta y)$")
    ax.set_ylabel("Relative difference to reference solution")
    ax.grid(True, which="both", alpha=0.25)
    ax.invert_xaxis()
    ax.legend(frameon=False)

    figure_path = FIG_DIR / "smooth_region_convergence_focus.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    return figure_path


def make_smooth_region_overlay(field: dict) -> Path:
    ref_on_field = {
        "u": field["u"],
        "v": field["v"],
        "p": field["p"],
        "cell_type": field["cell_type"].astype(float),
    }
    masks = region_masks(field, ref_on_field)

    speed = np.sqrt(field["u"] ** 2 + field["v"] ** 2)
    speed = np.ma.masked_where(field["cell_type"] != 0, speed)

    fig, ax = plt.subplots(figsize=(8.6, 3.8), constrained_layout=True)
    pcm = ax.pcolormesh(field["X"], field["Y"], speed, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=ax, label="Velocity magnitude")

    solid = np.ma.masked_where(field["cell_type"] != 1, np.ones_like(field["cell_type"], dtype=float))
    ax.pcolormesh(
        field["X"],
        field["Y"],
        solid,
        shading="auto",
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        alpha=0.55,
    )

    smooth = np.ma.masked_where(~masks["smooth"], np.ones_like(field["cell_type"], dtype=float))
    ax.pcolormesh(
        field["X"],
        field["Y"],
        smooth,
        shading="auto",
        cmap="Greens",
        vmin=0.0,
        vmax=1.0,
        alpha=0.35,
    )

    boundary_contour = masks["smooth"].astype(float)
    ax.contour(
        field["X"],
        field["Y"],
        boundary_contour,
        levels=[0.5],
        colors=["limegreen"],
        linewidths=1.6,
    )

    circle = plt.Circle(STEP_CORNER, CORNER_RADIUS, color="crimson", fill=False, linestyle="--", linewidth=1.2)
    ax.add_patch(circle)
    ax.scatter([STEP_CORNER[0]], [STEP_CORNER[1]], color="crimson", s=18, zorder=5)
    ax.text(STEP_CORNER[0] + 0.12, STEP_CORNER[1] + 0.08, "corner", color="crimson", fontsize=9)
    ax.text(6.15, 3.42, r"$\Omega_s$", color="darkgreen", fontsize=12)

    ax.set_xlim(field["x"][0], field["x"][-1])
    ax.set_ylim(field["y"][0], field["y"][-1])
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Smooth-region mask on the 320x128 reference velocity field")

    figure_path = FIG_DIR / "smooth_region_mask_320x128.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    return figure_path


def main() -> None:
    ensure_dirs()

    manager = BatchExperimentManager(output_dir=str(OUT_DIR / "raw_solver_output"), enable_fvm_baseline=False)
    field_files = [run_case(manager, nx, ny) for nx, ny in GRIDS]
    fields = [load_field(path) for path in field_files]
    reference = fields[-1]
    interpolators = make_interpolators(reference)

    rows = []
    for field in fields:
        ref_on_field = interpolate_reference(interpolators, field)
        masks = region_masks(field, ref_on_field)
        meta = field["metadata"]
        row = {
            "case_name": meta["case_name"],
            "nx": meta["nx"],
            "ny": meta["ny"],
            "dx": meta["dx"],
            "dy": meta["dy"],
            "h": meta["h"],
            "dt": meta["dt"],
            "final_time": meta["final_time"],
            "final_steps": meta["final_steps"],
            "final_residual": meta["final_residual"],
            "final_C_d": meta["final_C_d"],
            "converged": meta["converged"],
            "convergence_reason": meta["convergence_reason"],
            "n_full": int(np.sum(masks["full"])),
            "n_smooth": int(np.sum(masks["smooth"])),
            "n_corner": int(np.sum(masks["corner"])),
        }
        for region, mask in masks.items():
            row[f"{region}_velocity_rel_diff"] = relative_velocity_l2(field, ref_on_field, mask)
            row[f"{region}_pressure_rel_diff"] = relative_pressure_l2(field, ref_on_field, mask)
        rows.append(row)

    rows = sorted(rows, key=lambda item: item["nx"])
    summary_file = OUT_DIR / "smooth_region_convergence.csv"
    write_csv(summary_file, rows)

    order_rows = []
    for metric in ("smooth_velocity_rel_diff", "smooth_pressure_rel_diff"):
        order_rows.extend(estimate_pairwise_orders(rows, metric))
    order_file = OUT_DIR / "smooth_region_orders.csv"
    write_csv(order_file, order_rows)

    richardson_rows = estimate_richardson_triplets(fields)
    richardson_file = OUT_DIR / "smooth_region_richardson_triplets.csv"
    write_csv(richardson_file, richardson_rows)

    figure_path = make_plot(rows)
    focus_figure_path = make_smooth_focus_plot(rows)
    mask_figure_path = make_smooth_region_overlay(reference)

    print("\nSmooth-region convergence summary:")
    print(f"  Summary CSV: {summary_file}")
    print(f"  Orders CSV:  {order_file}")
    print(f"  Richardson:  {richardson_file}")
    print(f"  Figure:      {figure_path}")
    print(f"  Focus plot:  {focus_figure_path}")
    print(f"  Mask plot:   {mask_figure_path}")
    print("\nPairwise smooth-region orders:")
    for row in order_rows:
        print(
            f"  {row['metric']}: {row['coarse_grid']} -> {row['fine_grid']} "
            f"p = {row['order']:.3f}"
        )

    print("\nThree-grid Richardson orders in the smooth region:")
    for row in richardson_rows:
        if row["region"] == "smooth":
            print(
                f"  {row['coarse_grid']} / {row['middle_grid']} / {row['fine_grid']}: "
                f"velocity p = {row['velocity_order']:.3f}, "
                f"pressure p = {row['pressure_order']:.3f}"
            )


if __name__ == "__main__":
    main()
