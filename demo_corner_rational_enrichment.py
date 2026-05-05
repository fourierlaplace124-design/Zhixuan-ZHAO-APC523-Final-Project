from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_reynolds_corner_density import (
    COARSE_GRID,
    FIELD_DIR,
    REFERENCE_GRID,
    STEP_CORNER,
    field_path,
    interpolate_reference,
    load_field,
)


PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR.parent / "experiment_results_corner_enrichment"
FIG_DIR = PROJECT_DIR / "figures"

REYNOLDS_CASES = [20, 100]
PATCH_RADIUS = 0.95
INNER_RADIUS = 0.08
TEST_STRIDE = 3
N_POLES = 12


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def comparable_fluid_mask(field: dict, ref_on_field: dict) -> np.ndarray:
    fluid = field["cell_type"] == 0
    ref_fluid = np.rint(ref_on_field["cell_type"]).astype(int) == 0
    finite = np.isfinite(ref_on_field["u"]) & np.isfinite(ref_on_field["v"])
    return fluid & ref_fluid & finite


def corner_patch_mask(field: dict, ref_on_field: dict) -> np.ndarray:
    x = field["X"] - STEP_CORNER[0]
    y = field["Y"] - STEP_CORNER[1]
    r = np.hypot(x, y)
    return comparable_fluid_mask(field, ref_on_field) & (r >= INNER_RADIUS) & (r <= PATCH_RADIUS)


def polynomial_basis(z: np.ndarray) -> np.ndarray:
    x = z.real
    y = z.imag
    return np.column_stack(
        [
            np.ones_like(x),
            x,
            y,
            x * x,
            x * y,
            y * y,
        ]
    )


def clustered_poles() -> np.ndarray:
    """Place poles exponentially in the solid block behind the step corner."""
    distances = np.geomspace(0.035, 1.35, N_POLES)
    direction = np.array([-1.0, -1.0]) / math.sqrt(2.0)
    pole_xy = STEP_CORNER[None, :] + distances[:, None] * direction[None, :]
    return (pole_xy[:, 0] - STEP_CORNER[0]) + 1j * (pole_xy[:, 1] - STEP_CORNER[1])


def rational_basis(z: np.ndarray) -> np.ndarray:
    poles = clustered_poles()
    cols = []
    for pole in poles:
        phi = 1.0 / (z - pole)
        cols.append(phi.real)
        cols.append(phi.imag)
    return np.column_stack(cols)


def design_matrix(z: np.ndarray, enriched: bool) -> np.ndarray:
    poly = polynomial_basis(z)
    if not enriched:
        return poly
    return np.column_stack([poly, rational_basis(z)])


def split_train_test(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.arange(indices.size)
    test = indices[(order % TEST_STRIDE) == 0]
    train = indices[(order % TEST_STRIDE) != 0]
    return train, test


def fit_component(A_train: np.ndarray, values_train: np.ndarray, A_eval: np.ndarray) -> np.ndarray:
    scale = np.linalg.norm(A_train, axis=0)
    scale[scale == 0.0] = 1.0
    coef, *_ = np.linalg.lstsq(A_train / scale, values_train, rcond=1e-10)
    return (A_eval / scale) @ coef


def rms_velocity(eu: np.ndarray, ev: np.ndarray) -> float:
    return float(np.sqrt(np.mean(eu * eu + ev * ev)))


def analyze_case(reynolds: int) -> dict:
    coarse = load_field(field_path(reynolds, *COARSE_GRID))
    reference = load_field(field_path(reynolds, *REFERENCE_GRID))
    ref_on_coarse = interpolate_reference(reference, coarse)

    patch = corner_patch_mask(coarse, ref_on_coarse)
    patch_indices = np.flatnonzero(patch.ravel())
    train_indices, test_indices = split_train_test(patch_indices)

    z_all = (
        coarse["X"].ravel()[patch_indices] - STEP_CORNER[0]
        + 1j * (coarse["Y"].ravel()[patch_indices] - STEP_CORNER[1])
    )
    eu_all = (coarse["u"] - ref_on_coarse["u"]).ravel()[patch_indices]
    ev_all = (coarse["v"] - ref_on_coarse["v"]).ravel()[patch_indices]

    train_local = np.isin(patch_indices, train_indices)
    test_local = np.isin(patch_indices, test_indices)

    rows = {
        "reynolds": reynolds,
        "n_patch": int(patch_indices.size),
        "n_train": int(train_indices.size),
        "n_test": int(test_indices.size),
    }

    raw_train = rms_velocity(eu_all[train_local], ev_all[train_local])
    raw_test = rms_velocity(eu_all[test_local], ev_all[test_local])
    rows["raw_rms_train"] = raw_train
    rows["raw_rms_test"] = raw_test

    corrected_fields = {}
    for name, enriched in (("polynomial", False), ("rational", True)):
        A = design_matrix(z_all, enriched=enriched)
        A_train = A[train_local]

        pred_u = fit_component(A_train, eu_all[train_local], A)
        pred_v = fit_component(A_train, ev_all[train_local], A)
        rem_u = eu_all - pred_u
        rem_v = ev_all - pred_v

        train_rms = rms_velocity(rem_u[train_local], rem_v[train_local])
        test_rms = rms_velocity(rem_u[test_local], rem_v[test_local])

        rows[f"{name}_n_basis"] = int(A.shape[1])
        rows[f"{name}_rms_train"] = train_rms
        rows[f"{name}_rms_test"] = test_rms
        rows[f"{name}_test_reduction"] = 1.0 - test_rms / raw_test

        corrected_fields[name] = {
            "patch": patch,
            "patch_indices": patch_indices,
            "rem_mag": np.sqrt(rem_u * rem_u + rem_v * rem_v),
        }

    save_case_figure(reynolds, coarse, ref_on_coarse, patch, corrected_fields)
    return rows


def patch_image(field: dict, values_on_patch: np.ndarray, patch: np.ndarray) -> np.ndarray:
    image = np.full(field["u"].shape, np.nan)
    image[patch] = values_on_patch
    return image


def save_case_figure(reynolds: int, coarse: dict, ref_on_coarse: dict, patch: np.ndarray, corrected: dict) -> None:
    raw_mag = np.sqrt((coarse["u"] - ref_on_coarse["u"]) ** 2 + (coarse["v"] - ref_on_coarse["v"]) ** 2)
    rat_mag = patch_image(coarse, corrected["rational"]["rem_mag"], patch)

    x = coarse["x"]
    y = coarse["y"]
    window = (
        (coarse["X"] >= STEP_CORNER[0] - 0.15)
        & (coarse["X"] <= STEP_CORNER[0] + PATCH_RADIUS)
        & (coarse["Y"] >= STEP_CORNER[1] - PATCH_RADIUS)
        & (coarse["Y"] <= STEP_CORNER[1] + PATCH_RADIUS)
    )
    vmax = np.nanpercentile(raw_mag[patch], 97)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True)
    panels = [
        ("Raw velocity error", np.where(patch, raw_mag, np.nan)),
        ("After rational enrichment", rat_mag),
    ]

    for ax, (title, data) in zip(axes, panels):
        shown = np.where(window, data, np.nan)
        im = ax.pcolormesh(x, y, shown, shading="auto", cmap="magma", vmin=0.0, vmax=vmax)
        ax.plot(STEP_CORNER[0], STEP_CORNER[1], "wo", ms=3, mec="black", mew=0.5)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(STEP_CORNER[0] - 0.15, STEP_CORNER[0] + PATCH_RADIUS)
        ax.set_ylim(STEP_CORNER[1] - PATCH_RADIUS, STEP_CORNER[1] + PATCH_RADIUS)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im, ax=axes, label="velocity error magnitude")
    fig.savefig(FIG_DIR / f"corner_rational_enrichment_Re{reynolds}.png", dpi=220)
    plt.close(fig)


def write_summary(rows: list[dict]) -> Path:
    path = OUT_DIR / "corner_rational_enrichment_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    ensure_dirs()
    missing = [
        str(path)
        for reynolds in REYNOLDS_CASES
        for path in (field_path(reynolds, *COARSE_GRID), field_path(reynolds, *REFERENCE_GRID))
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing cached fields:\n" + "\n".join(missing))

    rows = [analyze_case(reynolds) for reynolds in REYNOLDS_CASES]
    summary_path = write_summary(rows)

    print(f"Saved summary: {summary_path}")
    print("Re | raw test RMS | polynomial reduction | rational reduction")
    for row in rows:
        print(
            f"{row['reynolds']:>3} | "
            f"{row['raw_rms_test']:.4e} | "
            f"{100.0 * row['polynomial_test_reduction']:.1f}% | "
            f"{100.0 * row['rational_test_reduction']:.1f}%"
        )


if __name__ == "__main__":
    main()
