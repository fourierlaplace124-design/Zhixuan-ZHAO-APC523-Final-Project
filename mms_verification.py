import csv
from pathlib import Path

import numpy as np

from mesh_generator import FiniteDifferenceMesh
from solver import NavierStokesSolver


PI = np.pi


def manufactured_fields(x, y):
    """Return exact u, v, p for a divergence-free manufactured field."""
    u = np.sin(PI * x) ** 2 * np.sin(2.0 * PI * y)
    v = -np.sin(PI * y) ** 2 * np.sin(2.0 * PI * x)
    p = np.sin(PI * x) * np.cos(PI * y)
    return u, v, p


def manufactured_derivatives(x, y):
    """Return exact derivatives needed by the steady momentum equations."""
    sin_px = np.sin(PI * x)
    sin_py = np.sin(PI * y)
    sin_2px = np.sin(2.0 * PI * x)
    sin_2py = np.sin(2.0 * PI * y)
    cos_px = np.cos(PI * x)
    cos_py = np.cos(PI * y)
    cos_2px = np.cos(2.0 * PI * x)
    cos_2py = np.cos(2.0 * PI * y)

    u_x = PI * sin_2px * sin_2py
    u_y = 2.0 * PI * sin_px**2 * cos_2py
    u_xx = 2.0 * PI**2 * cos_2px * sin_2py
    u_yy = -4.0 * PI**2 * sin_px**2 * sin_2py

    v_x = -2.0 * PI * sin_py**2 * cos_2px
    v_y = -PI * sin_2py * sin_2px
    v_xx = 4.0 * PI**2 * sin_py**2 * sin_2px
    v_yy = -2.0 * PI**2 * cos_2py * sin_2px

    p_x = PI * cos_px * cos_py
    p_y = -PI * sin_px * sin_py

    return u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, p_x, p_y


def exact_forcing(x, y, viscosity):
    """
    Return forcing for steady incompressible Navier-Stokes:

        u dot grad(u) + grad(p) - nu Laplacian(u) = f.
    """
    u, v, _ = manufactured_fields(x, y)
    u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, p_x, p_y = (
        manufactured_derivatives(x, y)
    )

    f_u = u * u_x + v * u_y + p_x - viscosity * (u_xx + u_yy)
    f_v = u * v_x + v * v_y + p_y - viscosity * (v_xx + v_yy)
    return f_u, f_v


def l2_norm(values):
    return float(np.sqrt(np.mean(values**2)))


def compute_rate(errors, spacings):
    rates = [None]
    for i in range(1, len(errors)):
        rates.append(np.log(errors[i - 1] / errors[i]) / np.log(spacings[i - 1] / spacings[i]))
    return rates


def run_case(nx, ny, viscosity):
    mesh = FiniteDifferenceMesh(nx, ny, [0.0, 1.0, 0.0, 1.0])
    mesh.cell_type[:, :] = 0

    u, v, p = manufactured_fields(mesh.X, mesh.Y)
    mesh.u = u.copy()
    mesh.v = v.copy()
    mesh.p = p.copy()

    flow_params = {
        "reynolds": 1.0 / viscosity,
        "viscosity": viscosity,
        "dt": 1.0,
        "inlet_velocity": 1.0,
    }
    solver_params = {"spatial_scheme": "central"}
    solver = NavierStokesSolver(mesh, flow_params, solver_params)

    du_conv, dv_conv = solver.compute_convection_central(mesh.u, mesh.v)
    du_diff, dv_diff = solver.compute_diffusion(mesh.u, mesh.v)
    dp_dx, dp_dy = solver.compute_pressure_gradient(mesh.p)

    f_u, f_v = exact_forcing(mesh.X, mesh.Y, viscosity)

    # solver.compute_convection_central returns -(u dot grad u), while
    # compute_diffusion returns +nu Laplacian(u).
    residual_u = -du_conv + dp_dx - du_diff - f_u
    residual_v = -dv_conv + dp_dy - dv_diff - f_v

    div = np.zeros_like(mesh.u)
    div[1:-1, 1:-1] = (
        (mesh.u[1:-1, 2:] - mesh.u[1:-1, :-2]) / (2.0 * mesh.dx)
        + (mesh.v[2:, 1:-1] - mesh.v[:-2, 1:-1]) / (2.0 * mesh.dy)
    )

    interior = np.s_[1:-1, 1:-1]
    momentum_error = np.sqrt(residual_u[interior] ** 2 + residual_v[interior] ** 2)

    return {
        "nx": nx,
        "ny": ny,
        "h": max(mesh.dx, mesh.dy),
        "l2_momentum": l2_norm(momentum_error),
        "linf_momentum": float(np.max(np.abs(momentum_error))),
        "l2_divergence": l2_norm(div[interior]),
        "linf_divergence": float(np.max(np.abs(div[interior]))),
    }


def run_mms(grid_sizes=(16, 24, 32, 48, 64), viscosity=0.05, output_dir="mms_results"):
    results = [run_case(n, n, viscosity) for n in grid_sizes]

    l2_rates = compute_rate([r["l2_momentum"] for r in results], [r["h"] for r in results])
    div_rates = compute_rate([r["l2_divergence"] for r in results], [r["h"] for r in results])

    for result, rate, div_rate in zip(results, l2_rates, div_rates):
        result["rate_l2_momentum"] = rate
        result["rate_l2_divergence"] = div_rate

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "mms_fdm_spatial_convergence.csv"
    markdown_path = output_path / "mms_fdm_spatial_convergence.md"

    fieldnames = [
        "nx",
        "ny",
        "h",
        "l2_momentum",
        "rate_l2_momentum",
        "linf_momentum",
        "l2_divergence",
        "rate_l2_divergence",
        "linf_divergence",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    with markdown_path.open("w") as f:
        f.write(
            "| Grid | h | L2 momentum residual | Momentum rate | "
            "L2 divergence | Divergence rate |\n"
        )
        f.write("|---:|---:|---:|---:|---:|---:|\n")
        for result in results:
            rate = result["rate_l2_momentum"]
            div_rate = result["rate_l2_divergence"]
            rate_text = "-" if rate is None else f"{rate:.3f}"
            div_rate_text = "-" if div_rate is None else f"{div_rate:.3f}"
            f.write(
                f"| {result['nx']}x{result['ny']} | "
                f"{result['h']:.3e} | "
                f"{result['l2_momentum']:.3e} | "
                f"{rate_text} | "
                f"{result['l2_divergence']:.3e} | "
                f"{div_rate_text} |\n"
            )

    print(f"MMS results written to {csv_path}")
    print(f"Markdown table written to {markdown_path}")
    print(
        "\nGrid       h          L2 momentum residual   rate    "
        "L2 divergence   rate"
    )
    for result in results:
        rate = result["rate_l2_momentum"]
        div_rate = result["rate_l2_divergence"]
        rate_text = "-" if rate is None else f"{rate:6.3f}"
        div_rate_text = "-" if div_rate is None else f"{div_rate:6.3f}"
        print(
            f"{result['nx']:4d}x{result['ny']:<4d}  "
            f"{result['h']:.3e}      "
            f"{result['l2_momentum']:.3e}       "
            f"{rate_text}    "
            f"{result['l2_divergence']:.3e}    "
            f"{div_rate_text}"
        )

    return results


if __name__ == "__main__":
    run_mms()
