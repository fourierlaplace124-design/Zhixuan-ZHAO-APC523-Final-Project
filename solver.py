import numpy as np
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import spsolve


class NavierStokesSolver:
    def __init__(self, mesh, flow_params, solver_params):
        self.mesh = mesh
        self.Re = flow_params['reynolds']
        self.nu = flow_params['viscosity']
        self.dt = flow_params['dt']
        self.inlet_velocity = flow_params.get('inlet_velocity', 1.0)

        self.spatial_scheme = solver_params.get('spatial_scheme', '中心差分')
        self.temporal_scheme = solver_params.get('temporal_scheme', '显式欧拉')

        self.time = 0.0
        self.step = 0

        self.u_old = self.mesh.u.copy()
        self.v_old = self.mesh.v.copy()

        self.residual_history = []
        self.max_velocity_history = []
        self.cfl_history = []

        self.drag_coefficient_history = []
        self.lift_coefficient_history = []
        self.time_history = []

        self.convergence_tolerance = solver_params.get('convergence_tolerance', 1e-6)
        self.max_iterations = solver_params.get('max_iterations', 100000)
        self.check_convergence_interval = solver_params.get('check_convergence_interval', 10)
        self.cd_convergence_tolerance = solver_params.get('cd_convergence_tolerance', 0.01)
        self.cd_convergence_window_time = solver_params.get('cd_convergence_window_time', 5.0)
        self.cd_convergence_min_time = solver_params.get(
            'cd_convergence_min_time',
            self.cd_convergence_window_time
        )
        self.last_cd_relative_change = None

    def check_cd_convergence(self):
        """
        Check steady-state convergence based on recent drag coefficient change.

        The criterion compares the current C_d against the oldest stored value
        that is at least cd_convergence_window_time in the past. This keeps the
        convergence window tied to physical time rather than to a grid-dependent
        number of time steps.
        """
        if len(self.drag_coefficient_history) < 2:
            return False
        if self.time < self.cd_convergence_min_time:
            return False

        current_cd = self.drag_coefficient_history[-1]
        current_time = self.time_history[-1]
        target_time = current_time - self.cd_convergence_window_time

        reference_cd = None
        for time_value, cd_value in zip(self.time_history, self.drag_coefficient_history):
            if time_value >= target_time:
                reference_cd = cd_value
                break

        if reference_cd is None:
            return False

        scale = max(abs(reference_cd), abs(current_cd), 1e-12)
        self.last_cd_relative_change = abs(current_cd - reference_cd) / scale
        return self.last_cd_relative_change < self.cd_convergence_tolerance

    def compute_convection_central(self, u, v):
        dx, dy = self.mesh.dx, self.mesh.dy

        du_conv = np.zeros_like(u)
        dv_conv = np.zeros_like(v)

        interior = np.s_[1:-1, 1:-1]
        fluid = (self.mesh.cell_type[interior] == 0)

        u_c = u[interior]
        v_c = v[interior]

        du_dx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
        du_dy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)
        dv_dx = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)
        dv_dy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)

        du_conv_i = -(u_c * du_dx + v_c * du_dy)
        dv_conv_i = -(u_c * dv_dx + v_c * dv_dy)

        du_inner = du_conv[interior]
        dv_inner = dv_conv[interior]
        du_inner[fluid] = du_conv_i[fluid]
        dv_inner[fluid] = dv_conv_i[fluid]

        return du_conv, dv_conv

    def compute_convection_upwind(self, u, v):
        dx, dy = self.mesh.dx, self.mesh.dy

        du_conv = np.zeros_like(u)
        dv_conv = np.zeros_like(v)

        interior = np.s_[1:-1, 1:-1]
        fluid = (self.mesh.cell_type[interior] == 0)

        u_c = u[interior]
        v_c = v[interior]

        du_dx = np.where(
            u_c > 0,
            (u_c - u[1:-1, :-2]) / dx,
            (u[1:-1, 2:] - u_c) / dx
        )
        dv_dx = np.where(
            u_c > 0,
            (v_c - v[1:-1, :-2]) / dx,
            (v[1:-1, 2:] - v_c) / dx
        )
        du_dy = np.where(
            v_c > 0,
            (u_c - u[:-2, 1:-1]) / dy,
            (u[2:, 1:-1] - u_c) / dy
        )
        dv_dy = np.where(
            v_c > 0,
            (v_c - v[:-2, 1:-1]) / dy,
            (v[2:, 1:-1] - v_c) / dy
        )

        du_conv_i = -(u_c * du_dx + v_c * du_dy)
        dv_conv_i = -(u_c * dv_dx + v_c * dv_dy)

        du_inner = du_conv[interior]
        dv_inner = dv_conv[interior]
        du_inner[fluid] = du_conv_i[fluid]
        dv_inner[fluid] = dv_conv_i[fluid]

        return du_conv, dv_conv


    def compute_diffusion(self, u, v):
        dx, dy = self.mesh.dx, self.mesh.dy

        du_diff = np.zeros_like(u)
        dv_diff = np.zeros_like(v)

        interior = np.s_[1:-1, 1:-1]
        fluid = (self.mesh.cell_type[interior] == 0)

        d2u_dx2 = (u[1:-1, 2:] - 2 * u[interior] + u[1:-1, :-2]) / dx**2
        d2u_dy2 = (u[2:, 1:-1] - 2 * u[interior] + u[:-2, 1:-1]) / dy**2
        d2v_dx2 = (v[1:-1, 2:] - 2 * v[interior] + v[1:-1, :-2]) / dx**2
        d2v_dy2 = (v[2:, 1:-1] - 2 * v[interior] + v[:-2, 1:-1]) / dy**2

        du_diff_i = self.nu * (d2u_dx2 + d2u_dy2)
        dv_diff_i = self.nu * (d2v_dx2 + d2v_dy2)

        du_inner = du_diff[interior]
        dv_inner = dv_diff[interior]
        du_inner[fluid] = du_diff_i[fluid]
        dv_inner[fluid] = dv_diff_i[fluid]

        return du_diff, dv_diff

    def compute_pressure_gradient(self, p):
        dx, dy = self.mesh.dx, self.mesh.dy

        dp_dx = np.zeros_like(p)
        dp_dy = np.zeros_like(p)

        interior = np.s_[1:-1, 1:-1]
        fluid = (self.mesh.cell_type[interior] == 0)

        dp_dx_i = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
        dp_dy_i = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)

        dp_dx_inner = dp_dx[interior]
        dp_dy_inner = dp_dy[interior]
        dp_dx_inner[fluid] = dp_dx_i[fluid]
        dp_dy_inner[fluid] = dp_dy_i[fluid]

        return dp_dx, dp_dy

    def solve_pressure_poisson(self, u_star, v_star):
        """
        Solve pressure Poisson equation with vectorized Jacobi updates.

        The outlet uses a fixed pressure reference, matching a pressure-outlet
        condition.  Other external boundaries use zero normal pressure
        gradient.
        """
        nx, ny = self.mesh.nx, self.mesh.ny
        dx, dy = self.mesh.dx, self.mesh.dy
        idx2 = 1.0 / dx**2
        idy2 = 1.0 / dy**2
        poisson_denominator = 2.0 * (idx2 + idy2)

        div_u = np.zeros((ny, nx))
        interior = np.s_[1:-1, 1:-1]
        fluid = (self.mesh.cell_type[interior] == 0)
        div_i = (
            (u_star[1:-1, 2:] - u_star[1:-1, :-2]) / (2 * dx)
            + (v_star[2:, 1:-1] - v_star[:-2, 1:-1]) / (2 * dy)
        )
        div_inner = div_u[interior]
        div_inner[fluid] = div_i[fluid]

        p = self.mesh.p.copy()
        p[:, -1] = 0.0
        max_iter = 100
        tol = 1e-6

        for iteration in range(max_iter):
            p_old = p.copy()

            p_update = (
                idx2 * (p_old[1:-1, 2:] + p_old[1:-1, :-2])
                + idy2 * (p_old[2:, 1:-1] + p_old[:-2, 1:-1])
                - div_u[interior] / self.dt
            ) / poisson_denominator
            p_inner = p[interior]
            p_inner[fluid] = p_update[fluid]

            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]
            p[:, 0] = p[:, 1]
            p[:, -1] = 0.0

            error = np.max(np.abs(p - p_old))
            if error < tol:
                break

        return p



    def compute_residual(self, u_new, v_new):
        du = u_new - self.u_old
        dv = v_new - self.v_old

        fluid_mask = (self.mesh.cell_type == 0)

        residual = np.sqrt(np.sum(du[fluid_mask]**2 + dv[fluid_mask]**2) / np.sum(fluid_mask))

        return residual

    def compute_cfl_number(self):
        """
        Compute current CFL number

        CFL = max(|u|) * dt / min(dx, dy)
        """
        u_max = np.max(np.abs(self.mesh.u))
        v_max = np.max(np.abs(self.mesh.v))
        velocity_max = max(u_max, v_max)

        grid_spacing = min(self.mesh.dx, self.mesh.dy)
        cfl = velocity_max * self.dt / grid_spacing

        return cfl

    def compute_drag_lift_coefficients(self):
        """
        Compute drag and lift coefficients on obstacle

        C_d = F_d / (0.5 * rho * U^2 * L)
        C_l = F_l / (0.5 * rho * U^2 * L)
        """
        obstacle_mask = (self.mesh.cell_type == 1)

        if not np.any(obstacle_mask):
            return 0.0, 0.0

        F_x = 0.0
        F_y = 0.0

        dx, dy = self.mesh.dx, self.mesh.dy

        for j in range(1, self.mesh.ny - 1):
            for i in range(1, self.mesh.nx - 1):
                if obstacle_mask[j, i]:
                    if not obstacle_mask[j, i+1]:  # Right face
                        F_x += self.mesh.p[j, i] * dy
                    if not obstacle_mask[j, i-1]:  # Left face
                        F_x -= self.mesh.p[j, i] * dy
                    if not obstacle_mask[j+1, i]:  # Top face
                        F_y += self.mesh.p[j, i] * dx
                    if not obstacle_mask[j-1, i]:  # Bottom face
                        F_y -= self.mesh.p[j, i] * dx

                    if not obstacle_mask[j, i+1]:
                        du_dx = (self.mesh.u[j, i+1] - self.mesh.u[j, i]) / dx
                        F_x += self.nu * du_dx * dy
                    if not obstacle_mask[j, i-1]:
                        du_dx = (self.mesh.u[j, i] - self.mesh.u[j, i-1]) / dx
                        F_x += self.nu * du_dx * dy

        # Characteristic length L = 1.0 (from experiment plan)
        L = 1.0
        rho = 1.0  # Assume density = 1
        U = self.inlet_velocity

        denom = 0.5 * rho * U**2 * L
        C_d = F_x / denom if denom > 0 else 0.0
        C_l = F_y / denom if denom > 0 else 0.0

        return C_d, C_l

    def check_divergence(self):
        if np.any(np.isnan(self.mesh.u)) or np.any(np.isnan(self.mesh.v)) or np.any(np.isnan(self.mesh.p)):
            return True

        if np.any(np.isinf(self.mesh.u)) or np.any(np.isinf(self.mesh.v)) or np.any(np.isinf(self.mesh.p)):
            return True

        max_velocity = max(np.max(np.abs(self.mesh.u)), np.max(np.abs(self.mesh.v)))
        if max_velocity > 100 * self.inlet_velocity:
            return True

        return False

    def step_forward(self):
        """
        Advance one time step

        Using fractional step projection method (Chorin's projection method)
        """
        self.u_old = self.mesh.u.copy()
        self.v_old = self.mesh.v.copy()

        u = self.mesh.u
        v = self.mesh.v
        p = self.mesh.p

        if self.spatial_scheme in ('upwind'):
            du_conv, dv_conv = self.compute_convection_upwind(u, v)
        else:  # Default central difference
            du_conv, dv_conv = self.compute_convection_central(u, v)

        du_diff, dv_diff = self.compute_diffusion(u, v)

        u_star = u + self.dt * (du_conv + du_diff)
        v_star = v + self.dt * (dv_conv + dv_diff)

        self.apply_boundary_conditions(u_star, v_star)

        p_new = self.solve_pressure_poisson(u_star, v_star)

        dp_dx, dp_dy = self.compute_pressure_gradient(p_new)
        u_new = u_star - self.dt * dp_dx
        v_new = v_star - self.dt * dp_dy

        self.apply_boundary_conditions(u_new, v_new)

        self.mesh.u = u_new
        self.mesh.v = v_new
        self.mesh.p = p_new

        self.time += self.dt
        self.step += 1

        residual = self.compute_residual(u_new, v_new)
        cfl = self.compute_cfl_number()

        self.residual_history.append(residual)
        self.cfl_history.append(cfl)

        return residual, cfl

    def apply_boundary_conditions(self, u, v):
        # Obstacle: no-slip
        obstacle_mask = (self.mesh.cell_type == 1)
        u[obstacle_mask] = 0.0
        v[obstacle_mask] = 0.0

        # Inlet: fixed velocity
        inlet_mask = (self.mesh.cell_type == 2)
        u[inlet_mask] = self.inlet_velocity
        v[inlet_mask] = 0.0

        # Walls: no-slip
        wall_mask = (self.mesh.cell_type == 4)
        u[wall_mask] = 0.0
        v[wall_mask] = 0.0

        # Outlet: zero gradient (Neumann boundary)
        outlet_mask = (self.mesh.cell_type == 3)
        outlet_indices = np.argwhere(outlet_mask)
        for j, i in outlet_indices:
            if i > 0:
                u[j, i] = u[j, i - 1]
                v[j, i] = v[j, i - 1]

    def run_simulation(self, total_time=None, output_interval=None, convergence_check=True):
        if total_time is not None:
            max_steps = max(1, int(np.ceil(total_time / self.dt)))
        else:
            max_steps = self.max_iterations

        print(f"Starting simulation:")
        print(f"  Reynolds number: {self.Re}")
        print(f"  Grid: {self.mesh.nx} x {self.mesh.ny}")
        print(f"  Time step: {self.dt}")
        print(f"  Max steps: {max_steps}")
        print(f"  Convergence tolerance: {self.convergence_tolerance}")
        print(f"  C_d convergence tolerance: {100 * self.cd_convergence_tolerance:.2f}%")
        print(f"  C_d convergence window: {self.cd_convergence_window_time}")
        print("-" * 60)

        converged = False
        diverged = False
        convergence_reason = ''

        for step in range(max_steps):
            residual, cfl = self.step_forward()

            if self.check_divergence():
                print(f"\n!!! Simulation DIVERGED at step {step}, time {self.time:.4f}")
                print(f"    CFL number: {cfl:.4f}")
                diverged = True
                break

            if step % self.check_convergence_interval == 0:
                C_d, C_l = self.compute_drag_lift_coefficients()
                self.drag_coefficient_history.append(C_d)
                self.lift_coefficient_history.append(C_l)
                self.time_history.append(self.time)

            if output_interval and step % output_interval == 0:
                C_d, C_l = self.compute_drag_lift_coefficients()
                print(f"Step: {step:6d} | Time: {self.time:8.4f} | Residual: {residual:.2e} | "
                      f"CFL: {cfl:.4f} | C_d: {C_d:.4f} | C_l: {C_l:.4f}")

            if convergence_check and step % self.check_convergence_interval == 0 and step > 0:
                if residual < self.convergence_tolerance:
                    print(f"\n*** Simulation CONVERGED at step {step}, time {self.time:.4f}")
                    print(f"    Final residual: {residual:.2e}")
                    converged = True
                    convergence_reason = 'residual'
                    break

                if self.check_cd_convergence():
                    print(f"\n*** Simulation CONVERGED at step {step}, time {self.time:.4f}")
                    print(f"    C_d relative change over recent window: "
                          f"{100 * self.last_cd_relative_change:.3f}%")
                    converged = True
                    convergence_reason = 'cd_stability'
                    break

            if cfl > 1.0 and step % output_interval == 0:
                print(f"    WARNING: CFL = {cfl:.4f} > 1.0, consider reducing time step")

        print("-" * 60)
        if diverged:
            print("Simulation status: DIVERGED")
        elif converged:
            print(f"Simulation status: CONVERGED ({convergence_reason})")
        else:
            print("Simulation status: COMPLETED (max iterations reached)")

        print(f"Total steps: {self.step}")
        print(f"Total time: {self.time:.4f}")
        final_residual = self.residual_history[-1] if self.residual_history else 0.0
        print(f"Final residual: {final_residual:.2e}")

        C_d, C_l = self.compute_drag_lift_coefficients()
        print(f"Final C_d: {C_d:.6f}")
        print(f"Final C_l: {C_l:.6f}")

        return {
            'converged': converged,
            'diverged': diverged,
            'final_time': self.time,
            'final_steps': self.step,
            'final_residual': final_residual,
            'final_C_d': C_d,
            'final_C_l': C_l,
            'convergence_reason': convergence_reason,
            'final_cd_relative_change': self.last_cd_relative_change
        }

    def export_history(self, filename):
        import csv

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'Residual', 'CFL', 'C_d', 'C_l'])

            # Align histories
            for i in range(len(self.time_history)):
                time = self.time_history[i]
                idx = min(i * self.check_convergence_interval, len(self.residual_history) - 1)
                residual = self.residual_history[idx] if idx < len(self.residual_history) else 0.0
                cfl = self.cfl_history[idx] if idx < len(self.cfl_history) else 0.0
                C_d = self.drag_coefficient_history[i] if i < len(self.drag_coefficient_history) else 0.0
                C_l = self.lift_coefficient_history[i] if i < len(self.lift_coefficient_history) else 0.0

                writer.writerow([time, residual, cfl, C_d, C_l])

        print(f"History data exported to: {filename}")




if __name__ == '__main__':
    print("Solver module test")
