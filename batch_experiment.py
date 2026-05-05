import os
import json
import numpy as np
from datetime import datetime
import csv
from collections import defaultdict

from mesh_generator import FiniteDifferenceMesh
from solver import NavierStokesSolver


class BatchExperimentManager:
    def __init__(self, output_dir='results', enable_fvm_baseline=False):
        self.output_dir = output_dir
        self.results_summary = []
        self.inlet_velocity = 1.0
        self.reference_length = 2.0  # backward-facing-step height
        self.convergence_tolerance = 1e-6
        self.cd_convergence_tolerance = 0.01
        self.cd_convergence_window_time = 5.0
        self.enable_fvm_baseline = False

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'history'), exist_ok=True)

    def create_obstacle_geometry(self):
        """
        Create the solid region for a backward-facing step.

        The channel is [0, 10] x [0, 4].  The upstream lower region
        [0, 4] x [0, 2] is solid, producing a backward-facing step at x = 4.
        """
        step_vertices = np.array([
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 2.0],
            [0.0, 2.0],
        ])

        return step_vertices

    def create_mesh(self, nx, ny):
        domain = [0, 10, 0, 4]  # [x_min, x_max, y_min, y_max]

        mesh = FiniteDifferenceMesh(nx, ny, domain)

        # Mark channel boundaries.  The left boundary is made an inlet only
        # after the solid step is marked below.
        mesh.cell_type[:, -1] = 3
        mesh.cell_type[0, :] = 4
        mesh.cell_type[-1, :] = 4

        step = self.create_obstacle_geometry()
        mesh.mark_obstacle(step)

        # Inlet is the open upper part of the left boundary.
        inlet_mask = (mesh.Y[:, 0] >= 2.0) & (mesh.cell_type[:, 0] != 4)
        mesh.cell_type[inlet_mask, 0] = 2

        inlet_velocity = self.inlet_velocity

        fluid_mask = (mesh.cell_type == 0)
        mesh.u[fluid_mask] = inlet_velocity
        mesh.v[fluid_mask] = 0.0

        wall_mask = (mesh.cell_type == 4)
        mesh.u[wall_mask] = 0.0
        mesh.v[wall_mask] = 0.0

        mesh.set_boundary_conditions(inlet_velocity=inlet_velocity)

        return mesh

    def get_viscosity(self, reynolds_number):
        """
        Compute kinematic viscosity from Reynolds number.

        For the backward-facing-step case, Re is based on the inlet speed and
        the step height, matching the COMSOL setup:

            Re = U * h / nu

        with U = 1.0 and h = 2.0.
        """
        return self.inlet_velocity * self.reference_length / reynolds_number

    def get_time_step(self, nx, ny, reynolds_number):
        domain = [0, 10, 0, 4]
        dx = (domain[1] - domain[0]) / nx
        dy = (domain[3] - domain[2]) / ny

        # CFL condition: dt <= CFL * min(dx, dy) / U_max
        # Assume U_max ~ 2 * U_inlet for safety
        cfl_dt = 0.5 * min(dx, dy) / (2.0 * self.inlet_velocity)

        # Diffusion stability: dt <= 0.25 * min(dx^2, dy^2) / nu
        nu = self.get_viscosity(reynolds_number)
        diff_dt = 0.25 * min(dx**2, dy**2) / nu

        dt = min(cfl_dt, diff_dt)

        return dt

    def _compute_conservation_diagnostics(self, mesh):
        """
        Compute mass-conservation diagnostics for the FDM solution.

        The two-dimensional flow rate is evaluated as the line integral of the
        horizontal velocity over the inlet and outlet boundaries.
        """
        inlet_mask = (mesh.cell_type == 2)
        outlet_mask = (mesh.cell_type == 3)

        def boundary_flux(mask):
            if not np.any(mask):
                return None, None
            u_values = mesh.u[mask]
            return float(np.sum(u_values) * mesh.dy), float(np.mean(u_values))

        flux_in, mean_u_in = boundary_flux(inlet_mask)
        flux_out, mean_u_out = boundary_flux(outlet_mask)
        flux_in_target = self.inlet_velocity * self.reference_length

        flux_imbalance = None
        flux_imbalance_relative = None
        if flux_in is not None and flux_out is not None:
            flux_imbalance = flux_out - flux_in
            flux_imbalance_relative = abs(flux_imbalance) / max(abs(flux_in), 1e-14)

        div_l2 = None
        div_max = None
        if mesh.nx > 2 and mesh.ny > 2:
            interior = np.s_[1:-1, 1:-1]
            fluid = (mesh.cell_type[interior] == 0)
            if np.any(fluid):
                divergence = (
                    (mesh.u[1:-1, 2:] - mesh.u[1:-1, :-2]) / (2 * mesh.dx)
                    + (mesh.v[2:, 1:-1] - mesh.v[:-2, 1:-1]) / (2 * mesh.dy)
                )
                div_values = divergence[fluid]
                div_l2 = float(np.sqrt(np.mean(div_values**2)))
                div_max = float(np.max(np.abs(div_values)))

        return {
            'flux_in': flux_in,
            'flux_in_target': flux_in_target,
            'flux_out': flux_out,
            'flux_imbalance': flux_imbalance,
            'flux_imbalance_relative': flux_imbalance_relative,
            'mean_u_inlet': mean_u_in,
            'mean_u_outlet': mean_u_out,
            'divergence_l2': div_l2,
            'divergence_max': div_max,
        }

    def run_single_case(self, reynolds_number, nx, ny, max_time=20.0,
                        spatial_scheme='中心差分'):
        case_name = f"Re{reynolds_number}_N{nx}x{ny}"
        print("\n" + "=" * 70)
        print(f"Running case: {case_name}")
        print("=" * 70)

        mesh = self.create_mesh(nx, ny)

        nu = self.get_viscosity(reynolds_number)
        dt = self.get_time_step(nx, ny, reynolds_number)

        flow_params = {
            'reynolds': reynolds_number,
            'viscosity': nu,
            'dt': dt,
            'inlet_velocity': self.inlet_velocity
        }

        solver_params = {
            'spatial_scheme': spatial_scheme,
            'temporal_scheme': '显式欧拉',
            'convergence_tolerance': self.convergence_tolerance,
            'cd_convergence_tolerance': self.cd_convergence_tolerance,
            'cd_convergence_window_time': self.cd_convergence_window_time,
            'max_iterations': 100000,
            'check_convergence_interval': 10
        }

        solver = NavierStokesSolver(mesh, flow_params, solver_params)

        output_interval = 100
        result = solver.run_simulation(
            total_time=max_time,
            output_interval=output_interval,
            convergence_check=True
        )

        history_file = os.path.join(self.output_dir, 'history', f'{case_name}_history.csv')
        solver.export_history(history_file)

        max_u = np.max(np.abs(mesh.u))
        max_v = np.max(np.abs(mesh.v))
        max_p = np.max(mesh.p)
        min_p = np.min(mesh.p)

        case_result = {
            'case_name': case_name,
            'reynolds': reynolds_number,
            'nx': nx,
            'ny': ny,
            'dx': mesh.dx,
            'dy': mesh.dy,
            'dt': dt,
            'nu': nu,
            'reference_length': self.reference_length,
            'inlet_velocity': self.inlet_velocity,
            'converged': result['converged'],
            'diverged': result['diverged'],
            'final_time': result['final_time'],
            'final_steps': result['final_steps'],
            'final_residual': result['final_residual'],
            'convergence_reason': result.get('convergence_reason', ''),
            'final_cd_relative_change': result.get('final_cd_relative_change'),
            'C_d': result['final_C_d'],
            'C_l': result['final_C_l'],
            'max_u': max_u,
            'max_v': max_v,
            'max_p': max_p,
            'min_p': min_p
        }

        case_result.update(self._compute_conservation_diagnostics(mesh))

        self.results_summary.append(case_result)

        print(f"\nCase {case_name} completed successfully")
        if case_result['flux_imbalance_relative'] is not None:
            print(f"  Mass flux in/out: {case_result['flux_in']:.6f} / "
                  f"{case_result['flux_out']:.6f}")
            print(f"  Relative flux imbalance: "
                  f"{100 * case_result['flux_imbalance_relative']:.3f}%")

        return case_result

    def run_all_cases(self):
        """
        Run the selected batch cases from the experiment plan

        Reynolds numbers: 20, 200
        Grid resolutions: 80x32, 120x48, 160x64, 200x80
        """
        reynolds_numbers = [20, 200]
        grid_resolutions = [(80, 32), (120, 48), (160, 64), (200, 80)]

        total_cases = len(reynolds_numbers) * len(grid_resolutions)
        current_case = 0

        print("\n" + "=" * 70)
        print("BATCH EXPERIMENT: Reynolds Number Study")
        print("=" * 70)
        print(f"Total cases to run: {total_cases}")
        print(f"Reynolds numbers: {reynolds_numbers}")
        print(f"Grid resolutions: {grid_resolutions}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 70)

        start_time = datetime.now()

        for Re in reynolds_numbers:
            for nx, ny in grid_resolutions:
                current_case += 1
                print(f"\n>>> Progress: {current_case}/{total_cases} <<<")

                try:
                    self.run_single_case(Re, nx, ny)
                except Exception as e:
                    print(f"ERROR in case Re={Re}, grid={nx}x{ny}: {e}")
                    import traceback
                    traceback.print_exc()

                    self.results_summary.append({
                        'case_name': f"Re{Re}_N{nx}x{ny}",
                        'reynolds': Re,
                        'nx': nx,
                        'ny': ny,
                        'converged': False,
                        'diverged': True,
                        'error': str(e)
                    })

        end_time = datetime.now()
        elapsed = end_time - start_time

        print("\n" + "=" * 70)
        print("BATCH EXPERIMENT COMPLETED")
        print("=" * 70)
        print(f"Total time: {elapsed}")
        print(f"Successful cases: {sum(1 for r in self.results_summary if r.get('converged', False))}")
        print(f"Failed cases: {sum(1 for r in self.results_summary if r.get('diverged', False))}")

        self.export_summary()
        self.summarize_nonconverged_cases()

    def export_summary(self):
        summary_file = os.path.join(self.output_dir, 'summary_all_cases.csv')

        with open(summary_file, 'w', newline='') as f:
            if not self.results_summary:
                return

            fieldnames = sorted({
                key
                for result in self.results_summary
                for key in result.keys()
            })
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for result in self.results_summary:
                writer.writerow(result)

        print(f"\nSummary exported to: {summary_file}")

    def _get_status_label(self, result):
        if result.get('diverged', False):
            return 'diverged'
        if result.get('converged', False):
            return 'converged'
        return 'not_converged'

    def _build_reference_cases(self):
        grouped = defaultdict(list)

        for result in self.results_summary:
            if result.get('converged', False):
                grouped[result['reynolds']].append(result)

        references = {}
        for reynolds, cases in grouped.items():
            references[reynolds] = min(cases, key=lambda r: r.get('dx', float('inf')))

        return references

    def _relative_error(self, value, reference_value):
        if value is None or reference_value is None:
            return None
        if abs(reference_value) < 1e-14:
            return None
        return abs(value - reference_value) / abs(reference_value)

    def summarize_nonconverged_cases(self):
        """
        Summarize which cases did not converge and estimate their errors
        """
        print("\n" + "=" * 70)
        print("NON-CONVERGED CASE SUMMARY")
        print("=" * 70)

        references = self._build_reference_cases()
        nonconverged_rows = []

        for result in self.results_summary:
            status = self._get_status_label(result)
            if status == 'converged':
                continue

            reynolds = result.get('reynolds')
            reference = references.get(reynolds)
            final_residual = result.get('final_residual')
            residual_ratio = None
            if final_residual is not None:
                residual_ratio = final_residual / self.convergence_tolerance

            row = {
                'case_name': result.get('case_name'),
                'status': status,
                'reynolds': reynolds,
                'nx': result.get('nx'),
                'ny': result.get('ny'),
                'final_time': result.get('final_time'),
                'final_steps': result.get('final_steps'),
                'final_residual': final_residual,
                'residual_over_tolerance': residual_ratio,
                'reference_case': reference.get('case_name') if reference else '',
                'error_C_d': self._relative_error(result.get('C_d'), reference.get('C_d')) if reference else None,
                'error_C_l': self._relative_error(result.get('C_l'), reference.get('C_l')) if reference else None,
                'error_max_u': self._relative_error(result.get('max_u'), reference.get('max_u')) if reference else None,
                'error_max_v': self._relative_error(result.get('max_v'), reference.get('max_v')) if reference else None,
                'error_max_p': self._relative_error(result.get('max_p'), reference.get('max_p')) if reference else None,
                'error_min_p': self._relative_error(result.get('min_p'), reference.get('min_p')) if reference else None,
                'error_message': result.get('error', '')
            }
            nonconverged_rows.append(row)

        if not nonconverged_rows:
            print("All cases converged within the specified tolerance.")
            return

        for row in nonconverged_rows:
            print(f"\nCase: {row['case_name']}")
            print(f"  Status: {row['status']}")
            if row['final_residual'] is not None:
                print(f"  Final residual: {row['final_residual']:.3e}")
            if row['residual_over_tolerance'] is not None:
                print(f"  Residual / tolerance: {row['residual_over_tolerance']:.2f}")
            if row['reference_case']:
                print(f"  Reference case: {row['reference_case']}")
                for key in ['error_C_d', 'error_C_l', 'error_max_u', 'error_max_v', 'error_max_p', 'error_min_p']:
                    value = row[key]
                    if value is not None:
                        print(f"  {key}: {value:.3%}")
            if row['error_message']:
                print(f"  Error: {row['error_message']}")

        output_file = os.path.join(self.output_dir, 'summary_nonconverged_cases.csv')
        with open(output_file, 'w', newline='') as f:
            fieldnames = list(nonconverged_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(nonconverged_rows)

        print(f"\nNon-converged case summary exported to: {output_file}")


def main():
    manager = BatchExperimentManager(output_dir='experiment_results')
    manager.run_all_cases()

    print("\n" + "=" * 70)
    print("All experiments completed!")
    print("Check the 'experiment_results' directory for outputs")
    print("=" * 70)


if __name__ == '__main__':
    main()
