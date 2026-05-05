import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt


class FiniteDifferenceMesh:
    def __init__(self, nx, ny, domain_bounds):
        self.nx = nx
        self.ny = ny
        self.xmin, self.xmax, self.ymin, self.ymax = domain_bounds

        self.dx = (self.xmax - self.xmin) / (nx - 1)
        self.dy = (self.ymax - self.ymin) / (ny - 1)

        self.x = np.linspace(self.xmin, self.xmax, nx)
        self.y = np.linspace(self.ymin, self.ymax, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # 0: fluid domain, 1: solid obstacle, 2: inlet, 3: outlet, 4: wall
        self.cell_type = np.zeros((ny, nx), dtype=int)

        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))

    def mark_obstacle(self, polygon_points):
        from matplotlib.path import Path

        polygon = Path(polygon_points)

        for j in range(self.ny):
            for i in range(self.nx):
                point = [self.X[j, i], self.Y[j, i]]
                if polygon.contains_point(point):
                    self.cell_type[j, i] = 1

    def mark_inlet(self, polygon_points):
        from matplotlib.path import Path
        polygon = Path(polygon_points)

        for j in range(self.ny):
            for i in range(self.nx):
                point = [self.X[j, i], self.Y[j, i]]
                if polygon.contains_point(point):
                    self.cell_type[j, i] = 2

    def mark_outlet(self, polygon_points):
        from matplotlib.path import Path
        polygon = Path(polygon_points)

        for j in range(self.ny):
            for i in range(self.nx):
                point = [self.X[j, i], self.Y[j, i]]
                if polygon.contains_point(point):
                    self.cell_type[j, i] = 3

    def set_boundary_conditions(self, inlet_velocity=1.0):
        inlet_mask = (self.cell_type == 2)
        self.u[inlet_mask] = inlet_velocity
        self.v[inlet_mask] = 0.0

        obstacle_mask = (self.cell_type == 1)
        self.u[obstacle_mask] = 0.0
        self.v[obstacle_mask] = 0.0

    def visualize_mesh(self, show_velocity=False):
        fig, ax = plt.subplots(figsize=(10, 8))

        cmap = plt.cm.colors.ListedColormap(['white', 'gray', 'green', 'red'])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

        im = ax.pcolormesh(self.X, self.Y, self.cell_type, cmap=cmap, norm=norm,
                          edgecolors='lightgray', linewidth=0.5)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label='Fluid Domain'),
            Patch(facecolor='gray', edgecolor='black', label='Obstacle'),
            Patch(facecolor='green', edgecolor='black', label='Inlet'),
            Patch(facecolor='red', edgecolor='black', label='Outlet')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        if show_velocity:
            skip = max(1, self.nx // 20)
            ax.quiver(self.X[::skip, ::skip], self.Y[::skip, ::skip],
                     self.u[::skip, ::skip], self.v[::skip, ::skip],
                     alpha=0.6, color='blue')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Finite Difference Mesh ({self.nx} × {self.ny})')
        ax.set_aspect('equal')
        plt.tight_layout()

        return fig, ax

    def export_to_file(self, filename):
        data = {
            'grid_info': {
                'nx': self.nx,
                'ny': self.ny,
                'dx': self.dx,
                'dy': self.dy,
                'bounds': [self.xmin, self.xmax, self.ymin, self.ymax]
            },
            'coordinates': {
                'x': self.x.tolist(),
                'y': self.y.tolist()
            },
            'cell_type': self.cell_type.tolist(),
            'initial_conditions': {
                'u': self.u.tolist(),
                'v': self.v.tolist(),
                'p': self.p.tolist()
            }
        }

        import json
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Mesh data exported to: {filename}")


def generate_mesh_from_config(config):
    nx = config['grid']['nx']
    ny = config['grid']['ny']

    all_points = []
    for shape in config['shapes']:
        if shape['type'] == 'domain':
            all_points.extend(shape['points'])

    if not all_points:
        raise ValueError("Domain definition not found!")

    all_points = np.array(all_points)
    xmin, ymin = all_points.min(axis=0)
    xmax, ymax = all_points.max(axis=0)

    margin = 0.05
    dx_margin = (xmax - xmin) * margin
    dy_margin = (ymax - ymin) * margin
    domain_bounds = [xmin - dx_margin, xmax + dx_margin,
                     ymin - dy_margin, ymax + dy_margin]

    mesh = FiniteDifferenceMesh(nx, ny, domain_bounds)

    # Obstacle markings are applied last so they are not overwritten.

    temp_markers = []
    for shape in config['shapes']:
        points = np.array(shape['points'])
        temp_markers.append((shape['type'], points))

    for shape_type, points in temp_markers:
        if shape_type == 'inlet':
            mesh.mark_inlet(points)
        elif shape_type == 'outlet':
            mesh.mark_outlet(points)

    for shape_type, points in temp_markers:
        if shape_type == 'obstacle':
            mesh.mark_obstacle(points)

    obstacle_mask = (mesh.cell_type == 1)
    inlet_count = np.sum(mesh.cell_type == 2)
    outlet_count = np.sum(mesh.cell_type == 3)
    obstacle_count = np.sum(obstacle_mask)

    print(f"\nMesh region statistics:")
    print(f"  Inlet cells: {inlet_count}")
    print(f"  Outlet cells: {outlet_count}")
    print(f"  Obstacle cells: {obstacle_count}")
    print(f"  Fluid cells: {np.sum(mesh.cell_type == 0)}")

    if obstacle_count == 0:
        print("  WARNING: No obstacle cells found!")
    if inlet_count == 0:
        print("  WARNING: No inlet cells found!")
    if outlet_count == 0:
        print("  WARNING: No outlet cells found!")

    inlet_velocity = config['flow_params']['inlet_velocity']
    mesh.set_boundary_conditions(inlet_velocity)

    return mesh


if __name__ == '__main__':
    print("Mesh generation module test")

    mesh = FiniteDifferenceMesh(50, 30, [0, 10, 0, 6])

    theta = np.linspace(0, 2*np.pi, 50)
    obstacle_points = np.column_stack([
        5 + 0.5 * np.cos(theta),
        3 + 0.5 * np.sin(theta)
    ])
    mesh.mark_obstacle(obstacle_points)

    inlet_points = [[0, 2], [0, 4], [0.5, 4], [0.5, 2]]
    mesh.mark_inlet(inlet_points)

    outlet_points = [[9.5, 2], [9.5, 4], [10, 4], [10, 2]]
    mesh.mark_outlet(outlet_points)

    mesh.set_boundary_conditions(inlet_velocity=1.0)

    fig, ax = mesh.visualize_mesh(show_velocity=True)
    plt.savefig('test_mesh.png', dpi=150)
    print("Test mesh saved to test_mesh.png")
