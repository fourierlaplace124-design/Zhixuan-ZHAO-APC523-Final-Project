import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def compute_omega_central(kx, ky, dx, dy, u, v, nu):
    omega_conv_x = -1j * u * np.sin(kx * dx) / dx
    omega_conv_y = -1j * v * np.sin(ky * dy) / dy

    omega_diff_x = -nu * 2 * (1 - np.cos(kx * dx)) / dx**2
    omega_diff_y = -nu * 2 * (1 - np.cos(ky * dy)) / dy**2

    omega = omega_conv_x + omega_conv_y + omega_diff_x + omega_diff_y

    return omega

def compute_omega_upwind(kx, ky, dx, dy, u, v, nu):
    if u >= 0:
        omega_conv_x = -u * (1 - np.cos(kx * dx)) / dx - 1j * u * np.sin(kx * dx) / dx
    else:
        omega_conv_x = -u * (1 - np.cos(-kx * dx)) / dx - 1j * u * np.sin(-kx * dx) / dx

    if v >= 0:
        omega_conv_y = -v * (1 - np.cos(ky * dy)) / dy - 1j * v * np.sin(ky * dy) / dy
    else:
        omega_conv_y = -v * (1 - np.cos(-ky * dy)) / dy - 1j * v * np.sin(-ky * dy) / dy
    omega_diff_x = -nu * 2 * (1 - np.cos(kx * dx)) / dx**2
    omega_diff_y = -nu * 2 * (1 - np.cos(ky * dy)) / dy**2

    omega = omega_conv_x + omega_conv_y + omega_diff_x + omega_diff_y

    return omega

def plot_stability_region_complex_plane(scheme='central', n_grid=300, n_k=50):
    real_range = np.linspace(-3, 1, n_grid)
    imag_range = np.linspace(-2, 2, n_grid)
    Real, Imag = np.meshgrid(real_range, imag_range)

    omega_dt = Real + 1j * Imag
    rho = 1 + omega_dt
    rho_mag = np.abs(rho)

    fig, ax = plt.subplots(figsize=(8, 8))

    levels = np.linspace(0, 2, 21)
    contourf = ax.contourf(Real, Imag, rho_mag, levels=levels, cmap='viridis')

    ax.contour(Real, Imag, rho_mag, levels=[1.0], colors='red', linewidths=3, linestyles='-')

    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('$|\\rho|$', fontsize=12)

    circle = Circle((-1, 0), 1, fill=False, edgecolor='red', linewidth=3, linestyle='-', label='$|\\rho|=1$ (stability boundary)')
    ax.add_patch(circle)
    ax.plot(-1, 0, 'r*', markersize=15, markeredgecolor='white', markeredgewidth=1, label='Stability center (-1, 0)', zorder=10)

    h = 0.1
    dx = dy = h
    u = 1.0
    v = 0.0
    nu = 0.05
    dt = 0.25 * h**2 / nu
    k_range = np.linspace(-np.pi, np.pi, n_k)
    omega_dt_samples = []
    for kx in k_range:
        for ky in k_range:
            if scheme == 'central':
                omega = compute_omega_central(kx, ky, dx, dy, u, v, nu)
            else:
                omega = compute_omega_upwind(kx, ky, dx, dy, u, v, nu)

            omega_dt_samples.append(omega * dt)

    omega_dt_samples = np.array(omega_dt_samples)
    ax.scatter(omega_dt_samples.real, omega_dt_samples.imag, c='cyan', s=1, alpha=0.5, label=f'$\\Omega\\Delta t$ samples ({scheme})')
    ax.set_xlabel('$\\mathrm{Re}(\\Omega\\Delta t)$', fontsize=12)
    ax.set_ylabel('$\\mathrm{Im}(\\Omega\\Delta t)$', fontsize=12)
    ax.set_title(f'Explicit Euler Stability Region\n({scheme} difference)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 1)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')

    plt.tight_layout()

    return fig, ax

def plot_cfl_fourier_stability():
    fig, ax = plt.subplots(figsize=(10, 7))

    CFL = np.linspace(0, 1.5, 200)
    Fo_central = (2 - CFL**2/2) / 4
    Fo_central = np.maximum(0, np.minimum(Fo_central, 0.5))
    Fo_upwind = (3 - CFL) / 4
    Fo_upwind = np.maximum(0, np.minimum(Fo_upwind, 0.5))
    ax.fill_between(CFL, 0, Fo_central, alpha=0.3, color='blue', label='Central difference (stable)')
    ax.plot(CFL, Fo_central, 'b-', linewidth=2)
    ax.fill_between(CFL, 0, Fo_upwind, alpha=0.2, color='red', label='Upwind (stable)')
    ax.plot(CFL, Fo_upwind, 'r--', linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, label='Fo = 0.5 (diffusion limit)')
    grids = [(80, 32), (160, 64), (320, 128)]
    domain = [0, 10, 0, 4]
    inlet_velocity = 1.0
    nu = 0.05

    cfl_actual = []
    fo_actual = []

    for nx, ny in grids:
        dx = (domain[1] - domain[0]) / nx
        dy = (domain[3] - domain[2]) / ny
        h = min(dx, dy)
        cfl_dt = 0.5 * h / (2.0 * inlet_velocity)
        diff_dt = 0.25 * h**2 / nu
        dt = min(cfl_dt, diff_dt)
        u_max = 2.0 * inlet_velocity
        CFL_val = u_max * dt / h
        Fo_val = nu * dt / h**2
        cfl_actual.append(CFL_val)
        fo_actual.append(Fo_val)

    ax.plot(cfl_actual, fo_actual, 'ko', markersize=10, label='Re=20 simulations', zorder=10)
    grid_labels = ['80×32', '160×64', '320×128']
    for i, (cfl, fo) in enumerate(zip(cfl_actual, fo_actual)):
        ax.annotate(grid_labels[i], (cfl, fo), xytext=(8, 8), textcoords='offset points',
                   fontsize=10, bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8))
    ax.set_xlabel('CFL = $|u|\\Delta t / h$', fontsize=13)
    ax.set_ylabel('Fo = $\\nu \\Delta t / h^2$', fontsize=13)
    ax.set_title('Stability Regions: Explicit Euler Time Integration', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 0.55)

    plt.tight_layout()

    return fig, ax

if __name__ == '__main__':
    fig1, ax1 = plot_stability_region_complex_plane('central', n_grid=400, n_k=30)
    fig1.savefig('figures/stability_complex_plane_central.pdf', dpi=300, bbox_inches='tight')
    fig1.savefig('figures/stability_complex_plane_central.png', dpi=150, bbox_inches='tight')

    fig2, ax2 = plot_cfl_fourier_stability()
    fig2.savefig('figures/stability_region_cfl_fourier.pdf', dpi=300, bbox_inches='tight')
    fig2.savefig('figures/stability_region_cfl_fourier.png', dpi=150, bbox_inches='tight')

    plt.show()
