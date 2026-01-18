from . import RPA, Plasma
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path

def rays(
        plot_path: Path,
        rays: np.ndarray,
        ray_rates: np.ndarray,
        currents: np.ndarray,
        rpa: RPA,
        plasmas: Plasma | list[Plasma]
        ) -> None:

    from . import sim

    if isinstance(plasmas, Plasma):
        plasmas = [plasmas]

    # geometry
    xa = rpa.aperture[0] / 2
    ya = rpa.aperture[1] / 2
    xs = rpa.sensor[0] / 2
    ys = rpa.sensor[1] / 2
    zd = rpa.depth
    max_size = 1.5 * max(2 * xs, 2 * ys, zd)

    plt.style.use('dark_background')
    plt.rcParams["grid.alpha"] = 0.2
    plt.grid(True)
    plot_width = 16
    fig, axs = plt.subplots(2, 3,
                            figsize=(plot_width, plot_width * 5/9),
                            gridspec_kw={'width_ratios': [4, 4, 1], 'height_ratios': [4, 1]}
                            )

    # rpa axes
    ax = axs[0, 0]
    assert(isinstance(ax, Axes))

    for ray in rays[::max(len(rays) // 5000, 1)]:
        ray = ray[~np.isnan(ray[:, 0]), :]
        ax.plot(ray[:, 2], ray[:, 1], color='w', alpha=0.9, linewidth=0.1)
        # ax.scatter(ray[-1, 2], ray[-1, 1], color='r', s=1)
    for screen in rpa.screens:
        z = screen.location
        ax.plot([z, z], [-ys, ys], color='w', linestyle='--', linewidth=1) # screens
        z_unit = ' mm' if z == 0.0 else ''
        v_unit = ' V' if z == 0.0 else ''
        ax.text(z, ys / 2 + max_size / 4, f'{screen.voltage:.1f}{v_unit}', fontsize=8, ha='center', va='center', rotation=45)
        ax.text(z, -ys / 2 - max_size / 4, f'{screen.location:.1f}{z_unit}', fontsize=8, ha='center', va='center', rotation=-45)
    ax.plot([0, 0, zd, zd, 0, 0], [ya, ys, ys, -ys, -ys, -ya], color='y', linewidth=5) # enclosure
    ax.plot([-1, 1, np.nan, -1, 1], [ya, ya, np.nan, -ya, -ya], color='r', linewidth=5, alpha = 0.5) # aperture
    ax.plot([0, 0, 0, 0, 0], [-max_size, -ys, np.nan, ys, max_size], color='y', linewidth=2) # aperture shield
    ax.plot([zd, zd], [-ys, ys], color='g', linewidth=5) # sensor
    
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_xlim((zd - max_size) / 2, (zd + max_size) / 2)
    ax.set_ylim(-max_size / 2, max_size / 2)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.grid()

    # sensor scatter plot
    ax = axs[0, 1]
    assert(isinstance(ax, Axes))
    sensor_hits = sim.collect_punctures(rays, rpa.depth, get_coords=True)
    weights = ray_rates[sensor_hits[:, 2].astype(int)]  / np.sum(ray_rates)

    ax.scatter(sensor_hits[:, 0], sensor_hits[:, 1], s=weights * 1e4, color='w', alpha=0.7)
    ax.plot([-xs, xs, xs, -xs, -xs], [-ys, -ys, ys, ys, -ys], color='g', linewidth=5) # sensor
    ax.plot([-xa, xa, xa, -xa, -xa], [-ya, -ya, ya, ya, -ya], color='r', linewidth=5, alpha=0.5) # aperture

    ax.set_xlabel('x (mm)')
    ax.set_xlim(-max_size / 2, max_size / 2)
    ax.set_ylim(-max_size / 2, max_size / 2)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.yaxis.set_ticklabels([])
    ax.grid()

    # right histogram
    ax = axs[0, 2]
    assert(isinstance(ax, Axes))
    bin_centers, current_density = hits_histogram(sensor_hits, plasmas[0], ray_rates, 1)
    max_current_density = max(rpa.aperture) * plasmas[0].jz

    ax.plot(current_density, bin_centers, color='w')

    ax.set_xlabel('Current (nA / mm)')
    ax.set_ylabel(f'y (mm)')
    ax.set_xlim(-0.1 * max_current_density, 1.1 * max_current_density)
    ax.set_ylim(-max_size / 2, max_size / 2)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.grid()

    # bottom histogram
    ax = axs[1, 1]
    assert(isinstance(ax, Axes))
    bin_centers, current_density = hits_histogram(sensor_hits, plasmas[0], ray_rates, 0)

    ax.plot(bin_centers, current_density, color='w')

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Current (nA / mm)')
    ax.set_xlim(-max_size / 2, max_size / 2)
    ax.set_ylim(-0.1 * max_current_density, 1.1 * max_current_density)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.grid()

    # iv-curve
    ax = axs[1, 0]
    assert(isinstance(ax, Axes))
    sweep = rpa.screens[rpa.sweep_screen_id].sweep.voltages
    max_current = rpa.aperture[0] * rpa.aperture[1] * plasmas[0].jz

    ax.plot(rpa.iv_curve[:, 0], rpa.iv_curve[:, 1], color='w', linewidth=3)

    ax.set_xlim(min(sweep), max(sweep))
    ax.set_ylim(-0.1 * max_current, 1.1 * max_current)
    ax.set_xlabel('Bias voltage (V)')
    ax.set_ylabel('Sensor current (nA)')
    ax.grid()

    # blank
    ax = axs[1, 2]
    assert(isinstance(ax, Axes))
    ax.axis('off')

    fig.suptitle(
        f'Beam velocity: ({plasmas[0].V[0]:.1f}, {plasmas[0].V[1]:.1f}, {plasmas[0].V[2]:.1f}) km s^-1' +
        f' — Beam energy: {plasmas[0].K:.1f} eV' +
        f' — Ion temperatures: (beam, background) = ({plasmas[0].Ti:.1f}, {plasmas[1].Ti:.1f}) eV' +
        f' — Densities: (beam, background) = ({plasmas[0].N}, {plasmas[1].N}) mm^-3\n'
        f'Magnetic field: ({1e6*plasmas[0].B[0]:.1f}, {1e6*plasmas[0].B[1]:.1f}, {1e6*plasmas[0].B[2]:.1f}) uT' +
        f' — Number of rays: {len(rays)}' +
        f' — Measured currents: (aperture, bias, sensor) = ({currents[0]:.1f}, {currents[1]:.1f}, {currents[2]:.1f}) nA' +
        f' — Maximum aperture current: {max_current:.1f} nA'
        )
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.06, top=0.88, wspace=0.07, hspace=0.07)
    fig.show()

    print(f'Saving figure...', end='\r')
    plot_path.parent.mkdir(exist_ok=True)
    fig.savefig(plot_path, dpi=500)
    print(f'Saved: {plot_path}')

    plt.close(fig)


def hits_histogram(
            hits: np.ndarray,
            plasma: Plasma,
            ray_rates: np.ndarray,
            axis: int,
            num_bins = 64
        ) -> tuple[np.ndarray, np.ndarray]:

    if len(hits) > 0:
        hits_axis = hits[:, axis]
        bins = np.linspace(hits_axis.min(), hits_axis.max(), num_bins)
        current_per_ray = plasma.Q * ray_rates[hits[:, 2].astype(int)] # nanoampere ray^-1
        current_per_bin, edges = np.histogram(hits_axis, bins=bins, weights=current_per_ray) # nanoampere bin^-1
        bin_widths = np.diff(edges) # millimeter
        bin_centers = 0.5 * (edges[:-1] + edges[1:]) # millimeter
        current_density = current_per_bin / bin_widths # nanoampere millimeter^-1
    else:
        current_density = np.array([np.nan])
        bin_centers = np.array([np.nan])
    
    return bin_centers, current_density


def rays_3d(
        rays: np.ndarray,
        rpa: RPA
        ) -> None:

    # plt.style.use('dark_background')
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    for ray in rays:
        ax.plot(ray[:, 2], ray[:, 0], ray[:, 1], color='k', alpha=0.9, linewidth=0.1)
    
    lim = 20
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_aspect('equal')

    fig.savefig('rays_3d.png')