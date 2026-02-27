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
        plasma: Plasma,
        do_electrons: bool = False,
        debug: bool = False
        ) -> None:

    from . import sim

    # geometry
    xa = rpa.aperture[0] / 2
    ya = rpa.aperture[1] / 2
    xs = rpa.sensor[0] / 2
    ys = rpa.sensor[1] / 2
    zd = rpa.depth
    max_size = 1.5 * max(2 * xs, 2 * ys, zd)

    # floating potential current
    if rpa.floating_potential:
        sgnV = abs(rpa.floating_potential) / rpa.floating_potential
        jz_floating = -sgnV * (plasma.N[0] + plasma.N[1]) * plasma.Q * (2 * abs(rpa.floating_potential) / plasma.Mi)**0.5
    else:
        jz_floating = 0
        # print(jz_floating)
    # print(plasma.jz)
    jz = plasma.jz + jz_floating

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

    for ray in rays[::max(len(rays) // 1000, 1)]:
        ray = ray[~np.isnan(ray[:, 0]), :]
        ax.plot(ray[:, 2], ray[:, 1], color='w', alpha=0.9, linewidth=0.1)
        if debug:
            ax.scatter(ray[::10, 2], ray[::10, 1], color='w', alpha=0.9, marker='.', s=1, linewidths=0)
            ax.scatter(ray[-1, 2], ray[-1, 1], color='r', marker='.', s=3,  linewidths=1)
    dy = 0
    for screen in rpa.screens:
        z = screen.location
        ax.plot([z, z], [-ys, ys], color='w', linestyle='--', linewidth=1) # screens
        z_unit = ' mm' if z == 0.0 else ''
        v_unit = ' V' if z == 0.0 else ''
        ax.text(z, ys / 2 + max_size / 4 - dy, f'{screen.voltage:.1f}{v_unit}', fontsize=8, ha='center', va='center', rotation=90)
        ax.text(z, -ys / 2 - max_size / 4 + dy, f'{screen.location:.1f}{z_unit}', fontsize=8, ha='center', va='center', rotation=-90)
    ax.plot([zd, zd], [-ys, ys], color='g', linestyle='--', linewidth=1) # sensor
    ax.plot([0, 0, zd + 1, zd + 1, 0, 0], [ya, ys, ys, -ys, -ys, -ya], color='y', linewidth=5) # enclosure
    ax.plot([-1, 1, np.nan, -1, 1], [ya, ya, np.nan, -ya, -ya], color='r', linewidth=5) # aperture
    ax.plot([0, 0, 0, 0, 0], [-max_size, -ys, np.nan, ys, max_size], color='y', linewidth=2) # aperture shield
        
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
    if rpa.is_ivm:
        off = 1.1
        ax.text( xs * off,  ys * off, 'I', color='w', size=15, ha='center', va='center')
        ax.text(-xs * off,  ys * off, 'II', color='r', size=15, ha='center', va='center')
        ax.text(-xs * off, -ys * off, 'III', color='g', size=15, ha='center', va='center')
        ax.text( xs * off, -ys * off, 'IV', color='b', size=15, ha='center', va='center')
        ax.plot([0, 0], [-ys, ys], color='g', linewidth=2) # sensor split lines
        ax.plot([-xs, xs], [0, 0], color='g', linewidth=2) # sensor split lines

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
    bin_centers, current_density = hits_histogram(sensor_hits, plasma, ray_rates, 1, do_electrons=do_electrons)
    max_current_density = max(rpa.aperture) * jz
    if do_electrons:
        max_current_density /= -plasma.Z

    ax.plot(current_density, bin_centers, color='w')

    ax.set_xlabel('Current (nA / mm)')
    ax.set_ylabel(f'y (mm)')
    ax.set_xlim(-0.1 * max_current_density, 1.2 * max_current_density)
    ax.set_ylim(-max_size / 2, max_size / 2)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.grid()

    # bottom histogram
    ax = axs[1, 1]
    assert(isinstance(ax, Axes))
    bin_centers, current_density = hits_histogram(sensor_hits, plasma, ray_rates, 0, do_electrons=do_electrons)

    ax.plot(bin_centers, current_density, color='w')

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Current (nA / mm)')
    ax.set_xlim(-max_size / 2, max_size / 2)
    ax.set_ylim(-0.1 * max_current_density, 1.2 * max_current_density)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.grid()

    # iv-curve
    ax = axs[1, 0]
    assert(isinstance(ax, Axes))
    sweep = rpa.screens[rpa.sweep_screen_id].sweep.voltages
    max_current = rpa.aperture[0] * rpa.aperture[1] * jz
    if rpa.is_ivm:
        max_current_factor = 0.5
    else:
        max_current_factor = 1.0
    if do_electrons:
        max_current /= -plasma.Z

    # dIdV = np.diff(rpa.iv_curve[:, 1]) / np.diff(rpa.iv_curve[:, 0])
    # voltage_centers = (rpa.iv_curve[:-1, 0] + rpa.iv_curve[1:, 0]) / 2

    if rpa.is_ivm:
        ax.plot(rpa.iv_curve[:, 0], rpa.iv_curve[:, 1], color='w', linewidth=3)
        ax.plot(rpa.iv_curve[:, 0], rpa.iv_curve[:, 2], color='r', linewidth=3)
        ax.plot(rpa.iv_curve[:, 0], rpa.iv_curve[:, 3], color='g', linewidth=3)
        ax.plot(rpa.iv_curve[:, 0], rpa.iv_curve[:, 4], color='b', linewidth=3)
    else:
        ax.plot(rpa.iv_curve[:, 0], rpa.iv_curve[:, 1], color='w', linewidth=3)
    # ax.plot(voltage_centers, -dIdV, color='b', linewidth=3)

    ax.set_xlim(min(sweep), max(sweep))
    ax.set_ylim(-0.1 * max_current * max_current_factor, 1.2 * max_current * max_current_factor)
    ax.set_xlabel('Bias voltage (V)')
    ax.set_ylabel('Sensor current (nA)')
    ax.grid()

    # blank
    ax = axs[1, 2]
    assert(isinstance(ax, Axes))
    ax.axis('off')

    fig.suptitle(
        f'Beam vel.: ({plasma.V[0]:.1f}, {plasma.V[1]:.1f}, {plasma.V[2]:.1f}) km s^-1' +
        f' — Beam energy: {plasma.K:.1f} eV' +
        f' — Ion temp.: (beam, bg) = ({plasma.Ti[0]:.1f}, {plasma.Ti[1]:.1f}) eV' +
        f' — Elec. temp.: (beam, bg) = ({plasma.Te[0]:.1f}, {plasma.Te[1]:.1f}) eV' +
        f' — Dens.: (beam, bg) = ({plasma.N[0]}, {plasma.N[1]}) mm^-3\n'
        f'Magnetic field: ({1e6*plasma.B[0]:.1f}, {1e6*plasma.B[1]:.1f}, {1e6*plasma.B[2]:.1f}) uT' +
        f' — Number of rays: {len(rays)}' +
        f' — Measured currents: (aperture, bias, sensor) = ({currents[0]:.1f}, {currents[1]:.1f}, {currents[2]:.1f}) nA' +
        f' — Maximum aperture current: {max_current:.1f} nA'
        )
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.06, top=0.88, wspace=0.07, hspace=0.07)
    fig.show()

    print(f'Saving figure ...', end='\r')
    plot_path.parent.mkdir(exist_ok=True)
    fig.savefig(plot_path, dpi=600)
    print(f'Saved: {plot_path}')

    plt.close(fig)


def hits_histogram(
            hits: np.ndarray,
            plasma: Plasma,
            ray_rates: np.ndarray,
            axis: int,
            num_bins = 64,
            do_electrons: bool = False
        ) -> tuple[np.ndarray, np.ndarray]:

    bin_centers = np.array([np.nan])
    current_density = np.array([np.nan])
    if len(hits) > 0:
        hits_axis = hits[:, axis]
        bins = np.linspace(hits_axis.min(), hits_axis.max(), num_bins)
        current_per_ray = plasma.Q * ray_rates[hits[:, 2].astype(int)] # nanoampere ray^-1
        if do_electrons:
            current_per_ray /= -plasma.Z
        current_per_bin, edges = np.histogram(hits_axis, bins=bins, weights=current_per_ray) # nanoampere bin^-1
        bin_widths = np.diff(edges) # millimeter bin^-1
        if all(bin_widths > 0.0):
            bin_centers = 0.5 * (edges[:-1] + edges[1:]) # millimeter
            current_density = current_per_bin / bin_widths # nanoampere millimeter^-1
    
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