from sailboat.rpa import RPA, Plasma, sim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path

def rays(
        rays: np.ndarray,
        ray_rates: np.ndarray,
        rpa: RPA,
        plasma: Plasma,
        filename: str = 'rays.png',
        plot_direc: Path = Path('.'),
        ) -> None:

    # geometry
    ax = rpa.aperture[0] / 2
    ay = rpa.aperture[1] / 2
    sx = rpa.sensor[0] / 2
    sy = rpa.sensor[1] / 2
    d = rpa.depth
    xticks = rpa.get_locations()
    max_size = 1.05 * max(2 * sy, d)

    currents = sim.get_currents(rays, ray_rates, [0.0, rpa.screens[1].location, rpa.depth]) * plasma.Q # nanoamperes

    plt.style.use('dark_background')
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(22, 10), gridspec_kw={'width_ratios': [4, 4, 1]})
    assert(isinstance(ax0, Axes))
    assert(isinstance(ax1, Axes))
    assert(isinstance(ax2, Axes))

    # rpa axes
    for ray in rays:
        ax0.plot(ray[:, 2], ray[:, 1], color='w', alpha=0.9, linewidth=0.1)

    # draw geometry
    ax0.plot([-1, 1, np.nan, -1, 1], [ay, ay, np.nan, -ay, -ay], color='r', linewidth=5) # aperture
    ax0.plot([0, 0, d, d, 0, 0], [ay, sy, sy, -sy, -sy, -ay], color='y', linewidth=5) # enclosure
    ax0.plot([d, d], [-sy, sy], color='g', linewidth=5) # sensor

    ax0t = ax0.twiny()

    ax0.set_xlabel('z (mm)')
    ax0.set_ylabel('y (mm)')
    ax0t.set_xlabel('voltage (V)')

    ax0.set_xticks(xticks)
    ax0t.set_xticks(xticks, rpa.get_voltages())

    ax0.set_xlim((d - max_size) / 2, (d + max_size) / 2)
    ax0t.set_xlim(ax0.get_xlim())
    ax0.set_ylim(-max_size / 2, max_size / 2)

    ax0.grid()

    # sensor plot and histogram
    sensor_hits = sim.collect_punctures(rays, rpa.depth, get_coords=True)
    weights = 100 * ray_rates[sensor_hits[:, 0].astype(int)]  / np.sum(ray_rates)
    num_bins = 64

    ax1.scatter(sensor_hits[:, 0], sensor_hits[:, 1], s=sensor_hits[:, 2] * 0.01, color='w', alpha=0.7)
    ax1.plot([-ax, ax, ax, -ax, -ax], [-ay, -ay, ay, ay, -ay], color='r', linewidth=5, alpha=0.5) # aperture
    ax1.plot([-sx, sx, sx, -sx, -sx], [-sy, -sy, sy, sy, -sy], color='g', linewidth=5) # sensor
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.set_xlim(-max_size / 2, max_size / 2)
    ax1.set_ylim(-max_size / 2, max_size / 2)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')

    ax2.hist(sensor_hits[:, 1], bins=num_bins, weights=weights, orientation='horizontal', color='w')
    ax2.set_xlabel('%')
    # ax2.set_xlim(0, 6)
    ax2.set_ylim(-max_size / 2, max_size / 2)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    ax2.grid()

    fig.suptitle(
        filename + 
        f' — Beam velocity: ({plasma.V[0]:.1f}, {plasma.V[1]:.1f}) km/s' + 
        f' — Temperature: {plasma.T:.1f} eV' + 
        f' — Number of rays: {len(rays)}' + 
        f' — Currents: (aperture, bias, sensor) = ({currents[0]:.2f}, {currents[1]:.2f}, {currents[2]:.2f}) nA'
        )
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0)
    fig.show()

    if not plot_direc:
        plot_direc = Path(__file__).parent / 'ray_plots'
    plot_direc.mkdir(exist_ok=True)
    print(plot_direc / filename)
    fig.savefig(plot_direc / filename)

    plt.close(fig)


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