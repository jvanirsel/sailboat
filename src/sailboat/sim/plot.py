from os import getenv, path
from gemini3d import read, utils
from sail_utils import md, dipole_alt_slice, dipole_glon_slice, si_units, make_gif
import numpy as np
import matplotlib.pyplot as plt

def plot(
        sim_name: str,
        variable: str,
        dat_order: int = 0,
        alt_ref: float = 300e3,
        alt_res: float = 20e3,
        glon_ref: float = None,
        glon_res: float = 1,
        mlon_ref: float = None,
        make_gifs: bool = True
        ):
    sim_root = getenv('GEMINI_SIM_ROOT')
    sim_path = path.join(sim_root, sim_name)
    plot_direc = md(path.join(sim_path, 'plots'))

    cfg = read.config(sim_path)
    xg = read.grid(sim_path)
    times = cfg['time']

    scl = 10**dat_order
    units = si_units(variable, dat_order)

    if not glon_ref:
        glon_ref = cfg['glon']
    if not mlon_ref:
        mlon_ref = xg['x3'][2 + xg['lx'][2] // 2] * 180 / np.pi

    make_gif(plot_direc, suffix='mlon=334deg.png')
    for time in times:
        time_str = utils.datetime2stem(time).replace(' ', '0')
        continue
        dat = read.frame(sim_path, time, var=variable)[variable] / scl

        plot_name = f'{variable}_{time_str}_alt={alt_ref / 1e3:.0f}km.png'
        plot_path = path.join(plot_direc, plot_name)
        title = f'{time_str} (alt = {alt_ref / 1e3:.0f}+/-{alt_res / 1e3:.0f} km)'
        dat_label = f'{variable} ({units})'
        # plot_alt_slice(xg, dat, alt_ref, alt_res, plot_path, title, dat_label)

        plot_name = f'{variable}_{time_str}_glon={glon_ref:.0f}deg.png'
        plot_path = path.join(plot_direc, plot_name)
        title = f'{time_str} (glon = {glon_ref:.0f}+/-{glon_res:.0f}°)'
        dat_label = f'{variable} ({units})'
        # plot_glon_slice(xg, dat, glon_ref, glon_res, alt_ref, plot_path, title, dat_label)

        plot_name = f'{variable}_{time_str}_mlon={mlon_ref:.0f}deg.png'
        plot_path = path.join(plot_direc, plot_name)
        title = f'{time_str} (mlon = {mlon_ref:.0f}°)'
        dat_label = f'{variable} ({units})'
        # plot_mlon_slice(xg, dat, mlon_ref, alt_ref, plot_path, title, dat_label)
    # if make_gifs:
        # make_gif(plot_direc, suffix=plot_name[-15:])

def plot_alt_slice(
        xg: dict,
        dat: np.ndarray,
        alt_ref: float,
        alt_res: float,
        plot_path: str,
        title: str,
        dat_label: str
        ):

    dat_slice, glon, glat = dipole_alt_slice(xg, dat, alt_ref, alt_res=alt_res)
    plt.pcolormesh(glon, glat, dat_slice, shading='auto')
    clb = plt.colorbar()
    clb.set_label(dat_label)
    plt.xlabel('geog lon (°)')
    plt.ylabel('geog lat (°)')
    plt.title(title)
    plt.show()
    print(f'Saving {plot_path}...')
    plt.savefig(plot_path)
    plt.close()


def plot_glon_slice(
        xg: dict,
        dat: np.ndarray,
        glon_ref: float,
        glon_res: float,
        alt_ref: float,
        plot_path: str,
        title: str,
        dat_label: str
        ):

    dat_slice, glat, alt = dipole_glon_slice(xg, dat, glon_ref, glon_res=glon_res)
    plt.pcolormesh(glat, alt / 1e3, dat_slice, shading='auto')
    clb = plt.colorbar()
    clb.set_label(dat_label)
    plt.xlabel('geog lat (°)')
    plt.ylabel('altitude (km)')
    plt.title(title)
    plt.ylim([0, 2 * alt_ref / 1e3])
    plt.show()
    print(f'Saving {plot_path}...')
    plt.savefig(plot_path)
    plt.close()


def plot_mlon_slice(
        xg: dict,
        dat: np.ndarray,
        mlon_ref: float,
        alt_ref: float,
        plot_path: str,
        title: str,
        dat_label: str,
        freeze_clim: bool = True
        ):
    x3 = xg['x3'][2:-2]
    x3id = np.argmin(np.abs(x3 - mlon_ref * np.pi / 180))
    dat_slice = dat[:, :, x3id]
    glat = xg['glat'][:, :, x3id]
    alt = xg['alt'][:, :, x3id]

    plt.pcolormesh(glat, alt / 1e3, dat_slice, shading='auto')
    clb = plt.colorbar()
    clb.set_label(dat_label)
    plt.xlabel('geog lat (°)')
    plt.ylabel('altitude (km)')
    plt.title(title)
    plt.ylim([0, 2 * alt_ref / 1e3])
    plt.clim([0, 3e12])
    plt.show()
    print(f'Saving {plot_path}...')
    plt.savefig(plot_path)
    plt.close()

if __name__ == "__main__":
    import sys
    plot(sys.argv[1], sys.argv[2])