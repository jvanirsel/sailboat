from sailboat import GEMINI_SIM_ROOT, _RE, utils as su
from gemini3d import read, utils as gu
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.axes import Axes
import xarray as xr
from pathlib import Path
from datetime import datetime
import typing as T

def quick_summary(
        cfg: dict,
        dat: xr.Dataset,
        time: datetime,
        slice_coord: int
        ) -> Path:

    time_str = gu.datetime2stem(time).replace(' ', '0')
    output_flag = cfg.get('flagoutput')
    grid_flag = cfg.get('gridflag')

    if slice_coord not in [1, 2, 3]:
        raise ValueError('slice_coord should be 1, 2, or 3')

    if grid_flag != 1:
        dat.coords['x1'] = dat.coords['x1'] / 1e3
        dat.coords['x2'] = dat.coords['x2'] / 1e3
        dat.coords['x3'] = dat.coords['x3'] / 1e3

    match output_flag:
        case 1:
            num_rows = 7
            num_cols = 4
        case 2:
            num_rows = 5
            num_cols = 2    
        case 3:
            num_rows = 1
            num_cols = 1
        case _:
            raise ValueError('Unknown output flag')

    def plot_slice(ax: Axes,
                   data: xr.DataArray,
                   var_name: str,
                   s: int | None = None
                   ):
        if var_name[0] == 'n':
            data = data.where(data > 0)
            data = xr.apply_ufunc(np.log10, data)
            cb_label = f'log10 {var_name}'
        else:
            _, order = su.cut_order(data)
            data = data / (10 ** order)
            cb_label = f'{var_name} / 1e{order}'
        im = data.plot.imshow(ax=ax, cmap='viridis', add_colorbar=True)
        cb = im.colorbar
        if cb:
            cb.set_label(cb_label)
            cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if s:
            ax.set_title(f'{var_name} (s={s})')
        else:
            ax.set_title(var_name)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 3 * num_rows))
    i = 0
    for v in range(len(dat.data_vars)):
        var_name = list(dat.data_vars)[v]
        if var_name in ['ns', 'vs1', 'Ts']:
            for s in range(6):
                ax = axs[i // num_cols, i % num_cols]
                dat_array: xr.DataArray = dat[var_name]
                var_data = dat_array.sel(species=s)
                data_slice = var_data.isel({f'x{slice_coord}': var_data.shape[slice_coord - 1] // 2})
                plot_slice(ax, data_slice, var_name, s=s)
                i += 1
        elif var_name == 'Phitop':
            ax = axs[i // num_cols, i % num_cols]
            data_slice = dat[var_name]
            plot_slice(ax, data_slice, var_name)
            i += 1
        else:
            ax = axs[i // num_cols, i % num_cols]
            var_data: xr.DataArray = dat[var_name]
            data_slice = var_data.isel({f'x{slice_coord}': var_data.shape[slice_coord - 1] // 2})
            plot_slice(ax, data_slice, var_name)
            i += 1

    plt.tight_layout()
    plot_direc = Path(cfg['nml']).parent / 'plots' / 'summary' / f'all_vars_x{slice_coord}_slice'
    plot_direc.mkdir(parents=True, exist_ok=True)
    plot_path = Path(plot_direc, f'all_x{slice_coord}_slice_{time_str}.png')
    print(f'Saving {plot_path}')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return plot_direc


def variable(
        sim_name: str,
        variable: str,
        dat_order: int = 0,
        alt_ref: float = 300e3,
        alt_res: float = 20e3,
        glon_ref: float | None = None,
        glon_res: float = 0.1,
        mlon_ref: float | None = None,
        make_gifs: bool = True,
        do_glon: bool = False
        ) -> None:
    
    sim_direc = Path(GEMINI_SIM_ROOT, sim_name)
    plot_direc = Path(sim_direc, 'plots', variable)
    plot_direc.mkdir(parents=True, exist_ok=True)

    cfg = read.config(sim_direc)
    xg = read.grid(sim_direc)
    times = cfg['time']

    scl = 10 ** dat_order
    units = su.si_units(variable, dat_order)

    if not mlon_ref:
        mlon_ref_value = xg['x3'][2 + xg['lx'][2] // 2] * 180 / np.pi
    else:
        mlon_ref_value = mlon_ref
    if not glon_ref:
        glon_ref_value = cfg['glon']
    else:
        glon_ref_value = glon_ref

    alt_filename_base = f'{variable}_alt={alt_ref / 1e3:.0f}km'
    mlon_filename_base = f'{variable}_mlon={mlon_ref_value:.0f}deg'
    glon_filename_base = f'{variable}_glon={glon_ref_value:.0f}deg'
    for time in times:
        time_str = gu.datetime2stem(time).replace(' ', '0')
        dat = read.frame(sim_direc, time, var={variable})[variable] / scl

        plot_path = Path(plot_direc, f'{alt_filename_base}_{time_str}.png')
        title = f'{time_str} (alt = {alt_ref / 1e3:.0f}+/-{alt_res / 1e3:.0f} km)'
        dat_label = f'{variable} ({units})'
        plot_alt_slice(xg, dat, alt_ref, alt_res, plot_path, title, dat_label)

        plot_path = Path(plot_direc, f'{mlon_filename_base}_{time_str}.png')
        title = f'{time_str} (mlon = {mlon_ref_value:.0f}°)'
        dat_label = f'{variable} ({units})'
        plot_mlon_slice(xg, dat, mlon_ref_value, alt_ref, plot_path, title, dat_label)

        if do_glon:
            plot_path = Path(plot_direc, f'{glon_filename_base}_{time_str}.png')
            title = f'{time_str} (glon = {glon_ref_value:.0f}+/-{glon_res:.0f}°)'
            dat_label = f'{variable} ({units})'
            plot_glon_slice(xg, dat, glon_ref_value, glon_res, alt_ref, plot_path, title, dat_label)

    if make_gifs:
        su.make_gif(plot_direc, prefix=alt_filename_base)
        su.make_gif(plot_direc, prefix=mlon_filename_base)
        if do_glon:
            su.make_gif(plot_direc, prefix=glon_filename_base)


def plot_alt_slice(
        xg: dict,
        dat: np.ndarray,
        alt_ref: float,
        alt_res: float,
        plot_path: Path,
        title: str,
        dat_label: str
        ) -> None:

    dat_slice, glon, glat = su.dipole_alt_slice(xg, dat, alt_ref, alt_res=alt_res)
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
        plot_path: Path,
        title: str,
        dat_label: str
        ) -> None:

    dat_slice, glat, alt = su.dipole_glon_slice(xg, dat, glon_ref, glon_res=glon_res)
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
        plot_path: Path,
        title: str,
        dat_label: str,
        ) -> None:
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
    plt.clim(0, 3e12)
    plt.show()
    print(f'Saving {plot_path}...')
    plt.savefig(plot_path)
    plt.close()


def grid(
        sim_name: str,
        xg_compare: dict = {},
        coord_type: T.Literal['geographic', 'ecef'] = 'geographic',
        trajectory: np.ndarray = np.empty((3, 0)),
        decimate: int = 10,
        zoom: bool = False
        ) -> None:
    
    sffx = ''
    do_plot_earth = True
    if zoom:
        sffx = '_zoom'
        decimate = 1
        do_plot_earth = False

    sim_direc = Path(GEMINI_SIM_ROOT, sim_name)
    plot_direc = Path(sim_direc, 'plots')
    plot_direc.mkdir(parents=True, exist_ok=True)
    plot_path = Path(plot_direc, f'grid_plot_{coord_type}{sffx}.png')
    views = [[0, 0], [-90, 0], [0, 90], [-45, 45]]

    xg_in = read.grid(sim_direc)
    if trajectory.shape[0] != 3:
        raise IndexError(f'trajectory shape should be (3, :)')

    _, axs = plt.subplots(2, 2, figsize=(16, 16), subplot_kw={'projection':'3d', 'proj_type': 'ortho'})

    i = 0
    for xg in [xg_in, xg_compare]:
        if xg == {}:
            continue
        
        if coord_type == 'geographic':
            coord1 = xg['glon']
            coord2 = xg['glat']
            coord3 = xg['alt'] / 1e3
            labels = ["Geographic longitude (°)", "Geographic latitude (°)", "Geographic altitude (km)"]

            trajectory[2, :] /= 1e3

        elif coord_type == 'ecef':
            coord1 = xg['glon']
            coord2 = xg['glat']
            coord3 = xg['alt']
            coord1, coord2, coord3 = su.geog_to_ecef(coord1, coord2, coord3, units='km')
            labels = ["ECEF X (km)", "ECEF Y (km)", "ECEF Z (km)"]

            trajectory_X, trajectory_Y, trajectory_Z = su.geog_to_ecef(*trajectory, units='km')
            trajectory = np.vstack([trajectory_X, trajectory_Y, trajectory_Z])
        
        if i==0:
            cl = 'k'
        else:
            cl = 'r'
        lw0 = 0.1
        lx = xg['lx']
        for vid in range(4):
            ax = axs[vid // 2, vid % 2]

            if coord_type == 'ecef' and do_plot_earth:
                ax.plot_surface(*earth(), alpha=0.5)

            for ix1 in list(range(0, lx[0], decimate)) + [lx[0]-1]:
                lw = lw0
                if ix1 in [0, lx[0]-1]:
                    lw = 10 * lw0
                ax.plot(coord1[ix1,  :,  0], coord2[ix1,  :,  0], coord3[ix1,  :,  0], color=cl, linewidth=lw)
                ax.plot(coord1[ix1,  :, -1], coord2[ix1,  :, -1], coord3[ix1,  :, -1], color=cl, linewidth=lw)
                ax.plot(coord1[ix1,  0,  :], coord2[ix1,  0,  :], coord3[ix1,  0,  :], color=cl, linewidth=lw)
                ax.plot(coord1[ix1, -1,  :], coord2[ix1, -1,  :], coord3[ix1, -1,  :], color=cl, linewidth=lw)
            for ix2 in list(range(0, lx[1], decimate)) + [lx[1]-1]:
                lw = lw0
                if ix2 in [0, lx[1]-1]:
                    lw = 10 * lw0
                ax.plot(coord1[ :, ix2,  0], coord2[ :, ix2,  0], coord3[ :, ix2,  0], color=cl, linewidth=lw)
                ax.plot(coord1[ :, ix2, -1], coord2[ :, ix2, -1], coord3[ :, ix2, -1], color=cl, linewidth=lw)
                ax.plot(coord1[ 0, ix2,  :], coord2[ 0, ix2,  :], coord3[ 0, ix2,  :], color=cl, linewidth=lw)
                ax.plot(coord1[-1, ix2,  :], coord2[-1, ix2,  :], coord3[-1, ix2,  :], color=cl, linewidth=lw)
            for ix3 in list(range(0, lx[2], decimate)) + [lx[2]-1]:
                lw = lw0
                if ix3 in [0, lx[2]-1]:
                    lw = 10 * lw0
                ax.plot(coord1[ 0,  :, ix3], coord2[ 0,  :, ix3], coord3[ 0,  :, ix3], color=cl, linewidth=lw)
                ax.plot(coord1[-1,  :, ix3], coord2[-1,  :, ix3], coord3[-1,  :, ix3], color=cl, linewidth=lw)
                ax.plot(coord1[ :,  0, ix3], coord2[ :,  0, ix3], coord3[ :,  0, ix3], color=cl, linewidth=lw)
                ax.plot(coord1[ :, -1, ix3], coord2[ :, -1, ix3], coord3[ :, -1, ix3], color=cl, linewidth=lw)
            
            ax.plot(coord1[:, 0, 0], coord2[:, 0, 0], coord3[:, 0, 0], color='r', linewidth=lw0 * 10, label='x1')
            ax.plot(coord1[0, :, 0], coord2[0, :, 0], coord3[0, :, 0], color='g', linewidth=lw0 * 10, label='x2')
            ax.plot(coord1[0, 0, :], coord2[0, 0, :], coord3[0, 0, :], color='b', linewidth=lw0 * 10, label='x3')

            i += 1

            if trajectory.shape[1] > 0:
                ax.plot(trajectory[0, :], trajectory[1, :], trajectory[2, :], 'm', label='trajectory')

            if vid == 0:
                ax.set_xticklabels([])
                ax.set_ylabel(labels[1])
                ax.set_zlabel(labels[2])
            elif vid == 1:
                ax.set_yticklabels([])
                ax.set_xlabel(labels[0])
                ax.set_zlabel(labels[2])
            elif vid == 2:
                ax.set_zticklabels([])
                ax.set_xlabel(labels[0])
                ax.set_ylabel(labels[1])
            elif vid == 3:
                ax.set_xlabel(labels[0])
                ax.set_ylabel(labels[1])
                ax.set_zlabel(labels[2])
            
            ax.azim = views[vid][0]
            ax.elev = views[vid][1]
            
            if zoom:
                # ax.set_xlim([253, 254])
                # ax.set_ylim([32, 34])
                # ax.set_zlim([80, 400])
                dlim = 400
                ax.set_xlim([np.min(trajectory[0, :]) - dlim, np.max(trajectory[0, :]) + dlim])
                ax.set_ylim([np.min(trajectory[1, :]) - dlim, np.max(trajectory[1, :]) + dlim])
                ax.set_zlim([np.min(trajectory[2, :]) - dlim, np.max(trajectory[2, :]) + dlim])

            if coord_type == 'ecef':
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                zlim = ax.get_zlim()
                ax.set_box_aspect([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]])

    plt.suptitle(f'{sim_name} grid ({coord_type}, dec={decimate})')
    plt.legend()
    plt.subplots_adjust(left=0, right=0.95, top=1, bottom=0.05, wspace=0, hspace=0)
    plt.show()
    print(f'Saving {plot_path}...')
    plt.savefig(plot_path)
    plt.close()


def earth(
        units: str = 'km'
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Return ECEF X, Y, and Z surface coordinates to be used in ax.plot_surface()
    
    :param units: Provide meters, "m", or kilometers, "km"
    :type units: str
    :return: ECEF X, Y, and Z
    :rtype: tuple[ndarray[_AnyShape, dtype[Any]], ndarray[_AnyShape, dtype[Any]], ndarray[_AnyShape, dtype[Any]]]
    '''

    if units == 'km':
        scl = 1e3
    elif units == 'm':
        scl = 1e0
    else:
        raise ValueError('Units should be "m" or "km"')

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = _RE * np.outer(np.cos(u), np.sin(v)) / scl
    y = _RE * np.outer(np.sin(u), np.sin(v)) / scl
    z = _RE * np.outer(np.ones_like(u), np.cos(v)) / scl

    return x, y, z


if __name__ == "__main__":
    import sys
    variable(sys.argv[1], sys.argv[2])