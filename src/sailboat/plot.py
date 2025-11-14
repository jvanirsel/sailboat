from os import path
from gemini3d import read, utils as gu
from sailboat import utils as su, a_bit_of_light_reading, GEMINI_SIM_ROOT
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def quick_summary(cfg: dict,
                  dat: xr.DataArray,
                  time,
                  slice_coord: int
                  ):
    
    time_str = gu.datetime2stem(time).replace(' ', '0')
    flag = cfg.get('flagoutput')
    grid_flag = cfg.get('gridflag')

    if grid_flag != 1:
        dat.coords['x1'] = dat.coords['x1'] / 1e3
        dat.coords['x2'] = dat.coords['x2'] / 1e3
        dat.coords['x3'] = dat.coords['x3'] / 1e3

    match flag:
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
            raise ValueError('Unknown output flag.')

    def plot_slice(ax, data, var_name: str, s: int = None):
        if var_name[0] == 'n':
            data = np.log10(data)
            cb_label = f'log10 {var_name}'
        else:
            data, order = su.cut_order(data)
            cb_label = f'{var_name} / 1e{order}'
        im = data.plot.imshow(ax=ax, cmap='viridis', add_colorbar=True)
        cb = im.colorbar
        cb.set_label(cb_label)
        cb.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        if s:
            ax.set_title(f'{var_name} (s={s})')
        else:
            ax.set_title(var_name)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 3*num_rows))
    i = 0
    for v in range(len(dat.data_vars)):
        var_name = list(dat.data_vars)[v]
        if var_name in ['ns', 'vs1', 'Ts']:
            for s in range(6):
                ax = axs[i // num_cols, i % num_cols]
                var_data = dat[var_name].sel(species=s)
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
            var_data = dat[var_name]
            data_slice = var_data.isel({f'x{slice_coord}': var_data.shape[slice_coord - 1] // 2})
            plot_slice(ax, data_slice, var_name)
            i += 1

    plt.tight_layout()
    plot_direc = su.md(path.join(cfg['nml'].parent, 'plots', 'summary', f'all_vars_x{slice_coord}_slice'))
    save_path =  path.join(plot_direc, f'all_x{slice_coord}_slice_{time_str}.png')
    print(f'Saving {save_path}')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def variable(
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
    sim_path = path.join(GEMINI_SIM_ROOT, sim_name)
    plot_direc = su.md(path.join(sim_path, 'plots', variable))

    cfg = read.config(sim_path)
    xg = a_bit_of_light_reading.grid(sim_path)
    times = cfg['time']

    scl = 10**dat_order
    units = su.si_units(variable, dat_order)

    if not glon_ref:
        glon_ref = cfg['glon']
    if not mlon_ref:
        mlon_ref = xg['x3'][2 + xg['lx'][2] // 2] * 180 / np.pi

    for time in times:
        time_str = gu.datetime2stem(time).replace(' ', '0')
        dat = read.frame(sim_path, time, var=variable)[variable] / scl

        plot_name_alt = f'{variable}_alt={alt_ref / 1e3:.0f}km_{time_str}.png'
        plot_path = path.join(plot_direc, plot_name_alt)
        title = f'{time_str} (alt = {alt_ref / 1e3:.0f}+/-{alt_res / 1e3:.0f} km)'
        dat_label = f'{variable} ({units})'
        # plot_alt_slice(xg, dat, alt_ref, alt_res, plot_path, title, dat_label)

        plot_name_glon = f'{variable}_glon={glon_ref:.0f}deg_{time_str}.png'
        plot_path = path.join(plot_direc, plot_name_glon)
        title = f'{time_str} (glon = {glon_ref:.0f}+/-{glon_res:.0f}°)'
        dat_label = f'{variable} ({units})'
        # plot_glon_slice(xg, dat, glon_ref, glon_res, alt_ref, plot_path, title, dat_label)

        plot_name_mlon = f'{variable}_mlon={mlon_ref:.0f}deg_{time_str}.png'
        plot_path = path.join(plot_direc, plot_name_mlon)
        title = f'{time_str} (mlon = {mlon_ref:.0f}°)'
        dat_label = f'{variable} ({units})'
        plot_mlon_slice(xg, dat, mlon_ref, alt_ref, plot_path, title, dat_label)

    if make_gifs:
        su.make_gif(plot_direc, suffix=plot_name_alt[-15:])
        su.make_gif(plot_direc, suffix=plot_name_glon[-15:])
        su.make_gif(plot_direc, suffix=plot_name_mlon[-15:])


def plot_alt_slice(
        xg: dict,
        dat: np.ndarray,
        alt_ref: float,
        alt_res: float,
        plot_path: str,
        title: str,
        dat_label: str
        ):

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
        plot_path: str,
        title: str,
        dat_label: str
        ):

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


def grid(
        sim_name: str,
        xg_compare: dict = {},
        coord_type: str ='geographic',
        trajectory: np.ndarray = np.empty((3, 0))
        ):
    
    sim_path = path.join(GEMINI_SIM_ROOT, sim_name)
    plot_direc = su.md(path.join(sim_path, 'plots'))
    filename = path.join(plot_direc, f'grid_plot_{coord_type}.png')
    views = [[0, 0], [-90, 0], [0, 90], [-45, 45]]

    xg_in = read.grid(sim_path)
    if trajectory.shape[0] != 3:
        raise ValueError(f'trajectory shape should be (3, :)')

    _, axs = plt.subplots(2, 2, figsize=(16, 16), subplot_kw={'projection':'3d'})

    i = 0
    for xg in [xg_in, xg_compare]:
        if xg == {}:
            continue
        
        if coord_type == 'geographic':
            coord1 = xg['glon']
            coord2 = xg['glat']
            coord3 = xg['alt'] / 1e3
            labels = ["Geographic longitude (°)", "Geographic latitude (°)", "Geographic altitude (km)"]
        elif coord_type == 'ecef':
            coord1 = xg['x'] / 1e3
            coord2 = xg['y'] / 1e3
            coord3 = xg['z'] / 1e3
            labels = ["ECEF X (km)", "ECEF Y (km)", "ECEF Z (km)"]
        else:
            raise ValueError(f'Unknown coord_type {coord_type}')
        
        if i==0:
            cl = 'k'
        else:
            cl = 'r'
        lw0 = 0.1
        lx = xg['lx']
        for vid in range(4):
            ax = axs[vid // 2, vid % 2]
            for ix1 in list(range(0, lx[0], 10)) + [lx[0]-1]:
                lw = lw0
                if ix1 in [0, lx[0]-1]:
                    lw = 10 * lw0
                ax.plot(coord1[ix1,  :,  0], coord2[ix1,  :,  0], coord3[ix1,  :,  0], color=cl, linewidth=lw)
                ax.plot(coord1[ix1,  :, -1], coord2[ix1,  :, -1], coord3[ix1,  :, -1], color=cl, linewidth=lw)
                ax.plot(coord1[ix1,  0,  :], coord2[ix1,  0,  :], coord3[ix1,  0,  :], color=cl, linewidth=lw)
                ax.plot(coord1[ix1, -1,  :], coord2[ix1, -1,  :], coord3[ix1, -1,  :], color=cl, linewidth=lw)
            for ix2 in list(range(0, lx[1], 10)) + [lx[1]-1]:
                lw = lw0
                if ix2 in [0, lx[1]-1]:
                    lw = 10 * lw0
                ax.plot(coord1[ :, ix2,  0], coord2[ :, ix2,  0], coord3[ :, ix2,  0], color=cl, linewidth=lw)
                ax.plot(coord1[ :, ix2, -1], coord2[ :, ix2, -1], coord3[ :, ix2, -1], color=cl, linewidth=lw)
                ax.plot(coord1[ 0, ix2,  :], coord2[ 0, ix2,  :], coord3[ 0, ix2,  :], color=cl, linewidth=lw)
                ax.plot(coord1[-1, ix2,  :], coord2[-1, ix2,  :], coord3[-1, ix2,  :], color=cl, linewidth=lw)
            for ix3 in list(range(0, lx[2], 10)) + [lx[2]-1]:
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
                ax.plot(trajectory[0, :], trajectory[1, :], trajectory[2, :] / 1e3, 'm', label='trajectory')

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
    plt.legend()
    plt.subplots_adjust(left=0, right=0.95, top=1, bottom=0.05, wspace=0, hspace=0)
    plt.show()
    print(f'Saving {filename}...')
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    import sys
    variable(sys.argv[1], sys.argv[2])