import matplotlib.pyplot as plt
from datetime import datetime
import requests
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from gemini3d.grid.convert import Re
from os import path, makedirs, listdir
import imageio.v3 as iio

def plot_all(cfg, dat, slice_coord: int):
    time =  dat['time'].values.astype('datetime64[ms]').astype(datetime)
    time_str = time.strftime('%Y%m%d_%H%M%S')
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
            data, order = cut_order(data)
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
                data_slice = var_data.isel({f'x{slice_coord + 1}': var_data.shape[slice_coord] // 2})
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
            data_slice = var_data.isel({f'x{slice_coord + 1}': var_data.shape[slice_coord] // 2})
            plot_slice(ax, data_slice, var_name)
            i += 1

    plt.tight_layout()
    plot_direc = md(path.join(cfg['nml'].parent, 'plots'))
    save_path =  path.join(plot_direc, f'all_vars_{time_str}_x{slice_coord + 1}_slice.png')
    print(f'Saving {save_path}')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def cut_order(data):
    order = np.floor(np.log10(np.max(np.abs(data))))
    if ~np.isfinite(order):
        order = 0
    return data / (10 ** order), int(order)


def dipole_to_geomag(q, p, phi) -> tuple:
    # q, p, theta > 0
    theta = np.arcsec((q * p**2)**(1/3))
    rho = (p * q**2)**(-1/3)
    r = Re * rho

    alt = r - Re
    mlon = np.rad2deg(phi)
    mlat = 90 - np.rad2deg(theta)

    return alt, mlon, mlat


def get_activity(date: datetime, f107a_range = 81):
    num_header_lines = 40
    delta_days = (f107a_range - 1) // 2
    id0 = (date - datetime(1932,1,1)).days + num_header_lines - delta_days

    url = 'https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt'
    response = requests.get(url)
    # lines = strsplit(webread(url),'\n');
    if response.status_code == 200:
        lines = response.text.split('\n')
    else:
        print(response.reason)
        raise RuntimeError(f'Status = {response.status_code}, {response.reason}')


    f107s = np.empty(f107a_range)
    for id in range(id0, id0 + f107a_range):
        data = np.array(lines[id].split())
        f107s[id - id0] = float(data[25]) # F10.7obs
        if id == id0 + delta_days:
            Ap = float(data[23])

    f107 = f107s[delta_days]
    f107p = f107s[delta_days - 1]
    f107a = np.mean(f107s)

    return {'f107': f107, 'f107p': f107p, 'f107a': f107a, 'Ap': Ap}


def plot_grid(cfg, xg, coord_type='geographic'):
    direc = md(path.join(cfg['nml'].parent, 'plots'))
    filename = path.join(direc, f'grid_plot_{coord_type}.png')

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    if coord_type == 'geographic':
        coord1 = xg['glon']
        coord2 = xg['glat']
        coord3 = xg['alt'] / 1e3
        labels = ["Longitude (°)", "Latitude (°)", "Altitude (km)"]
    elif coord_type == 'ecef':
        coord1 = xg['x'] / 1e3
        coord2 = xg['y'] / 1e3
        coord3 = xg['z'] / 1e3
        labels = ["X (km)", "Y (km)", "Z (km)"]
    else:
        raise ValueError(f'Unknown coord_type {coord_type}')
    ax.plot(coord1[ :,  0,  0], coord2[ :,  0,  0], coord3[ :,  0,  0], color='k', label='x1 / q')
    ax.plot(coord1[ :,  0, -1], coord2[ :,  0, -1], coord3[ :,  0, -1], color='k')
    ax.plot(coord1[ :, -1,  0], coord2[ :, -1,  0], coord3[ :, -1,  0], color='k')
    ax.plot(coord1[ :, -1, -1], coord2[ :, -1, -1], coord3[ :, -1, -1], color='k')
    ax.plot(coord1[ 0 , :,  0], coord2[ 0,  :,  0], coord3[ 0,  :,  0], color='b', label='x2 / p')
    ax.plot(coord1[-1,  :,  0], coord2[-1,  :,  0], coord3[-1,  :,  0], color='b')
    ax.plot(coord1[ 0 , :, -1], coord2[ 0,  :, -1], coord3[ 0,  :, -1], color='b')
    ax.plot(coord1[-1,  :, -1], coord2[-1,  :, -1], coord3[-1,  :, -1], color='b')
    ax.plot(coord1[ 0,  0,  :], coord2[ 0,  0,  :], coord3[ 0,  0,  :], color='r', label='x3 / phi')
    ax.plot(coord1[-1,  0,  :], coord2[-1,  0,  :], coord3[-1,  0,  :], color='r')
    ax.plot(coord1[ 0, -1,  :], coord2[ 0, -1,  :], coord3[ 0, -1,  :], color='r')
    ax.plot(coord1[-1, -1,  :], coord2[-1, -1,  :], coord3[-1, -1,  :], color='r')
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    plt.legend()
    plt.show()
    print(filename)
    plt.savefig(filename)
    plt.close()


def md(direc) -> path:
    makedirs(direc, exist_ok=True)
    return direc


def get_dipole_slice_ids(xg: dict, dat: xr.DataArray, alt_ref: float) -> xr.DataArray:
    lx = xg['lx']
    alt = xg['alt']
    glat = xg['glat']
    glon = xg['glon']
    x2ids =  np.argmin(np.abs(alt - alt_ref), axis=1)

    dat_tmp = np.empty((lx[0], lx[2]))
    glat_tmp = np.empty(dat_tmp.shape)
    glon_tmp = np.empty(dat_tmp.shape)
    for x1id in range(lx[0]):
        for x3id in range(lx[2]):
            x2id = x2ids[x1id, x3id]
            dat_tmp[x1id, x3id] = dat['ne'][x1id, x2id, x3id]
            glat_tmp[x1id, x3id] = glat[x1id, x2id, x3id]
            glon_tmp[x1id, x3id] = glon[x1id, x2id, x3id]
    
    dat_out = xr.DataArray(
        dat_tmp,
        dims = ("glat", "glon"),
        coords = {
            "GLAT": (("glat", "glon"), glat_tmp),
            "GLON": (("glat", "glon"), glon_tmp)
            },
        name = "test"
    )
    return dat_out


def make_gif(plot_direc):
    pngs = [f for f in listdir(plot_direc) if f.endswith('.png')]
    frames = [iio.imread(path.join(plot_direc, p)) for p in pngs]
    iio.imwrite(path.join(plot_direc, pngs[0][:-4] + '.gif'), frames, duration=0.1)