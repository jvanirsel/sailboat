import matplotlib.pyplot as plt
from datetime import datetime
import requests
import numpy as np
import xarray as xr
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
    plot_direc = md(path.join(cfg['nml'].parent, 'plots', f'all_vars_x{slice_coord + 1}_slice'))
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


def plot_grid(
        cfg: dict,
        xg_in: dict,
        xg_compare: dict = {},
        coord_type: str ='geographic',
        trajectory: np.ndarray = np.empty((3, 0))
        ):
    direc = md(path.join(cfg['nml'].parent, 'plots'))
    filename = path.join(direc, f'grid_plot_{coord_type}.png')
    views = [[0, 0], [-90, 0], [0, 90], [-45, 45]]

    if trajectory.shape[0] != 3:
        raise ValueError(f'trajectory shape should be (3, :)')

    # ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')
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
            # ax.set_zlim([0, 500])
    plt.legend()
    # plt.tight_layout()
    plt.subplots_adjust(left=0, right=0.95, top=1, bottom=0.05, wspace=0, hspace=0)
    plt.show()
    print(f'Saving {filename}...')
    plt.savefig(filename)
    plt.close()


def md(direc) -> path:
    makedirs(direc, exist_ok=True)
    return direc


def dipole_alt_slice(
        xg: dict,
        dat: np.ndarray,
        alt_ref: float,
        alt_res: float = 20e3
        ) -> np.ndarray:
    lx = xg['lx']
    glon = xg['glon']
    glat = xg['glat']
    alt = xg['alt']

    lx1 = lx[0]
    lx2 = lx[1]
    lx3 = lx[2]
    x2ids =  np.argmin(np.abs(alt - alt_ref), axis=1)

    dat_out = np.full((lx[0], lx[2]), np.nan)
    glon_out = np.full(dat_out.shape, np.nan)
    glat_out = np.full(dat_out.shape, np.nan)
    for x1id in range(lx1):
        for x3id in range(lx3):
            x2id = x2ids[x1id, x3id]
            x2idp = np.max((x2id-1, 0))
            x2idn = np.min((x2id+1, lx2-1))

            altp = alt[x1id, x2idp, x3id]
            altn = alt[x1id, x2idn, x3id]

            dalt = np.abs(alt[x1id, x2id, x3id] - alt_ref)
            if dalt < alt_res:
                datp = dat[x1id, x2idp, x3id]
                datn = dat[x1id, x2idn, x3id]
                dat_out[x1id, x3id] = datp + (datn - datp) * (alt_ref - altp) / (altn - altp)

            glonp = glon[x1id, x2idp, x3id]
            glonn = glon[x1id, x2idn, x3id]
            glatp = glat[x1id, x2idp, x3id]
            glatn = glat[x1id, x2idn, x3id]

            glon_out[x1id, x3id] = glonp + (glonn - glonp) * (alt_ref - altp) / (altn - altp)
            glat_out[x1id, x3id] = glatp + (glatn - glatp) * (alt_ref - altp) / (altn - altp)
    
    return dat_out, glon_out, glat_out


def dipole_glon_slice(
        xg: dict,
        dat: np.ndarray,
        glon_ref: float,
        glon_res: float = 1
        ) -> np.ndarray:
    lx = xg['lx']
    glon = xg['glon']
    glat = xg['glat']
    alt = xg['alt']

    lx1 = lx[0]
    lx2 = lx[1]
    lx3 = lx[2]
    x3ids =  np.argmin(np.abs(glon - glon_ref), axis=2)

    dat_out = np.full((lx[0], lx[1]), np.nan)
    glat_out = np.full(dat_out.shape, np.nan)
    alt_out = np.full(dat_out.shape, np.nan)
    for x1id in range(lx1):
        for x2id in range(lx2):
            x3id = x3ids[x1id, x2id]
            x3idp = np.max((x3id-1, 0))
            x3idn = np.min((x3id+1, lx3-1))

            glonp = glon[x1id, x2id, x3idp]
            glonn = glon[x1id, x2id, x3idn]

            dglon = np.abs(glon[x1id, x2id, x3id] - glon_ref)
            if dglon < glon_res:
                datp = dat[x1id, x2id, x3idp]
                datn = dat[x1id, x2id, x3idn]
                # linear interpolation of non-linear angle = bad. hoping thetap ~= thetan and rp ~= rn
                dat_out[x1id, x2id] = datp + (datn - datp) * (glon_ref - glonp) / (glonn - glonp)

            glatp = glat[x1id, x2id, x3idp]
            glatn = glat[x1id, x2id, x3idn]
            altp = alt[x1id, x2id, x3idp]
            altn = alt[x1id, x2id, x3idn]

            glat_out[x1id, x2id] = glatp + (glatn - glatp) * (glon_ref - glonp) / (glonn - glonp)
            alt_out[x1id, x2id] = altp + (altn - altp) * (glon_ref - glonp) / (glonn - glonp)
    
    return dat_out, glat_out, alt_out

def make_gif(plot_direc, suffix='.png', buffer=100):
    pngs = [f for f in listdir(plot_direc) if f.endswith(suffix)]
    pngs.sort()
    frames = [iio.imread(path.join(plot_direc, p)) for p in pngs]
    gif_name = path.join(plot_direc, pngs[0][:-4] + '.gif')
    print(f'Saving {gif_name}...')
    iio.imwrite(gif_name, frames,
                duration=0.1,
                loop=0)
    
def si_units(variable: str, order) -> str:
    prefix = {-9: 'n', -6: 'u', -3: 'm', 0: '', 3: 'k', 6: 'M', 9: 'G'}
    if variable[0] == 'n':
        base_units = 'm-3'
    elif variable[0] == 'v':
        base_units = 'm s-1'
    elif variable[0] == 'T':
        base_units = 'K'
    elif variable[0] == 'J':
        base_units = 'A m-2'
    elif 'Phi' in variable:
        base_units = 'V'
    return prefix[order] + base_units