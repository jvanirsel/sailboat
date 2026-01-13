from sailboat import _WGS84_A, _WGS84_E2
from gemini3d import read, utils
from datetime import datetime
import requests
import numpy as np
import imageio.v3 as iio
import xarray as xr
from pathlib import Path
import typing as T
from os import listdir


def cut_order(
        data: np.ndarray | xr.DataArray
        ) -> tuple[np.ndarray | xr.DataArray, int]:
    
    '''
    Divide data by maximal order of magnitude.
    Returns new data of max order unity and the calculated order.
    '''

    if isinstance(data, np.ndarray):
        all_zero = np.count_nonzero(data) == 0
    elif isinstance(data, xr.DataArray):
        all_zero = (data.values == 0).all()
    if all_zero:
        order = 0
    else:
        order = np.floor(np.log10(np.max(np.abs(data))))
    return data / (10 ** order), int(order)


def get_activity(
        date: datetime,
        f107a_range: int = 81
        ) -> dict:
    
    '''
    Calculate the solar activity levels from gfz-potsdam.de.
    Returns dictionary containing:
    - f107: the F10.7 local noon-time observed solar radio flux 
    - f107p: the F10.7 value of the previous day
    - f107a: the n-day averaged F10.7 value set by f107a_range
    - Ap: the daily equivalent planetary amplitude
    '''

    num_header_lines = 40
    delta_days = (f107a_range - 1) // 2
    id0 = (date - datetime(1932,1,1)).days + num_header_lines - delta_days

    url_path = 'https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt'
    response = requests.get(url_path)
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
    f107a = round(np.mean(f107s), 1)

    return {'f107': f107, 'f107p': f107p, 'f107a': f107a, 'Ap': Ap}


def dipole_alt_slice(
        xg: dict,
        data: np.ndarray,
        alt_ref: float,
        alt_res: float = 20e3
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    '''
    Linearly interpolate data on a tilted dipole grid to a reference altitude.
    Returns lx1 x lx3 data array along with interpolated geographic longitude and latitude arrays.
    '''

    lx = xg['lx']
    glon = xg['glon']
    glat = xg['glat']
    alt = xg['alt']
    lx1 = lx[0]
    lx2 = lx[1]
    lx3 = lx[2]

    # gather nearest neighbour indeces for x2
    x2ids =  np.argmin(np.abs(alt - alt_ref), axis=1)

    data_out = np.full((lx1, lx3), np.nan)
    glon_out = np.full(data_out.shape, np.nan)
    glat_out = np.full(data_out.shape, np.nan)
    for x1id in range(lx1):
        for x3id in range(lx3):

            # collect x2 previous, central, and next indeces
            x2id = x2ids[x1id, x3id]
            x2idp = np.max((x2id-1, 0))
            x2idn = np.min((x2id+1, lx2-1))

            # collect previous and next altitude
            altp = alt[x1id, x2idp, x3id]
            altn = alt[x1id, x2idn, x3id]

            # ignore values where altitude at central x2 index is out of range
            dalt = np.abs(alt[x1id, x2id, x3id] - alt_ref)
            if dalt < alt_res:

                # collect previous and next data, and linearly interpolate
                datap = data[x1id, x2idp, x3id]
                datan = data[x1id, x2idn, x3id]
                data_out[x1id, x3id] = datap + (datan - datap) * (alt_ref - altp) / (altn - altp)

            # collect previous and next geographic longitudes and latitudes, and linearly interpolate
            glonp = glon[x1id, x2idp, x3id]
            glonn = glon[x1id, x2idn, x3id]
            glatp = glat[x1id, x2idp, x3id]
            glatn = glat[x1id, x2idn, x3id]
            glon_out[x1id, x3id] = glonp + (glonn - glonp) * (alt_ref - altp) / (altn - altp)
            glat_out[x1id, x3id] = glatp + (glatn - glatp) * (alt_ref - altp) / (altn - altp)
    
    return data_out, glon_out, glat_out


def dipole_glon_slice(
        xg: dict,
        data: np.ndarray,
        glon_ref: float,
        glon_res: float = 1
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    '''
    Linearly interpolate data on a tilted dipole grid to a reference geographic longitude.
    Returns lx1 x lx2 data array along with interpolated geographic latitude and altitude arrays.
    '''

    lx = xg['lx']
    glon = xg['glon']
    glat = xg['glat']
    alt = xg['alt']
    lx1 = lx[0]
    lx2 = lx[1]
    lx3 = lx[2]

    # gather nearest neighbour indeces for x3
    x3ids =  np.argmin(np.abs(glon - glon_ref), axis=2)

    data_out = np.full((lx1, lx2), np.nan)
    glat_out = np.full(data_out.shape, np.nan)
    alt_out = np.full(data_out.shape, np.nan)
    for x1id in range(lx1):
        for x2id in range(lx2):

            # collect x3 previous, central, and next indeces
            x3id = x3ids[x1id, x2id]
            x3idp = np.max((x3id-1, 0))
            x3idn = np.min((x3id+1, lx3-1))

            # collect previous and next longitude
            glonp = glon[x1id, x2id, x3idp]
            glonn = glon[x1id, x2id, x3idn]

            # ignore values where longitude at central x3 index is out of range
            dglon = np.abs(glon[x1id, x2id, x3id] - glon_ref)
            if dglon < glon_res:

                # collect previous and next data, and linearly interpolate
                datap = data[x1id, x2id, x3idp]
                datan = data[x1id, x2id, x3idn]
                data_out[x1id, x2id] = datap + (datan - datap) * (glon_ref - glonp) / (glonn - glonp)

            # collect previous and next geographic latitudes and altitudes, and linearly interpolate
            glatp = glat[x1id, x2id, x3idp]
            glatn = glat[x1id, x2id, x3idn]
            altp = alt[x1id, x2id, x3idp]
            altn = alt[x1id, x2id, x3idn]
            glat_out[x1id, x2id] = glatp + (glatn - glatp) * (glon_ref - glonp) / (glonn - glonp)
            alt_out[x1id, x2id] = altp + (altn - altp) * (glon_ref - glonp) / (glonn - glonp)
    
    return data_out, glat_out, alt_out


def make_gif(
        plot_direc: Path,
        suffix: str = '.png',
        prefix: str = '',
        filename: str = ''
        ) -> None:
    
    '''
    Generate gif out of all image files in plot directory with given suffix.
    Frame sequency is that of the sorted list of filenames.
    '''

    # collect filenames and sort them
    image_filenames = [f for f in listdir(plot_direc) if f.endswith(suffix) and f.startswith(prefix)]
    image_filenames.sort()

    # generate frames from filenames
    frames = [iio.imread(Path(plot_direc, img)) for img in image_filenames]

    # save gif using first filename stem
    if not filename:
        filename = image_filenames[0][:-4] + '.gif'
    else:
        filename = filename.split('.')[0] + '.gif'
    
    gif_path = Path(plot_direc, filename)
    print(f'Saving {gif_path}...')
    iio.imwrite(gif_path, frames, duration=0.1, loop=0)


def si_units(
        variable: str,
        order: int
        ) -> str:
    
    prefix = {
        -9: 'n',
        -6: 'µ',
        -3: 'm',
        -2: 'c',
        -1: 'd',
        0: '',
        3: 'k',
        6: 'M',
        9: 'G',
        }
    
    if variable[0] == 'n':
        base_units = 'm-3'
        if order == -8:
            order = -2
        elif order != 0:
            raise ValueError('Order should not be 0 for density variables')
    elif variable[0] == 'v':
        base_units = 'm s-1'
    elif variable[0] == 'T':
        base_units = 'K'
    elif variable[0] == 'J':
        base_units = 'A m-2'
    elif 'Phi' in variable:
        base_units = 'V'
    return prefix[order] + base_units


def check_activity(
        cfg: dict
        ) -> None:
    
    times = cfg['time']
    time = times[len(times) // 2]
    activity = get_activity(time)
    all_match = True
    for v in ['f107a', 'f107', 'Ap']:
        if activity[v] != cfg[v]:
            print(f'Value of f107a in config.nml is {cfg[v]}, but is {activity[v]} for simulation halftime.')
            all_match = False
    if all_match:
        print('  Activity levels in config.nml match www-app3.gfz-potsdam.de values.')


def internet_access() -> None:

    try:
        requests.get('https://www.google.com')
    except:
        return False
    return True


def simulation_finished_setup(
        sim_direc: Path
        ) -> None:
    
    if not Path(sim_direc, 'config.nml').is_file():
        return False
    cfg = read.config(sim_direc)
    return Path(sim_direc, cfg['indat_file']).is_file()


def simulation_finished(
        sim_direc: Path
        ) -> None:
    
    if not Path(sim_direc, 'config.nml').is_file():
        return False
    if not simulation_finished_setup(sim_direc):
        return False
    cfg = read.config(sim_direc)
    final_output_filename = utils.datetime2stem(cfg['time'][-1]) + '.h5'
    return Path(sim_direc, final_output_filename).is_file()


def geog_to_ecef(
        glon: np.ndarray,
        glat: np.ndarray,
        galt: np.ndarray,
        is_geodetic: bool = True,
        in_degrees: bool = True,
        units: T.Literal['m', 'km'] = 'm',
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    glon, glat, galt = np.broadcast_arrays(glon, glat, galt)
    out_shape = glon.shape
    
    glon = glon.ravel()
    glat = glat.ravel()
    galt = galt.ravel()

    if in_degrees:
        glon = np.deg2rad(glon)
        glat = np.deg2rad(glat)

    sin_lon = np.sin(glon)
    sin_lat = np.sin(glat)
    cos_lon = np.cos(glon)
    cos_lat = np.cos(glat)

    scl = 1.0 + int(units == 'km') * 999.0

    a = _WGS84_A / scl
    galt /= scl

    if is_geodetic:
        e2 = _WGS84_E2
        N = a / np.sqrt(1.0 - e2 * sin_lat**2)

        ecef_X = (N + galt) * cos_lat * cos_lon
        ecef_Y = (N + galt) * cos_lat * sin_lon
        ecef_Z = (N * (1.0 - e2) + galt) * sin_lat
    else:
        ecef_X = a * cos_lat * cos_lon
        ecef_Y = a * cos_lat * sin_lon
        ecef_Z = a * sin_lat

    ecef_X = ecef_X.reshape(out_shape)
    ecef_Y = ecef_Y.reshape(out_shape)
    ecef_Z = ecef_Z.reshape(out_shape)

    return ecef_X, ecef_Y, ecef_Z


