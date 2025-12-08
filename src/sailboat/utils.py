from sailboat import RE, GEMINI_SIM_ROOT
from gemini3d import read, utils
from datetime import datetime
import requests
import numpy as np
from os import path, makedirs, listdir
import imageio.v3 as iio


def cut_order(data: np.ndarray) -> tuple[np.ndarray, int]:
    '''
    Divide data by maximal order of magnitude.
    Returns new data of max order unity and the calculated order.
    '''
    order = np.floor(np.log10(np.max(np.abs(data))))
    if ~np.isfinite(order):
        order = 0
    return data / (10 ** order), int(order)


def dipole_to_geomag(q, p, phi):
    # q, p, theta > 0
    theta = np.arcsec((q * p**2)**(1/3))
    rho = (p * q**2)**(-1/3)
    r = RE * rho

    alt = r - RE
    mlon = np.rad2deg(phi)
    mlat = 90 - np.rad2deg(theta)

    return alt, mlon, mlat


def get_activity(date: datetime, f107a_range: int = 81) -> dict:
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


def md(direc: str) -> str:
    '''
    Make directory shortcut
    '''
    makedirs(direc, exist_ok=True)
    return direc


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


def make_gif(plot_direc: str, suffix: str = '.png'):
    '''
    Generate gif out of all image files in plot directory with given suffix.
    Frame sequency is that of the sorted list of filenames.
    '''
    # collect filenames and sort them
    image_filenames = [f for f in listdir(plot_direc) if f.endswith(suffix)]
    image_filenames.sort()

    # generate frames from filenames
    frames = [iio.imread(path.join(plot_direc, img)) for img in image_filenames]

    # save gif using first filename stem
    gif_path = path.join(plot_direc, image_filenames[0][:-4] + '.gif')
    print(f'Saving {gif_path}...')
    iio.imwrite(gif_path, frames, duration=0.1, loop=0)


def si_units(variable: str, order) -> str:
    prefix = {-9: 'n',
              -6: 'µ',
              -3: 'm',
              -2: 'c',
              -1: 'd',
              0: '',
              3: 'k',
              6: 'M',
              9: 'G'}
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


def check_activity(cfg):
    times = cfg['time']
    time = times[len(times) // 2]
    activity = get_activity(time)
    all_match = True
    for v in ['f107a', 'f107', 'Ap']:
        if activity[v] != cfg[v]:
            print(f'Value of f107a in config.nml is {cfg[v]}, but is {activity[v]} for simulation halftime.')
            all_match = False
    if all_match:
        print('Activity levels in config.nml match www-app3.gfz-potsdam.de values.')


def internet_access():
    try:
        requests.get('https://www.google.com')
    except:
        return False
    return True


def simulation_finished_setup(sim_direc):
    if not path.isfile(path.join(sim_direc, 'config.nml')):
        return False
    cfg = read.config(sim_direc)
    initial_conditions_path = path.join(sim_direc, cfg['indat_file'])
    return path.isfile(initial_conditions_path)


def simulation_finished(sim_direc):
    if not path.isfile(path.join(sim_direc, 'config.nml')):
        return False
    if not simulation_finished_setup(sim_direc):
        return False
    cfg = read.config(sim_direc)
    final_output_filename = utils.datetime2stem(cfg['time'][-1]) + '.h5'
    return path.isfile(path.join(sim_direc, final_output_filename))


# def change_sim_name(old_sim_name: str,
#                     new_sim_name: str,
#                     new_eq_direc: str = ''):
    
#     sim_direc = path.join(GEMINI_SIM_ROOT, old_sim_name)
#     old_eq_direc = str(read.config(sim_direc)['eq_dir'])
#     if not new_eq_direc:
#         new_eq_direc = old_eq_direc.replace(old_sim_name, new_sim_name)
#     print(old_eq_direc)
#     print(new_eq_direc)



