from sailboat import interpolate
from gemini3d import utils
from os import path, makedirs, listdir
from datetime import datetime, timedelta
from scipy.constants import h, c
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import h5py

def convert_solar_flux(cfg: dict,
                       xg: dict, # required for setup_functions in config.nml
                       solflux_in_extension: str = 'nc4',
                       solflux_in_direc: str = 'data/apep/2023/fism2_masked'
                       ):

    ## read simulation data
    sim_direc = path.dirname(cfg['nml'])
    try:
        sim_solflux_direc = path.join(sim_direc, cfg['solfluxdir'])
        sim_solflux_dt = cfg['dtsolflux']
    except KeyError:
        raise KeyError('solfluxdir and/or dtsolflux not found in config file. \
                       Please adjust gemini3d/config.py to include solflux namelist.')
    if not path.isdir(sim_solflux_direc):
        makedirs(sim_solflux_direc)
    sim_times = []
    t = cfg['time'][0]
    while t <= cfg['time'][-1]:
        sim_times.append(t)
        t += sim_solflux_dt

    ## read input filenames
    solflux_filenames = [f for f in listdir(solflux_in_direc) if f.endswith('.' + solflux_in_extension)]
    if not solflux_filenames:
        raise FileNotFoundError(f'No .nc4 files found in {solflux_in_direc}')
    solflux_times = [datetime.strptime(f[13:28], '%Y%m%d_%H%M%S') for f in solflux_filenames]
    if solflux_times[0] - sim_solflux_dt > sim_times[0] or \
       solflux_times[-1] + sim_solflux_dt < sim_times[-1]:
        raise ValueError('Solar flux data out of simulation time range\n' \
                         f'  Simulation time range: {sim_times[0]} -- {sim_times[-1]}\n' \
                         f'  Solar flux time range: {solflux_times[0]} -- {solflux_times[-1]}')

    ## read solar flux grid and wavelength data
    solflux_data = h5py.File(path.join(solflux_in_direc, solflux_filenames[0]))
    solflux_tgcm_data = solflux_data['sb_tgcm'] # first 22 tgcm spectral bins match gemini bins
    glat = np.array(solflux_data['Lat'][:], dtype=np.float64) # degrees
    glon = np.array(solflux_data['Lon'][:], dtype=np.float64) # degrees
    wvl0 = np.array(solflux_tgcm_data['Start_wavelength'][:22], dtype=np.float64) * 1e-9 # m
    wvl1 = np.array(solflux_tgcm_data['End_wavelength'][:22], dtype=np.float64) * 1e-9 # m
    
    ## calculate average photon energy per spectral bin to convert W/m2 to photons/m2/s
    avg_photon_energy = h * c * np.log(wvl1 / wvl0) / (wvl1 - wvl0) # J (<E> = h c <1 / wvl>)

    ## wrap longtiudes to 0 -- 360 for gemini use
    glon = glon % 360
    glon_ids = np.argsort(glon)
    glon = glon[glon_ids]

    ## loop through simulation times
    for sim_time in sim_times:
        print(f'Processing simulation time {sim_time}...', end='\r')

        ## nearest neighbour time interp
        tid = np.argmin(np.abs([(sim_time - t).total_seconds() for t in solflux_times]))
        solflux_path = path.join(solflux_in_direc, solflux_filenames[tid])
        solflux_data = h5py.File(solflux_path)
        solflux_tgcm_data = solflux_data['sb_tgcm']

        ## read irradiance data
        irr = np.array(solflux_tgcm_data['msk_irradiance_tlsm'][:, :, :22], dtype=np.float64) # W/m2
        irr = irr / avg_photon_energy # photons/m2/s
        irr = irr[:, glon_ids, :].transpose(1,0,2) # llon x llat x 22
        Iinf = np.array(irr.transpose(2, 1, 0)) # preserve irr.shape when read by fortran

        ## save to hdf5
        h5_name = utils.datetime2stem(sim_time)
        f = h5py.File(path.join(sim_solflux_direc, h5_name + '.h5'), 'w')
        h5_ds = f.create_dataset('/Iinf', data=Iinf)
        h5_ds.attrs['units'] = 'photons meters^-2 seconds^-1'
        h5_ds.attrs['description'] = 'masked irradiance in bins defined by solomon & qian (2005)'
        h5_ds.attrs['ref_url'] = 'https://doi.org/10.1029/2005JA011160'
        h5_ds.attrs['wavelength_bins'] = [wvl0, wvl1]
        h5_ds.attrs['wavelength_units'] = 'meters'
        h5_ds.attrs['version'] = '1.1.0'
        h5_ds.attrs['tracked_changes'] = 'nearest neigbour interpolation'
        f.close()

    ## save simulation size and grid information
    f = h5py.File(path.join(sim_solflux_direc, 'simsize.h5'), 'w')
    f.create_dataset('/llat', data=glat.size)
    f.create_dataset('/llon', data=glon.size)
    f.close()

    f = h5py.File(path.join(sim_solflux_direc, 'simgrid.h5'), 'w')
    f.create_dataset('/mlat', data=glat) # glat instead of mlat is ok here
    f.create_dataset('/mlon', data=glon)
    f.close()

    solflux_data.close()
    print('Done converting solar flux data...' + ' ' * 40)


def convert_ephemeris():
    min_alt = 50e3 # m
    start_times = [np.datetime64('2023-10-14T16:00:00'), 
                np.datetime64('2023-10-14T16:35:00'), 
                np.datetime64('2023-10-14T17:10:00')]

    ephemeris_data_h5 = h5py.File(f'data/apep/ephemeris.h5', 'w')
    rocket_ids = [386, 387, 388, 392, 393, 394]

    for rid in rocket_ids:
        if rid < 390:
            ephemeris_in_path = f'data/apep/2023/ephemeris/Apep1_x{rid}.txt'
            data = np.loadtxt(ephemeris_in_path, delimiter='\t', dtype=np.float64)
            time = start_times[rid - 386] + (data[:, 0]*1e6).astype('timedelta64[us]')
            day0 = time[0].astype('datetime64[D]')
            us_of_day = (time - day0).astype(np.int64)
            lat = data[:, 1]
            lon = data[:, 2] % 360
            alt = data[:, 3] * 1e3

        else:
            ephemeris_in_path = f'data/apep/2024/ephemeris/36_{rid}_MainPayload_GPS_Trajectory.dat'
            data = np.loadtxt(ephemeris_in_path, delimiter='\t', skiprows=1, dtype=str)
            day0 = np.datetime64('2024-04-08')
            us_of_day = np.round((data[:, 3].astype(int) * 3600
                         + data[:, 4].astype(int) * 60
                         + data[:, 5].astype(np.float64)) * 1e6).astype(np.int64)
            time = day0 + us_of_day.astype('timedelta64[us]')
            lat = data[:, 6].astype(np.float64)
            lon = data[:, 7].astype(np.float64) % 360
            alt = data[:, 8].astype(np.float64) * 1e3

        print(f'Processing {ephemeris_in_path}...')
        
        ds = ephemeris_data_h5.create_dataset(f'/36.{rid}/raw/time', data=us_of_day, dtype=np.int64)
        ds.attrs['Description'] = f'Microsecond of {day0}'
        ds.attrs['Units'] = 'microseconds'

        ds = ephemeris_data_h5.create_dataset(f'/36.{rid}/raw/latitude', data=lat, dtype=np.float64)
        ds.attrs['Description'] = 'Geodetic latitude'
        ds.attrs['Units'] = 'degrees'

        ds = ephemeris_data_h5.create_dataset(f'/36.{rid}/raw/longitude', data=lon, dtype=np.float64)
        ds.attrs['Description'] = 'Geodetic longitude'
        ds.attrs['Units'] = 'degrees'

        ds = ephemeris_data_h5.create_dataset(f'/36.{rid}/raw/altitude', data=alt, dtype=np.float64)
        ds.attrs['Description'] = 'Geodetic altitude'
        ds.attrs['Units'] = 'meters'

        is_above_min_alt = alt > min_alt
        is_before_max_time = time < day0 + np.timedelta64(1, 'D')
        ids = is_above_min_alt & is_before_max_time

        dt = np.median(np.diff(us_of_day[ids]))
        us_of_day_interp = np.arange(np.min(us_of_day[ids]), np.max(us_of_day[ids]) + dt/2, dt, dtype=np.int64)

        flat = interp1d(us_of_day[ids], lat[ids])
        flon = interp1d(us_of_day[ids], lon[ids])
        falt = interp1d(us_of_day[ids], alt[ids])

        lat_interp = flat(us_of_day_interp)
        lon_interp = flon(us_of_day_interp)
        alt_interp = falt(us_of_day_interp)

        ds = ephemeris_data_h5.create_dataset(f'/36.{rid}/interpolated/time', data=us_of_day_interp, dtype=np.int64)
        ds.attrs['Description'] = f'Interpolated microsecond of {day0}'
        ds.attrs['Units'] = 'microseconds'

        ds = ephemeris_data_h5.create_dataset(f'/36.{rid}/interpolated/latitude', data=lat_interp, dtype=np.float64)
        ds.attrs['Description'] = 'Interpolated geodetic latitude'
        ds.attrs['Units'] = 'degrees'

        ds = ephemeris_data_h5.create_dataset(f'/36.{rid}/interpolated/longitude', data=lon_interp, dtype=np.float64)
        ds.attrs['Description'] = 'Interpolated geodetic longitude'
        ds.attrs['Units'] = 'degrees'

        ds = ephemeris_data_h5.create_dataset(f'/36.{rid}/interpolated/altitude', data=alt_interp, dtype=np.float64)
        ds.attrs['Description'] = 'Interpolated geodetic altitude'
        ds.attrs['Units'] = 'meters'

    ephemeris_data_h5.close()


def convert_slp_data():
    slp_data_h5 = h5py.File('data/apep/2023/slp.h5', 'w')

    for rid in range(386, 389):

        ## get ephemeris data
        time, _, _, alt = get_trajectory(rid, cadence_Hz=5000)

        ## read slp data, downleg only
        slp_in_path = f'data/apep/2023/slp/SLP_{rid}_v3_nosmoothing.txt'
        data = np.loadtxt(slp_in_path, delimiter='\t', dtype=np.float64, skiprows=1)
        slp_time = data[:, 0] * 1e6 # convert to microseconds
        slp_alt = data[:, 1] * 1e3 # convert to meters
        slp_density = data[:, 2]
        slp_electron_temp = data[:, 3]

        ## change time-of-flight to datetime64 
        max_alt_id = np.argmax(alt)
        downleg_alt = alt[max_alt_id:]
        downleg_time = time[max_alt_id:].astype(np.float64)
        ind = np.argmin(np.abs(slp_alt[0] - downleg_alt)) # match altitudes at start of slp data
        t0 = downleg_time[ind]
        slp_time = np.array(slp_time - slp_time[0] + t0, dtype='datetime64[us]')

        ## save h5 datasets
        day0 = slp_time[0].astype('datetime64[D]')
        us_of_day = (slp_time - day0).astype(np.int64)

        ds = slp_data_h5.create_dataset(f'/36.{rid}/time', data=us_of_day, dtype=np.int64)
        ds.attrs['description'] = f'Microsecond of {day0}'
        ds.attrs['units'] = 'microseconds'

        ds = slp_data_h5.create_dataset(f'/36.{rid}/altitude', data=slp_alt, dtype=np.float64)
        ds.attrs['description'] = 'Downleg altitude'
        ds.attrs['units'] = 'meters'

        ds = slp_data_h5.create_dataset(f'/36.{rid}/ion_density', data=slp_density, dtype=np.float64)
        ds.attrs['description'] = 'Ion density'
        ds.attrs['units'] = 'meters-3'

        ds = slp_data_h5.create_dataset(f'/36.{rid}/electron_temperature', data=slp_electron_temp, dtype=np.float64)
        ds.attrs['description'] = 'Electron temperature'
        ds.attrs['units'] = 'Kelvin'
        
    slp_data_h5.close()


def get_slp_data(rid: int,
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    slp_path = '/home2/vanirsej/sailboat/src/sailboat/data/apep/2023/slp.h5'
    slp_data_h5 = h5py.File(slp_path)[f'36.{rid}']
    time = np.array(slp_data_h5['time'], dtype=np.int64)
    time = np.array([datetime(2023, 10, 14) + timedelta(microseconds=int(us)) for us in time], dtype='datetime64[us]')
    alt = np.array(slp_data_h5['altitude'], dtype=np.float64)
    ion_density = np.array(slp_data_h5['ion_density'], dtype=np.float64)
    electron_temperature = np.array(slp_data_h5['electron_temperature'], dtype=np.float64)
    return time, alt, ion_density, electron_temperature


def fix_ephemeris_txt_files():
    for rid in range(386, 389):
        ephemeris_in_path = f'data/apep/2023/ephemeris/Apep1_x{rid}.txt'
        print(f'Fixing {ephemeris_in_path}...')

        with open(ephemeris_in_path, 'r') as f:
            lines = f.readlines()

        count = 0
        line_num = 0
        with open(ephemeris_in_path, 'w') as f:
            for line in lines:
                line_num += 1
                if len(line.split()) == 4:
                    f.write(line)
                else:
                    count += 1
                    print(f'  Problem at {line_num}')
            print(f'  Found and fixed {count} problematic lines.')


def get_trajectory(rid: int,
                   data_type: str = 'interpolated',
                   cadence_Hz: int = 50
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dtid = 5000 // cadence_Hz
    ephemeris_path = '/home2/vanirsej/sailboat/src/sailboat/data/apep/2023/ephemeris.h5'
    ephemeris_data_h5 = h5py.File(ephemeris_path)[f'36.{rid}/{data_type}']
    time = np.array(ephemeris_data_h5['time'][::dtid], dtype=np.int64)
    time = np.array([datetime(2023, 10, 14) + timedelta(microseconds=int(us)) for us in time], dtype='datetime64[us]')
    glon = np.array(ephemeris_data_h5['longitude'][::dtid], dtype=np.float64)
    glat = np.array(ephemeris_data_h5['latitude'][::dtid], dtype=np.float64)
    alt = np.array(ephemeris_data_h5['altitude'][::dtid], dtype=np.float64)
    return time, glon, glat, alt


def plot_trajectories():
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    for rid in range(386, 389):
        clr = (0, 0, (rid - 386) / 2)
        for data_type in ['raw', 'interpolated']:
            time, glon, glat, alt = get_trajectory(rid, data_type=data_type)

            bad_ids = alt == 0
            bad_ids |= [t > datetime(2023, 10, 15) for t in time]
            time = time[~bad_ids]
            glon = glon[~bad_ids]
            glat = glat[~bad_ids]
            alt = alt[~bad_ids]

            if data_type == 'interpolated':
                lns = '-'
                lnw = 3
            else:
                lns = '-'
                lnw = 1
            lbl = f'36.{rid} ({data_type})'

            axs[0, 0].plot(time, glon, label=lbl, linestyle=lns, color=clr, linewidth=lnw)
            axs[1, 0].plot(time, glat, label=lbl, linestyle=lns, color=clr, linewidth=lnw)
            axs[2, 0].plot(time, alt / 1e3, label=lbl, linestyle=lns, color=clr, linewidth=lnw)
            axs[0, 0].set_ylabel('Geographic longitude (deg)')
            axs[1, 0].set_ylabel('Geographic latitude (deg)')
            axs[2, 0].set_ylabel('Geographic altitude (km)')

            axs[0, 1].plot(glon, glat, label=lbl, linestyle=lns, color=clr, linewidth=lnw)
            axs[1, 1].plot(glat, alt / 1e3, label=lbl, linestyle=lns, color=clr, linewidth=lnw)
            axs[2, 1].plot(glon, alt / 1e3, label=lbl, linestyle=lns, color=clr, linewidth=lnw)
            axs[0, 1].set_xlabel('Geographic longitude (deg)')
            axs[0, 1].set_ylabel('Geographic latitude (deg)')
            axs[1, 1].set_xlabel('Geographic latitude (deg)')
            axs[1, 1].set_ylabel('Geographic altitude (km)')
            axs[2, 1].set_xlabel('Geographic longitude (deg)')
            axs[2, 1].set_ylabel('Geographic altitude (km)')
    
    axs[0, 1].legend(loc='upper right')
    
    for ax in axs[:, 0]:
        ax.grid()
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        ax.set_xlabel('UTC time on 2023-10-14')
    
    for ax in axs[:, 1]:
        ax.grid()

    fig.suptitle('Apep-1 Ephemeris Data')
    fig.tight_layout()
    fig.show()

    plot_path = 'data/apep/2023/ephemeris/ephemeris.png'
    print(f'Saving {plot_path}...')
    fig.savefig(plot_path)

        
def interpolate_trajectory(sim_direc: str,
                           rid: int
                           ):
    
    variable = 'ne'
    out_path = path.join(sim_direc, f'interpolated.h5')

    if path.isfile(out_path):
        data = h5py.File(out_path)[f'/36.{rid}/{variable}'][:]
    else:
        data = interpolate.trajectory(sim_direc, *get_trajectory(rid), variables=variable)
        h5f = h5py.File(out_path, 'w')
        
        ds = h5f.create_dataset(f'/36.{rid}/{variable}', data=data)

        ds.attrs['description'] = f'Interpolated {variable} along Apep-1 trajectory 36.{rid}'
        ds.attrs['units'] = 'meters^-3'

        h5f.close()

    return data

