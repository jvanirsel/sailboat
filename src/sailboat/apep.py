from sailboat import GEMINI_SIM_ROOT
from gemini3d import utils
from os import path, makedirs, listdir
from datetime import datetime, timedelta
from scipy.constants import h, c
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
        h5_ds.attrs['units'] = 'photons m^-2 s^-1'
        h5_ds.attrs['description'] = 'masked irradiance in bins defined by solomon & qian (2005)'
        h5_ds.attrs['ref_url'] = 'https://doi.org/10.1029/2005JA011160'
        h5_ds.attrs['wavelength_bins'] = [wvl0, wvl1]
        h5_ds.attrs['wavelength_units'] = 'm'
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