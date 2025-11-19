import numpy as np
import h5py
from gemini3d import utils, read
from os import path, makedirs, listdir
from datetime import datetime
from scipy.constants import h, c
from sailboat import GEMINI_SIM_ROOT

def convert_solar_flux(sim_name: str,
                       solflux_direc: str,
                       in_extension: str = 'nc4'
                       ):

    ## load config data
    sim_root = GEMINI_SIM_ROOT
    sim_direc = path.join(sim_root, sim_name)
    cfg = read.config(sim_direc)

    ## read input filenames
    solflux_filenames = [f for f in listdir(solflux_direc) if f.endswith('.' + in_extension)]
    if not solflux_filenames:
        raise FileNotFoundError(f'No .nc4 files found in {solflux_direc}')

    ## read solflux data directory from config
    try:
        sim_solflux_direc = path.join(sim_direc, cfg['solfluxdir'])
    except KeyError:
        raise KeyError('solfluxdir not found in config file. Please adjust gemini3d/config.py to include solflux namelist.')
    if not path.isdir(sim_solflux_direc):
        makedirs(sim_solflux_direc)

    ## loop through nc4 files
    first_file = True
    for solflux_filename in solflux_filenames:
        print(f'Processing {solflux_filename}...', end='\r')
        solflux_path = path.join(solflux_direc, solflux_filename)
        solflux_data_nc4 = h5py.File(solflux_path)
        solflux_tgcm_data_nc4 = solflux_data_nc4['sb_tgcm'] # first 22 tgcm spectral bins match gemini bins

        ## read coordinate information (only once)
        if first_file:
            glat = np.array(solflux_data_nc4['Lat'][:], dtype=np.float64) # degrees
            glon = np.array(solflux_data_nc4['Lon'][:], dtype=np.float64) # degrees
            wvl0 = np.array(solflux_tgcm_data_nc4['Start_wavelength'][:22], dtype=np.float64) * 1e-9 # m
            wvl1 = np.array(solflux_tgcm_data_nc4['End_wavelength'][:22], dtype=np.float64) * 1e-9 # m
            
            ## calculate average photon energy per spectral bin to convert W/m2 to photons/m2/s
            avg_photon_energy = h * c * np.log(wvl1 / wvl0) / (wvl1 - wvl0) # J (<E> = h c <1 / wvl>)

            ## wrap longtiudes to 0 -- 360 for gemini use
            glon = glon % 360
            glon_ids = np.argsort(glon)
            glon = glon[glon_ids]

            first_file = False

        ## read irradiance data
        irr = np.array(solflux_tgcm_data_nc4['msk_irradiance_tlsm'][:, :, :22], dtype=np.float64) # W/m2
        irr = irr / avg_photon_energy # photons/m2/s
        irr = irr[:, glon_ids, :].transpose(1,0,2) # llon x llat x 22
        Iinf = np.array(irr.transpose(2, 1, 0)) # preserve irr.shape when read by fortran

        ## save to hdf5
        h5_date = datetime.strptime(solflux_filename[13:28], '%Y%m%d_%H%M%S')
        h5_name = utils.datetime2stem(h5_date)
        f = h5py.File(path.join(sim_solflux_direc, h5_name + '.h5'), 'w')
        h5_ds = f.create_dataset('/Iinf', data=Iinf)
        h5_ds.attrs['units'] = 'photons m^-2 s^-1'
        h5_ds.attrs['description'] = 'masked irradiance in bins defined by solomon & qian (2005)'
        h5_ds.attrs['ref_url'] = 'https://doi.org/10.1029/2005JA011160'
        h5_ds.attrs['wavelength_bins'] = [wvl0, wvl1]
        h5_ds.attrs['wavelength_units'] = 'm'
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

    solflux_data_nc4.close()
    print('Done' + ' ' * 80)