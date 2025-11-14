import numpy as np
import h5py
from gemini3d import utils, read
from os import path, getenv, makedirs, listdir
from datetime import datetime
from scipy.constants import h, c
import sailboat

def convert_solar_flux(sim_name: str, nc4_direc: str):

    ## load config data
    sim_root = sailboat.GEMINI_SIM_ROOT
    sim_path = path.join(sim_root, sim_name)
    cfg = read.config(sim_path)

    ## read nc4 filenames
    nc4_filenames = [f for f in listdir(nc4_direc) if f.endswith('.nc4')]
    if not nc4_filenames:
        raise FileNotFoundError(f'No .nc4 files found in {nc4_direc}')

    ## read solflux data directory from config
    try:
        h5_dir = path.join(sim_path, cfg['solfluxdir'])
    except KeyError:
        raise KeyError('solfluxdir not found in config file. Please adjust gemini3d/config.py to include solflux parameters.')
    if not path.exists(h5_dir):
        makedirs(h5_dir)

    ## loop through nc4 files
    first_file = True
    for nc4_filename in nc4_filenames:
        print(f'Processing {nc4_filename}...', end='\r')
        nc4_path = path.join(nc4_direc, nc4_filename)
        nc4_ds = h5py.File(nc4_path)
        nc4_ds_tgcm = nc4_ds['sb_tgcm'] # first 22 tgcm spectral bins match gemini bins

        ## read coordinate information (only once)
        if first_file:
            glat = np.array(nc4_ds['Lat'][:], dtype=np.float64) # degrees
            glon = np.array(nc4_ds['Lon'][:], dtype=np.float64) # degrees
            wvl0 = np.array(nc4_ds_tgcm['Start_wavelength'][:22], dtype=np.float64) * 1e-9 # m
            wvl1 = np.array(nc4_ds_tgcm['End_wavelength'][:22], dtype=np.float64) * 1e-9 # m
            
            ## calculate average photon energy per spectral bin to convert W/m2 to photons/m2/s
            avg_photon_energy = h * c * np.log(wvl1 / wvl0) / (wvl1 - wvl0) # J (<E> = h c <1 / wvl>)

            ## wrap longtiudes to 0 -- 360 for gemini use
            glon = glon % 360
            glon_ids = np.argsort(glon)
            glon = glon[glon_ids]

            first_file = False

        ## read irradiance data
        irr = np.array(nc4_ds_tgcm['msk_irradiance_tlsm'][:, :, :22], dtype=np.float64) # W/m2
        irr = irr / avg_photon_energy # photons/m2/s
        irr = irr[:, glon_ids, :].transpose(1,0,2) # llon x llat x 22
        Iinf = np.array(irr.transpose(2, 1, 0)) # preserve irr.shape when read by fortran

        ## save to hdf5
        h5_date = datetime.strptime(nc4_filename[13:28], '%Y%m%d_%H%M%S')
        h5_name = utils.datetime2stem(h5_date)
        f = h5py.File(path.join(h5_dir, h5_name + '.h5'), 'w')
        h5_ds = f.create_dataset('/Iinf', data=Iinf)
        h5_ds.attrs['units'] = 'photons m^-2 s^-1'
        h5_ds.attrs['description'] = 'masked irradiance in bins defined by solomon & qian (2005)'
        h5_ds.attrs['ref_url'] = 'https://doi.org/10.1029/2005JA011160'
        h5_ds.attrs['wavelength_bins'] = [wvl0, wvl1]
        h5_ds.attrs['wavelength_units'] = 'm'
        f.close()

    ## save simulation size and grid information
    f = h5py.File(path.join(h5_dir, 'simsize.h5'), 'w')
    f.create_dataset('/llat', data=glat.size)
    f.create_dataset('/llon', data=glon.size)
    f.close()

    f = h5py.File(path.join(h5_dir, 'simgrid.h5'), 'w')
    f.create_dataset('/mlat', data=glat) # glat instead of mlat is ok here
    f.create_dataset('/mlon', data=glon)
    f.close()

    nc4_ds.close()
    print('Done' + ' ' * 80)

if __name__ == '__main__':
    # sim_name = 'apep_2023_nux2'
    # nc4_direc = path.join('..', 'apep', '2023', 'fism2_masked')
    from sys import argv
    convert_solar_flux(argv[1], argv[2])