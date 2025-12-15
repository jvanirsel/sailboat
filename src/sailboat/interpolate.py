from sailboat import RE
from gemini3d import read
from gemini3d.grid import convert
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from pathlib import Path
import xarray as xr

def trajectory(
        sim_direc: Path,
        traj_times: np.ndarray,
        traj_gdlons: np.ndarray,
        traj_gdlats: np.ndarray,
        traj_gdalts: np.ndarray,
        variables: set[str]
        ) -> np.ndarray:
    
    cfg = read.config(sim_direc)
    xg = read.grid(sim_direc, var={'x1', 'x2', 'x3'})
    sim_times = np.array(cfg['time'], dtype='datetime64[us]')
    var_dict = {'electron_density': 'ne',
                'electron_temperature': 'Te'}
    variables_gemini = {var_dict[v] for v in variables}

    if 'lq' not in cfg.keys():
        raise KeyError('interpolate.trajectory currently not working for cartesian grids')

    sim_qs = xg['x1'][2:-2]
    sim_ps = xg['x2'][2:-2]
    sim_phis = xg['x3'][2:-2]
    
    tid0_old = None
    dat0_old = xr.Dataset()
    dat1_old = xr.Dataset()
    dat_out = np.empty((len(traj_times), len(variables)))
    for did, (traj_time, traj_gdlon, traj_gdlat, traj_gdalt) in enumerate(
        zip(traj_times, traj_gdlons, traj_gdlats, traj_gdalts)):

        print(f'Current trajectory time: {traj_time}', end='\r')

        # convert input trajectory data to gemini dipole coordinates
        traj_phi, traj_theta = convert.geog2geomag(traj_gdlon, traj_gdlat)
        traj_rad = RE + traj_gdalt
        traj_q = ((RE / traj_rad) ** 2) * np.cos(traj_theta)
        traj_p = (traj_rad / RE) / ( np.sin(traj_theta)**2 )

        # tid = np.argmin(np.abs([traj_time - t for t in sim_times]))
        assert(isinstance(traj_time, np.datetime64))
        tid = int(np.argmin(np.abs(sim_times - traj_time)))
        dt = sim_times[tid] - traj_time
        
        # next nearest time is later
        if dt < 0:
            tid0 = tid
            tid1 = tid + 1
        # next nearest time is earlier
        else:
            tid0 = tid - 1
            tid1 = tid

        # tid0 = max(tid - 1, 0)
        # tid1 = min(tid, len(sim_times) - 1)

        # # if the nearest is earlier, flip order
        # if sim_times[tid] < traj_time:
        #     tid0 = tid
        #     tid1 = min(tid + 1, len(sim_times) - 1)

        # read new data only if tid has shifted
        time0: np.datetime64 = sim_times[tid0]
        time1: np.datetime64 = sim_times[tid1]

        if tid0 != tid0_old:
            dat0 = read.frame(sim_direc, time0.astype(datetime), var=variables_gemini)
            dat1 = read.frame(sim_direc, time1.astype(datetime), var=variables_gemini)
            dat0_old = dat0
            dat1_old = dat1
        else:
            dat0 = dat0_old
            dat1 = dat1_old
        tid0_old = tid0

        for vid, variable in enumerate(variables):
            d0 = dat0[var_dict[variable]]
            d1 = dat1[var_dict[variable]]

            interp0 = RegularGridInterpolator((sim_qs, sim_ps, sim_phis), d0,
                                            method='linear', bounds_error=False, fill_value=np.nan)
            interp1 = RegularGridInterpolator((sim_qs, sim_ps, sim_phis), d1,
                                            method='linear', bounds_error=False, fill_value=np.nan)
            
            dat0_interp = interp0([traj_q, traj_p, traj_phi])
            dat1_interp = interp1([traj_q, traj_p, traj_phi])
            dat_out[did, vid] = dat0_interp + (dat1_interp - dat0_interp) * (traj_time - time0) / (time1 - time0)
                
    print('Done interpolating simulation data...' + ' ' * 40)

    return dat_out