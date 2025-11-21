from sailboat import RE
from gemini3d import read
from gemini3d.grid import convert
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator
import numpy as np

def trajectory(
        sim_direc: str,
        traj_times: np.datetime64,
        traj_glons: np.ndarray,
        traj_glats: np.ndarray,
        traj_galts: np.ndarray,
        variables: str
        ) -> np.ndarray:
    
    cfg = read.config(sim_direc)
    xg = read.grid(sim_direc, var=['x1', 'x2', 'x3'])
    sim_times = np.array(cfg['time'], dtype='datetime64[us]')

    sim_qs = xg['x1'][2:-2]
    sim_ps = xg['x2'][2:-2]
    sim_phis = xg['x3'][2:-2]
    
    tid0_old = -1
    dat_id = 0
    dat_out = np.empty(np.shape(traj_times))
    for traj_time, traj_glon, traj_glat, traj_galt in zip(traj_times, traj_glons, traj_glats, traj_galts):
        print(f'Current trajectory time: {traj_time}')

        # convert input trajectory data to gemini dipole coordinates
        traj_phi, traj_theta = convert.geog2geomag(traj_glon, traj_glat)
        traj_rad = RE + traj_galt
        q_traj = (RE / traj_rad)**2 * np.cos(traj_theta)
        p_traj = (traj_rad / RE) / ( np.cos(traj_theta)**2 )

        tid = np.argmin(np.abs([traj_time - t for t in sim_times]))
        dt = sim_times[tid] - traj_time

        # next nearest time is later
        if dt < 0:
            tid0 = tid
            tid1 = tid + 1
        # next nearest time is earlier
        else:
            tid0 = tid - 1
            tid1 = tid
        
        print(f't0: {sim_times[tid0]}')
        print(f't:  {traj_time}')
        print(f't1: {sim_times[tid1]}')

        # read new data only if tid has shifted
        time0 = sim_times[tid0]
        time1 = sim_times[tid1]
        if tid0 != tid0_old:
            dat0 = read.frame(sim_direc, time0.astype(datetime), var=variables)[variables]
            dat1 = read.frame(sim_direc, time1.astype(datetime), var=variables)[variables]
            interp0 = RegularGridInterpolator((sim_qs, sim_ps, sim_phis), dat0,
                                            method='linear', bounds_error=False, fill_value=None)
            interp1 = RegularGridInterpolator((sim_qs, sim_ps, sim_phis), dat1,
                                            method='linear', bounds_error=False, fill_value=None)
        tid0_old = tid0

        data0 = interp0([q_traj, p_traj, traj_phi])
        data1 = interp1([q_traj, p_traj, traj_phi])
        dat_out[dat_id] = data0 + (data1 - data0) *(traj_time - time0) / (time1 - time0)
        dat_id += 1
    
    return dat_out