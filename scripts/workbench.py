from sailboat import plot, GEMINI_SIM_ROOT, interpolate, apep, read as sread, sim
from gemini3d import read
from os import path
import h5py
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

sim_name = 'apep_2023_veia20'
sim_direc = path.join(GEMINI_SIM_ROOT, sim_name)
cfg = read.config(sim_direc)
eq_direc = cfg['eq_dir']

apep.convert_solar_flux(cfg, cfg, solflux_in_direc='/home2/vanirsej/sailboat/src/sailboat/data/apep/2023/fism2_masked')

# print(cfg.keys())

# xg_comp = read.grid(eq_direc)
# plot.grid(sim_name, xg_comp)
quit()

# xg = read.grid(sim_direc)
# xg_eq = read.grid(eq_direc)

# plot.grid(sim_name, xg_eq)


# times = cfg['time']

ephemeris_path = '/home2/vanirsej/sailboat/src/sailboat/data/apep/2023/ephemeris.h5'
ephemeris_data_h5 = h5py.File(ephemeris_path)['36.386/interpolated']
# # dtid = 30 * 5000
dtid = 1
time = np.array(ephemeris_data_h5['time'][::dtid], dtype=np.int64)
# glon = ephemeris_data_h5['longitude'][::dtid]
# glat = ephemeris_data_h5['latitude'][::dtid]
alt = ephemeris_data_h5['altitude'][::dtid]
# time = np.datetime64('2023-10-14') + time.astype('timedelta64[us]')
time = [datetime(2023, 10, 14) + timedelta(microseconds=int(us)) for us in time]

def UTsec(time:datetime):
    print(time.hour * 3600 + time.minute * 60 + time.second)

UTsec(time[0])
UTsec(time[np.argmax(alt)])
UTsec(time[-1])

# ne = interpolate.trajectory(sim_direc, time, glon, glat, alt, 'ne')

# plt.plot(ne, alt)
# plt.show()
# plt.savefig('../plots/36.386_ne_interp.png')


# for time in times:
#     dat = read.frame(sim_direc, time)
#     for coord in range(1, 4):
#         plot.quick_summary(cfg, dat, time, slice_coord=coord)
# plot.grid(sim_name)
