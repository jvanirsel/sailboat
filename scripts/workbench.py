from sailboat import plot, GEMINI_SIM_ROOT, interpolate, apep, read as sread, sim, utils
from gemini3d import read
from os import path
import h5py
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

apep.convert_ephemeris()
quit()

# apep.convert_ephemeris()
# apep.plot_trajectories()
# apep.convert_slp_data()
# quit()

# for rid in [386, 387, 388]:
#     time, _, _, alt = apep.get_trajectory(rid)
#     ind = np.argmax(alt)
#     t = time[ind]
#     sod = (t - t.astype('datetime64[D]')).astype(int)/1e6

#     print('-' * 40 + f' {rid} ' + '-' * 40 + '\n\n.../config.nml:')
#     print(f'  UTsec0 = {round(sod)-600-7200}')
#     print('  tdur = 7200')
#     print('  dtout = 600')

#     print('\n\n..._30s/config.nml:')
#     print(f'  UTsec0 = {round(sod)-600}')
#     print('  tdur = 1200')
#     print('  dtout = 30')
# quit()

# sim_name = 'apep_2023_veia10_30s'
# sim_direc = path.join(GEMINI_SIM_ROOT, sim_name)
# cfg = read.config(sim_direc)
# eq_direc = cfg['eq_dir']

plt.figure(figsize=(12, 12))

for rid in [386, 387, 388]:
    v = (rid - 386) / 2
    clr = (v, 0, 0)

    sim_name = f'apep1_{rid}_veia20_30s'
    sim_direc = path.join(GEMINI_SIM_ROOT, sim_name)
    ne10 = apep.interpolate_trajectory(sim_direc, rid)
    # ne20 = apep.interpolate_trajectory(sim_direc.replace('veia10','veia20'), 386)

    time, glon, glat, alt = apep.get_trajectory(rid)
    _, alt_slp, density_slp, _ = apep.get_slp_data(rid)
    
    apogee_ind = np.argmax(alt)
    alt = alt[apogee_ind:]
    ne10 = ne10[apogee_ind:]

    density_slp = gaussian_filter1d(density_slp, len(density_slp)//100)

    plt.plot(np.log10(ne10), alt/1e3, label=f'36.{rid}', color=clr, linestyle='--')
    # plt.plot(np.log10(ne20), alt/1e3, label='gemini')
    plt.plot(np.log10(density_slp), alt_slp/1e3, label='slp', color=clr)
    plt.xlim([10.5, 12.5])
    plt.ylim([70, 360])
    plt.legend()
    plt.xlabel('log10(ne) [m-3]')
    plt.ylabel('Altitude [km]')
    plt.grid()

    plt.show()
plot_path = f'../../plots/36.{rid}_ne_interp_veia20.png'
print(f'Saving {plot_path}')
plt.savefig(plot_path)
plt.close()