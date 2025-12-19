from sailboat import plot, GEMINI_SIM_ROOT, interpolate, apep, sim, utils
from gemini3d import read
import h5py
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

# sim_name = f'apep1_386_veia20_30s'
sim_name = 'test'
sim_direc = Path(GEMINI_SIM_ROOT, sim_name)
cfg = read.config(sim_direc)
time = cfg['time'][-1]


# dat = read.frame(sim_direc, time)
# for sid in [1, 2, 3]:
#     plot.quick_summary(cfg, dat, time, sid)

# plot.variable(sim_name, 'ne')

_, gdlon, gdlat, gdalt = apep.get_trajectory(386)
plot.grid(sim_name, coord_type='ecef', trajectory=np.vstack([gdlon, gdlat, gdalt]), zoom=True)

# apep.convert_solar_flux(cfg, cfg)

# apep.convert_ephemeris()
# apep.plot_trajectories()

# apep.interpolate_trajectory(sim_direc, 386)

quit()



# apep.convert_ephemeris()
# apep.plot_trajectories()
# apep.convert_slp_data()
# quit()

# for rid in [386, 387, 388]:
#     time, _, _, alt = apep.get_trajectory(rid)
#     ind = np.argmax(alt)
#     t = time[ind]
#     sod = (t - t.astype('datetime64[D]')).astype(int) / 1e6

#     print('-' * 40 + f' {rid} ' + '-' * 40 + '\n.../config.nml:')
#     print(f'UTsec0 = {round(sod)-600-7200}')
#     print('tdur = 7200')
#     print('dtout = 600')

#     print('\n..._30s/config.nml:')
#     print(f'UTsec0 = {round(sod)-600}')
#     print('tdur = 1200')
#     print('dtout = 30\n')
# quit()

# sim_name = 'apep_2023_veia10_30s'
# sim_direc = path.join(GEMINI_SIM_ROOT, sim_name)
# cfg = read.config(sim_direc)
# eq_direc = cfg['eq_dir']

fig, axs = plt.subplots(1, 2, figsize=(12, 12))

for rid in [386, 387, 388]:
    v = (rid - 386) / 2
    clr = (v, 0, 0)

    # get sweeping Langmuir probe data
    _, slp_gdalt, slp_density, slp_electron_temp = apep.get_slp_data(rid)
    slp_density = gaussian_filter1d(slp_density, len(slp_density) // 100)
    slp_electron_temp = gaussian_filter1d(slp_electron_temp, len(slp_electron_temp) // 100)

    # get simulation interpolated data
    sim_name = f'apep1_{rid}_evibcool1_30s'
    # sim_name = f'apep1_{rid}_veia20_30s'
    sim_direc = Path(GEMINI_SIM_ROOT, sim_name)
    sim_data = apep.interpolate_trajectory(sim_direc, rid)
    _, _, _, sim_gdalt = apep.get_trajectory(rid)
    
    apogee_id = np.argmax(sim_gdalt)
    sim_gdalt = sim_gdalt[apogee_id:]
    sim_density = sim_data[apogee_id:, 0]
    sim_electron_temp = sim_data[apogee_id:, 1]
    
    ax = axs[0]
    ax.plot(np.log10(sim_density), sim_gdalt / 1e3, label=f'36.{rid} (gemini)', color=clr, linestyle='--')
    ax.plot(np.log10(slp_density), slp_gdalt / 1e3, label=f'36.{rid} (slp)', color=clr)
    ax.set_xlim(10.5, 12.5)
    ax.set_ylim([70, 360])
    ax.legend()
    ax.set_xlabel('log10 Density [m-3]')
    ax.set_ylabel('Altitude [km]')
    ax.grid()
    
    ax = axs[1]
    ax.plot(sim_electron_temp, sim_gdalt / 1e3, label=f'36.{rid} (gemini)', color=clr, linestyle='--')
    ax.plot(slp_electron_temp, slp_gdalt / 1e3, label=f'36.{rid} (slp)', color=clr)
    # plt.xlim([10.5, 12.5])
    ax.set_ylim([70, 360])
    ax.legend()
    ax.set_xlabel('Electron temperature [K]')
    ax.set_ylabel('Altitude [km]')
    ax.grid()
    
    fig.suptitle(f'APEP 1 Simulation interpolations (apep1_###_{sim_name[10:]})')
    fig.show()

plot_path = f'../../plots/apep1_sim_interpolations_{sim_name[10:]}.png'
print(f'Saving {plot_path}')
fig.savefig(plot_path)
plt.close(fig)