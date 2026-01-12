from sailboat import plot, GEMINI_SIM_ROOT, interpolate, apep, sim, utils, SAILBOAT_ROOT
from gemini3d import read
import h5py
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

# sim_name = f'apep1_387_unmask'
# # sim_name = 'test'
# sim_direc = Path(GEMINI_SIM_ROOT, sim_name)
# cfg = read.config(sim_direc)
# time = cfg['time'][-1]

# time0 = cfg['time'][0]
# t0 = int(time0.hour * 3600 + time0.minute * 60 + time0.second)
# t1 = int(time.hour * 3600 + time.minute * 60 + time.second)
# # t0 = (time0 - time0.astype("datetime64[D]")).astype("timedelta64[s]").astype(int)
# # t1 = (time - time.astype("datetime64[D]")).astype("timedelta64[s]").astype(int)

# solflux_direc = Path(sim_direc, 'inputs', 'solflux')
# solfux_simgrid_path = Path(solflux_direc, 'simgrid.h5')
# with h5py.File(solfux_simgrid_path, 'r') as simgrid_data:
#     lat = np.array(simgrid_data['mlat'], dtype=np.float64)
#     lon = np.array(simgrid_data['mlon'], dtype=np.float64)
# for t in range(t0, t1, 1200):
#     solflux_path = Path(solflux_direc, f'20231014_{t:05d}.000000.h5')
#     with h5py.File(solflux_path, 'r') as solflux_data:
#         irr = np.transpose(np.array(solflux_data['Iinf'], dtype=np.float64), (1, 2, 0))
#         print(irr.shape)

#         fig, axs = plt.subplots(5, 5, figsize=(16, 12))

#         for i in range(irr.shape[2]):
#             ax = axs[i//5, i%5]
#             im = ax.pcolormesh(lon, lat, np.log10(irr[:, :, i]), clim=(5, 15), shading='auto')
#             # ax.set_xlim(200, 320)
#             # ax.set_ylim(-70, 70)
#             fig.colorbar(im, ax=ax)
#             # ax.colorbar()
#         fig.show()
#         fig.savefig(SAILBOAT_ROOT / f'../../plots/irr_{sim_name}_20231014_{t:05d}.png')
# quit()

# for t in range(140000, 210000, 10000):
#     solflux_path = f'/home2/vanirsej/sailboat/src/sailboat/data/apep/2023/fism2_masked/FISM2_MASKED_20231014_{t}.nc4'
#     with h5py.File(solflux_path, 'r') as solflux_data:
#         lat = np.array(solflux_data['Lat'], dtype=np.float64)
#         lon = np.array(solflux_data['Lon'], dtype=np.float64) + 180.0
#         iseclipse = np.array(solflux_data['iseclipse'], dtype=np.float64)
#         solflux_tgcm_data = solflux_data['sb_tgcm']
#         assert isinstance(solflux_tgcm_data, h5py.Group)

#         irr = np.array(solflux_tgcm_data['msk_irradiance_tlsm'], dtype=np.float64)
#         irr = irr[:, :, :22]
#         print(irr.shape)

#         fig, axs = plt.subplots(5, 5, figsize=(16, 12))

#         for i in range(irr.shape[2]):
#             ax = axs[i//5, i%5]
#             im = ax.pcolormesh(lon, lat, np.log10(irr[:, :, i]), clim=(-7.5, -3.1), shading='auto')
#             # ax.set_xlim(200, 320)
#             # ax.set_ylim(-70, 70)
#             fig.colorbar(im, ax=ax)
#             # ax.colorbar()
#         axs[-1, -1].pcolormesh(lon, lat, iseclipse, shading='auto')
#         fig.show()
#         fig.savefig(f'../../plots/irr_{t}.png')
# quit()

# dat = read.frame(sim_direc, time)
# for sid in [1, 2, 3]:
#     plot.quick_summary(cfg, dat, time, sid)

# plot.variable(sim_name, 'ne')

# _, gdlon, gdlat, gdalt = apep.get_trajectory(386)
# plot.grid(sim_name, coord_type='ecef', trajectory=np.vstack([gdlon, gdlat, gdalt]), zoom=False)

# apep.convert_solar_flux(cfg, cfg)

# apep.convert_ephemeris()
# apep.plot_trajectories()

# apep.interpolate_trajectory(sim_direc, 386)

# quit()



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
sim_name = ''
for rid in [386, 387, 388]:
    v = (rid - 386) / 2
    clr = (v, 0, 0)

    # get sweeping Langmuir probe data
    _, slp_gdalt, slp_density, slp_electron_temp = apep.get_slp_data(rid)
    slp_density = gaussian_filter1d(slp_density, len(slp_density) // 100)
    slp_electron_temp = gaussian_filter1d(slp_electron_temp, len(slp_electron_temp) // 100)

    # get simulation interpolated data
    # sim_name = f'apep1_{rid}_evibcool1_30s'
    # sim_name = f'apep1_{rid}_veia20_30s'
    sim_name = f'apep1_{rid}_hires_30s'
    # sim_name = f'apep1_{rid}_autoirr_30s'
    # sim_name = f'apep1_{rid}_unmask_30s'
    sim_direc = Path(GEMINI_SIM_ROOT, sim_name)
    sim_data = apep.interpolate_trajectory(sim_direc, rid)
    _, _, _, sim_gdalt = apep.get_trajectory(rid)
    
    apogee_id = np.argmax(sim_gdalt)
    sim_gdalt = sim_gdalt[apogee_id:]
    sim_density = sim_data[apogee_id:, 0]
    sim_electron_temp = sim_data[apogee_id:, 1]

    # print(f'\nrid: {rid}')
    # for ds in [sim_density, slp_density, sim_electron_temp, slp_electron_temp]:
    #     print(np.min(ds), np.max(ds), ds.shape)
    
    ax = axs[0]
    ax.plot(np.log10(sim_density), sim_gdalt / 1e3, label=f'36.{rid} (gemini)', color=clr, linestyle='--')
    ax.plot(np.log10(slp_density), slp_gdalt / 1e3, label=f'36.{rid} (slp)', color=clr)
    ax.set_xlim(9.9, 12.5)
    ax.set_ylim([70, 360])
    ax.legend()
    ax.set_xlabel('log10 Density [m-3]')
    ax.set_ylabel('Altitude [km]')
    ax.grid()
    
    ax = axs[1]
    ax.plot(sim_electron_temp, sim_gdalt / 1e3, label=f'36.{rid} (gemini)', color=clr, linestyle='--')
    ax.plot(slp_electron_temp, slp_gdalt / 1e3, label=f'36.{rid} (slp)', color=clr)
    ax.set_xlim([200, 2300])
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