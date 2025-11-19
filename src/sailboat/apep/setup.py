from gemini3d import read, find
from gemini3d.grid import tilted_dipole
from os import path, makedirs
import sailboat

sim_root = sailboat.GEMINI_SIM_ROOT
sim_name = 'apep_2023_nux2_test'
sim_direc = path.join(sim_root, sim_name)

cfg = read.config(sim_direc)
eq_direc = cfg['eq_dir']

if path.isfile(path.join(eq_direc, 'config.nml')):
    cfg_eq = read.config(eq_direc)
    time = cfg_eq['time'][-1]
    if path.isfile(find.frame(eq_direc, time)):
        print('Final equilibrium output found...')
    else:
        raise FileNotFoundError(f'Please run equilibrium simultation {eq_direc}')
else:
    sailboat.write.pbs(eq_direc, is_eq=True)

# xg_eq = read.grid(path_eq)
# xg_sim = tilted_dipole.tilted_dipole3d(cfg_sim)
# xg_sim = read.grid(path_sim)

# model.setup(path_eq, path_eq)
# model.setup(path_sim, path_sim)

# apep_ephem = h5py.File('../apep/2023/ephemeris.h5')['36.386/interpolated']
# traj = np.array((apep_ephem['longitude'][:], apep_ephem['latitude'][:], apep_ephem['altitude'][:]))

# plot_grid(cfg_eq, xg_eq, xg_compare=xg_sim, trajectory=traj)
# plot_grid(cfg_sim, xg_sim, trajectory=traj)


# convert_solar_flux(sim_name, '/home2/vanirsej/sailboat/apep/2023/fism2_masked')


# for time in cfg['time']:
#     dat = read.frame(sim_direc, time)
#     for ci in range(3):
#         plot_all(cfg, dat, ci)
    
# make_gif('/home2/vanirsej/gemini/sims/apep_2023/plots/all_vars_x1_slice')
# make_gif('/home2/vanirsej/gemini/sims/apep_2023/plots/all_vars_x2_slice')
# make_gif('/home2/vanirsej/gemini/sims/apep_2023/plots/all_vars_x3_slice')

# make_gif('/home2/vanirsej/gemini/sims/apep_2023/plots/', suffix='_300km.png')

# t0 = 51000
# t1 = 0 * 24 * 3600 + 54600
# t2 = t0 + 26400
# m0 = 8 * 60 + 19
# m1 = 20 * 60 + 28
# print((t2 - t1) * (m1 - m0) / (t1 - t0) / 60)