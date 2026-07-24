from sailboat import SAILBOAT_ROOT, utils
import sailboat.rpa as srpa
import time
import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py

# rpa_direc = SAILBOAT_ROOT / 'data' / 'rpa' / sys.argv[1]
# with h5py.File(rpa_direc / 'measured_09_data.h5') as h5f:
#     grp = h5f['parameters']
#     print(grp['density/fit'][:])
# quit()

# log = SAILBOAT_ROOT / '..' / '..' / 'log.log'
# with open(log) as f:
#     reader = csv.reader(f)
#     t = 0
#     ni = 100000
#     data = np.full((ni, 6), np.nan)
#     for i, row in enumerate(reader):
#         if i >= ni:
#             break
#         z, r, v, dt, dx = map(float, row)
#         t += dt
#         data[i, :] = [t, z, r, v, 100*dt, 100*dx]
#         # print(z, r, v, dt, dx)

# plt.plot(data[:, 1:], label=['z', 'r', 'v', 'dt', 'dx'])
# plt.legend()
# plt.xlim([2000, 3000])
# plt.ylim([-2, 22])
# plt.savefig('log.png')
# # input()
# quit()

# def print_attrs(obj, indent="  "):
#     """Print attributes of an HDF5 object."""
#     for key, value in obj.attrs.items():
#         print(f"{indent}Attribute: {key} = {value}")

# def print_h5(name, obj):
#     """Callback function for visiting HDF5 objects."""
#     print(f"\nObject name: {name}")
#     print(f"Type: {type(obj).__name__}")

#     # Print attributes
#     if len(obj.attrs) > 0:
#         print_attrs(obj)
#     else:
#         print("  No attributes")

#     # If it's a dataset, print its value
#     if isinstance(obj, h5py.Dataset):
#         try:
#             data = obj[()]
#             print("  Data:")
#             print(data)
#         except Exception as e:
#             print(f"  Could not read data: {e}")

# def read_h5_file(file_path):
#     with h5py.File(file_path, "r") as f:
#         print(f"Reading HDF5 file: {file_path}")
#         f.visititems(print_h5)

# if __name__ == "__main__":

# cfg_id = 0
# sffx = ''
# rpa_direc = SAILBOAT_ROOT / 'data' / 'rpa' / sys.argv[1]
# rpa_direc = rpa_direc.expanduser()
# gif_path = rpa_direc / f'config_{cfg_id:02d}{sffx}.gif'
# plot_direc = rpa_direc / f'plots{sffx}'
# utils.make_gif(plot_direc, prefix=f'config_{cfg_id:02d}_step_', filename=gif_path)
# quit()

t0 = time.perf_counter()
raven_direc = SAILBOAT_ROOT / 'data' / 'rpa' / sys.argv[1]
if len(sys.argv) >=2:
    batch_direc = SAILBOAT_ROOT / 'data' / 'rpa' / sys.argv[2]
    srpa.write.batch(raven_direc, batch_direc, [10, 100, 1000], [3, 6, 9, 12], [0, 8, 16, 24], [0.01, 0.05, 0.10, 0.5])
# srpa.write.config_toml(rpa_direc, {'ion_temperature': 999.999, 'beam_velocity': [0.0, 1.0, 2.0]})
srpa.sim.run(raven_direc, do_electrons=False, debug=True, do_example_plots=False, go_fast=True)
# srpa.measure.plasma_parameters(rpa_direc / 'config_10_data.h5')
t1 = time.perf_counter()
print(f'Elapsed time: {(t1 - t0) / 60:.2f} minutes')
quit()

# t0 = time.perf_counter()
# srpa.sim.run(sim_direc, do_electrons=True)
# t1 = time.perf_counter()
# print(f'Elapsed electron time: {t1 - t0:.2f} seconds')

    # h5_file_path = sim_direc / 'config_00_data.h5'  # change to your file path
    # read_h5_file(h5_file_path)

densities = [50, 100, 500]
beam_speeds = []
ion_temperatures = [0.01, 0.05, 0.1]


