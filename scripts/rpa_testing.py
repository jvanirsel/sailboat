from sailboat import SAILBOAT_ROOT
import sailboat.rpa as srpa
import time
import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py

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

t0 = time.perf_counter()
sim_direc = SAILBOAT_ROOT / 'data' / 'rpa' / sys.argv[1]
srpa.sim.run(sim_direc, debug=False)
t1 = time.perf_counter()
print(f'Elapsed time: {t1 - t0:.2f} seconds')

    # h5_file_path = sim_direc / 'config_00_data.h5'  # change to your file path
    # read_h5_file(h5_file_path)


