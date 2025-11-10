from gemini3d import read
from os import path, getenv, listdir
import matplotlib.pyplot as plt
from sail_utils import get_dipole_slice_ids, md, make_gif
import imageio.v3 as iio
import numpy as np


sim_root = getenv('GEMINI_SIM_ROOT')
sim_name = 'apep_2023'
ic_path = path.join(sim_root, 'ics', sim_name + '_eq')
sim_path = path.join(sim_root, sim_name)

cfg = read.config(sim_path)
xg = read.grid(sim_path)

plot_direc = md(path.join(sim_path, 'plots'))
alt_ref = 300e3

for tid in range(10):
    time = cfg['time'][tid]
    time_str = time.strftime('%Y%m%d_%H%M%S')
    dat = read.frame(sim_path, time, var='ne')
    ne_slice = get_dipole_slice_ids(xg, dat, alt_ref)
    plot_path = path.join(plot_direc, f'ne_{time_str}_{alt_ref / 1e3:.0f}km.png')

    plt.pcolormesh(ne_slice['GLON'], ne_slice['GLAT'], ne_slice, shading='auto')
    clb = plt.colorbar()
    clb.set_label('ne (m-3)')
    plt.xlabel('geog lon (°)')
    plt.ylabel('geog lat (°)')
    plt.title(f'{time_str} (~{alt_ref / 1e3:.0f} km)')
    plt.show()
    plt.savefig(plot_path)
    plt.close()
