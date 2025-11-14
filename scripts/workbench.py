from sailboat import plot, GEMINI_SIM_ROOT
from gemini3d import read
from os import path

sim_name = 'apep_2023_nux2'
sim_path = path.join(GEMINI_SIM_ROOT, sim_name)
cfg = read.config(sim_path)
times = cfg['time']

for time in times:
    dat = read.frame(sim_path, time)
    for coord in range(1, 4):
        plot.quick_summary(cfg, dat, time, slice_coord=coord)
# plot.grid(sim_name)
