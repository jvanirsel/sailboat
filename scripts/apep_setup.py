from gemini3d import model, find, read
from os import getenv, path
from sail_utils import get_activity, plot_all
from datetime import datetime, timedelta
from convert_solar_flux import  convert_solar_flux

sim_root = getenv('GEMINI_SIM_ROOT')
sim_name = 'apep_2023'
ic_path = path.join(sim_root, 'ics', sim_name + '_eq')
sim_path = path.join(sim_root, sim_name)

# convert_solar_flux(sim_name, path.join('..', 'apep', '2023'))
# model.setup(sim_path, sim_path)

cfg = read.config(sim_path)
for tid in range(10):
    dat = read.frame(sim_path, cfg['time'][tid])
    for ci in range(3):
        plot_all(cfg, dat, ci)
