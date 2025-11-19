from sailboat import write, GEMINI_SIM_ROOT
from gemini3d import model
from os import path

sim_name = 'apep_2023'
sim_direc = path.join(GEMINI_SIM_ROOT, sim_name)
ic_direc = path.join(GEMINI_SIM_ROOT, 'ics', sim_name + '_eq')
# write.config('testing', is_ic=True)
# write.pbs(ic_direc, num_hours=24, num_nodes=2)
write.ic_config(sim_direc)
# model.setup(ic_direc, ic_direc)