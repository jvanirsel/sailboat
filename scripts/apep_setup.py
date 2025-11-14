from sailboat import write, GEMINI_SIM_ROOT
from gemini3d import model
from os import path

sim_name = 'apep_2023_hires'
sim_path = path.join(GEMINI_SIM_ROOT, sim_name)
ic_path = path.join(GEMINI_SIM_ROOT, 'ics', sim_name + '_eq')
# write.config('testing', is_ic=True)
write.pbs(sim_name + '_eq', num_hours=24, num_nodes=2, is_ic=True)
model.setup(ic_path, ic_path)