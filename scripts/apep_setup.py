from sailboat import write, utils, GEMINI_SIM_ROOT
from gemini3d import model, read
from os import path, makedirs

sim_name = 'testing'
sim_direc = path.join(GEMINI_SIM_ROOT, sim_name)

# read configuration and equilibrium directory
cfg = read.config(sim_direc)
eq_direc = cfg['eq_dir']

if utils.simulation_finished(sim_direc):
    raise FileExistsError('Simulation already done')

if not path.isdir(eq_direc):
    makedirs(eq_direc)
    print(f'Created equilibrium directory: {eq_direc}')

if utils.internet_access():
    utils.check_activity(cfg)
else:
    print('No internet access; skipped activity levels check...')

if not utils.simulation_finished(eq_direc):
    if not utils.simulation_finished_setup(eq_direc):
        write.eq_config(sim_direc)
        model.setup(eq_direc, eq_direc)
    write.pbs(eq_direc)
    command = 'msub ' + path.join(eq_direc, 'submit.pbs')
    raise FileNotFoundError(f'Please run the following command:\n\n{command}\n')

print('Equilibrium simulation done...')

if not utils.simulation_finished_setup(sim_direc):
    model.setup(sim_direc, sim_direc)
write.pbs(sim_direc)

command = 'msub ' + path.join(sim_direc, 'submit.pbs')
print('Simulation setup finished.' \
      'Run the following command to submit job:\n\n{command}\n')

