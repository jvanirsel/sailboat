from sailboat import utils, write, plot, GEMINI_SIM_ROOT
from gemini3d import read, model
from os import path, makedirs

def setup(sim_name: str):
    '''
    Setup for gemini simulation.
    Create and setup equilibrium simulation if not already done.
    Double-check solar activity levels.
    Write PBS script and provide submit command for convenience.
    '''

    # read configuration and equilibrium directory
    sim_direc = path.join(GEMINI_SIM_ROOT, sim_name)
    cfg = read.config(sim_direc)
    eq_direc = cfg['eq_dir']

    if utils.simulation_finished(sim_direc):
        raise FileExistsError('Simulation already done')

    if not path.isdir(eq_direc):
        makedirs(eq_direc)
        print(f'Created equilibrium directory: {eq_direc}')

    # check if activity levels match
    if utils.internet_access():
        utils.check_activity(cfg)
    else:
        print('No internet access; skipped activity levels check...')

    # setup equilibrium simulation
    if not utils.simulation_finished(eq_direc):
        if not utils.simulation_finished_setup(eq_direc):
            write.eq_config(sim_direc)
            model.setup(eq_direc, eq_direc)
        write.pbs(eq_direc)
        command = 'msub ' + path.join(eq_direc, 'submit.pbs')
        raise FileNotFoundError('Equilibrium simulation setup done. ' \
                                f'Please run the following command:\n\n{command}\n')
    print('Equilibrium simulation done...')

    # setup simulation
    if not utils.simulation_finished_setup(sim_direc):
        model.setup(sim_direc, sim_direc)
    write.pbs(sim_direc)
    command = 'msub ' + path.join(sim_direc, 'submit.pbs')
    print('Simulation setup finished. ' \
        f'Run the following command to submit job:\n\n{command}\n')


def process(sim_name: str):
    '''
    Series of post-processing tasks.
    Plot grid and list of variables.
    '''
    plot.grid(sim_name)
    plot.variable(sim_name, 'ne')


if __name__ == '__main__':
    from sys import argv
    sim_direc = path.join(GEMINI_SIM_ROOT, argv[1])
    if utils.simulation_finished(sim_direc):
        process(argv[1])
    else:
        setup(argv[1])
    