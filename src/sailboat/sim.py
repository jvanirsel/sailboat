from . import utils, write, plot, GEMINI_SIM_ROOT
from gemini3d import read, model
from pathlib import Path

def setup(
        sim_name: str,
        copy_files: bool = False
        ) -> None:
    
    '''
    Setup for gemini simulation.
    Create and setup equilibrium simulation if not already done.
    Double-check solar activity levels.
    Write PBS script and provide submit command for convenience.
    '''

    # read configuration and equilibrium directory
    sim_direc = Path(GEMINI_SIM_ROOT, sim_name).expanduser()
    cfg = read.config(sim_direc)
    eq_direc = Path(cfg['eq_dir']).expanduser()

    if utils.simulation_finished(sim_direc):
        print(f'Simulation {sim_direc} already finished')
        return

    if not eq_direc.is_dir():
        try:
            eq_direc.mkdir()
        except FileNotFoundError:
            eq_direc.parent.mkdir()
            eq_direc.mkdir()
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
        if copy_files:
            utils.vega_rsync(eq_direc, eq_direc.parent)
        command = 'msub ' + str(utils.collapseuser(eq_direc) / 'submit.pbs')
        print('Equilibrium simulation setup done. ' \
                                f'Please run the following command on VEGA:\n\n{command}\n')
        return
    print('Equilibrium simulation done...')

    # setup simulation
    if not utils.simulation_finished(sim_direc):
        if not utils.simulation_finished_setup(sim_direc):
            model.setup(sim_direc, sim_direc)
        write.pbs(sim_direc)
        if copy_files:
            utils.vega_rsync(sim_direc, sim_direc.parent)
        command = 'msub ' + str(utils.collapseuser(sim_direc) / 'submit.pbs')
        print('Simulation setup finished. ' \
            f'Please run the following command on VEGA:\n\n{command}\n')
        return
    print('Simulation done...')


def copy_from_vega(
        sim_name: str
        ) -> None:
    
    sim_direc = Path(GEMINI_SIM_ROOT, sim_name).expanduser()
    utils.vega_rsync(sim_direc, sim_direc / '*', receive=True)
    

def process(
        sim_name: str
        ) -> None:
    
    '''
    Series of post-processing tasks.
    Plot grid and list of variables. TBD
    '''

    plot.grid(sim_name)

    sim_direc = Path(GEMINI_SIM_ROOT, sim_name)
    plot_direc = sim_direc
    cfg = read.config(sim_direc)
    for xid in [1, 2, 3]:
        for time in cfg['time']:
            dat = read.frame(sim_direc, time)
            plot_direc = plot.quick_summary(cfg, dat, time, xid)
        utils.make_gif(plot_direc)
    
    # plot.variable(sim_name, 'ne')


if __name__ == '__main__':
    from sys import argv
    sim_direc = Path(GEMINI_SIM_ROOT, argv[1])
    if not sim_direc.is_dir():
        raise NotADirectoryError(f'{sim_direc} is not a directory')
    if utils.simulation_finished(sim_direc):
        process(argv[1])
    else:
        setup(argv[1])
    