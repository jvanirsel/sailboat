from os import path, getenv, makedirs

def pbs(
        sim_name: str,
        is_ic: bool = False,
        queue: str = 'normalq',
        num_nodes: int = 1,
        num_procs_per_node: int = 192,
        num_hours: float = 24
        ):
    
    HOME = getenv('HOME')
    GEMINI_ROOT = getenv('GEMINI_ROOT')
    GEMINI_SIM_ROOT = getenv('GEMINI_SIM_ROOT')

    if not all([HOME, GEMINI_ROOT, GEMINI_SIM_ROOT]):
        raise ValueError('One or emore environment variables not found: ' \
        'HOME, GEMINI_ROOT, GEMINI_SIM_ROOT')

    subdirec = ''
    if is_ic:
        subdirec = 'ics'
    sim_path = path.join(GEMINI_SIM_ROOT, subdirec, sim_name)
    log_path = path.join(HOME, 'logs')
    scratch_path = path.join(HOME, 'scratch')
    work_path = path.join(scratch_path, sim_name)

    if not path.isdir(sim_path):
        raise FileNotFoundError(f'Cannot find simulation directory {sim_path}')

    pbs_file = path.join(sim_path, 'submit.pbs')
    out_file = path.join(log_path, sim_name + '.${PBS_JOBID}.out')
    err_file = out_file[:-3] + 'err'
    gemini_bin = path.join(GEMINI_ROOT, 'build', 'gemini.bin')

    pbs_pfx = '#PBS'
    shell_name = '/bin/bash'

    dys, rem = divmod(num_hours, 24)
    hrs, rem = divmod(rem, 1)
    mns, rem = divmod(rem * 60, 1)
    scs = rem * 60
    walltime_str = f'{round(dys):02d}:{round(hrs):02d}:{round(mns):02d}:{round(scs):02d}'

    module_list = ['gcc/8.5.0-gcc-8.5.0-cokvw3c',
                   'openmpi/5.0.2-gcc-8.5.0-diludms',
                   'netlib-lapack/3.11.0-gcc-8.5.0-hlxv33x']

    if not path.isdir(log_path):
        makedirs(log_path)

    with open(pbs_file, 'w') as f:
        f.write('# Command options:\n')
        f.write(f'{pbs_pfx} -N {sim_name}\n')
        f.write(f'{pbs_pfx} -S {shell_name}\n')
        f.write(f'{pbs_pfx} -q {queue}\n')
        f.write(f'{pbs_pfx} -l nodes={num_nodes}:ppn={num_procs_per_node}\n')
        f.write(f'{pbs_pfx} -l walltime={walltime_str}\n')
        f.write(f'{pbs_pfx} -o {out_file}\n')
        f.write(f'{pbs_pfx} -e {err_file}\n')
        f.write(f'{pbs_pfx} -V\n\n')

        f.write('# Load modules:\n')
        f.write('module purge\n')
        for module in module_list:
            f.write(f'module load {module}\n')
        f.write('\n')

        f.write('Commands to run:\n')
        f.write(f'cp -r {sim_path} {scratch_path}\n')
        f.write(f'cp {gemini_bin} {work_path}\n')
        f.write(f'cd {work_path}\n')
        f.write(f'mpiexec gemini.bin . > {sim_name}.out 2> {sim_name}.err\n')
        f.write(f'cp -nr {work_path} {GEMINI_SIM_ROOT}\n\n')
