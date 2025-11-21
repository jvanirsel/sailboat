from gemini3d import read
from sailboat import HOME, GEMINI_ROOT
from os import path, makedirs
from datetime import datetime, timedelta

def eq_config(
        sim_direc: str,
        tdur: int = 64800,
        dtout: int = 3600,
        min_lx: int = 192,
        nmf: float = 5e11,
        nme: float = 2e11
        ):
    
    def get_value(line: str):
        value = line.replace(' ', '').replace('\n', '').split('=')[-1]
        if value.isdigit():
            value = int(value)
        elif value.replace('e', '').isdecimal():
            value = float(value)
        return value

    sim_direc = path.normpath(sim_direc)

    cfg = read.config(sim_direc)
    sim_cfg_path = cfg['nml']
    eq_direc = cfg['eq_dir']
    if not path.isdir(eq_direc):
        makedirs(eq_direc)
    eq_cfg_path = path.join(eq_direc, 'config.nml')

    ignore_namelists = ['neutral_perturb',
                        'precip',
                        'efield',
                        'solflux',
                        'fields',
                        'fang',
                        'fang_pars',
                        ]
    ignore_variables = ['setup_functions',
                        ]

    new_lines = []
    do_write = True
    with open(sim_cfg_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if do_write:
                if line[1:-1] in ignore_namelists:
                    do_write = False
                elif line.startswith(tuple(ignore_variables)):
                    pass
                elif line.startswith('ymd'):
                    ymd = datetime.strptime(get_value(line), '%Y,%m,%d')
                elif line.startswith('UTsec0'):
                    time = ymd + timedelta(seconds=int(get_value(line))-tdur)
                    UTsec0 = 3600 * time.hour + 60 * time.minute + time.second
                    new_lines.append(datetime.strftime(time, 'ymd = %Y, %m, %d') + '\n')
                    new_lines.append(f'UTsec0 = {UTsec0:.0f}\n')
                elif line.startswith('tdur'):
                    new_lines.append(f'tdur = {tdur:.0f}\n')
                elif line.startswith('dtout'):
                    new_lines.append(f'dtout = {dtout:.0f}\n')
                elif line.startswith('dtheta'):
                    dtheta = int(get_value(line)) + 5
                    new_lines.append(f'dtheta = {dtheta:.0f}\n')
                elif line.startswith('dphi'):
                    dphi = int(get_value(line)) + 5
                    new_lines.append(f'dphi = {dphi:.0f}\n')
                elif line.startswith('lq ') or line.startswith('lq='):
                    lq = max(int(get_value(line)) / 2 // min_lx * min_lx, min_lx)
                    new_lines.append(f'lq = {lq:.0f}\n')
                elif line.startswith('lp ') or line.startswith('lp='):
                    lp = max(int(get_value(line)) / 2 // min_lx * min_lx, min_lx)
                    new_lines.append(f'lp = {lp:.0f}\n')
                elif line.startswith('lphi'):
                    lphi = int(get_value(line)) // 2
                    new_lines.append(f'lphi = {lphi:.0f}\n')
                elif line.startswith('altmin'):
                    altmin = get_value(line) / 2
                    new_lines.append(f'altmin = {altmin/1e3:.0f}e3\n')
                elif line.startswith('eq_dir'):
                    new_lines.append(f'nmf = {nmf/1e11:.0f}e11\n')
                    new_lines.append(f'nme = {nme/1e11:.0f}e11\n')
                else:
                    new_lines.append(line)
            elif line == '/':
                do_write = True
                new_lines.pop(-1)
    
    with open(eq_cfg_path, 'w') as f:
        f.writelines(new_lines)


def pbs(
        sim_direc: str,
        queue: str = 'normalq',
        num_nodes: int = 1,
        num_procs_per_node: int = 192,
        num_hours: float = 24,
        email: str = ''
        ):

    sim_direc = path.normpath(sim_direc)
    sim_name = path.basename(sim_direc)
    log_direc = path.join(HOME, 'logs')
    scratch_direc = path.join(HOME, 'scratch')
    work_direc = path.join(scratch_direc, sim_name)

    if not path.isdir(sim_direc):
        raise FileNotFoundError(f'Cannot find simulation directory: {sim_direc}')
    if not path.isdir(log_direc):
        makedirs(log_direc)
    if not email:
        email = f'{path.basename(HOME)}@erau.edu'

    pbs_path = path.join(sim_direc, 'submit.pbs')
    stdout_path = path.join(log_direc, sim_name + '.${PBS_JOBID}.out')
    stderr_path = stdout_path[:-3] + 'err'
    gemini_bin_path = path.join(GEMINI_ROOT, 'build', 'gemini.bin')

    pbs_pfx = '#PBS'
    shell_path = '/bin/bash'

    dys, rem = divmod(num_hours, 24)
    hrs, rem = divmod(rem, 1)
    mns, rem = divmod(rem * 60, 1)
    scs = rem * 60
    walltime_str = f'{round(dys):02d}:{round(hrs):02d}:{round(mns):02d}:{round(scs):02d}'

    module_list = ['gcc/8.5.0-gcc-8.5.0-cokvw3c',
                   'openmpi/5.0.2-gcc-8.5.0-diludms',
                   'netlib-lapack/3.11.0-gcc-8.5.0-hlxv33x']

    with open(pbs_path, 'w') as f:
        f.write('# Command options:\n')
        f.write(f'{pbs_pfx} -N {sim_name}\n')
        f.write(f'{pbs_pfx} -S {shell_path}\n')
        f.write(f'{pbs_pfx} -q {queue}\n')
        f.write(f'{pbs_pfx} -l nodes={num_nodes}:ppn={num_procs_per_node}\n')
        f.write(f'{pbs_pfx} -l walltime={walltime_str}\n')
        f.write(f'{pbs_pfx} -M {email}\n')
        f.write(f'{pbs_pfx} -m ef\n')
        f.write(f'{pbs_pfx} -o {stdout_path}\n')
        f.write(f'{pbs_pfx} -e {stderr_path}\n')
        f.write(f'{pbs_pfx} -V\n\n')

        f.write('# Load modules:\n')
        f.write('module purge\n')
        for module in module_list:
            f.write(f'module load {module}\n')
        f.write('\n')

        f.write('Commands to run:\n')
        f.write(f'cp -r {sim_direc} {scratch_direc}\n')
        f.write(f'cp {gemini_bin_path} {work_direc}\n')
        f.write(f'cd {work_direc}\n')
        f.write(f'mpiexec gemini.bin . > {sim_name}.out 2> {sim_name}.err\n')
        f.write(f'cp -nr {work_direc} {path.dirname(sim_direc)}\n')


