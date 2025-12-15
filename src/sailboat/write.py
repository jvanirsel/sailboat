from gemini3d import read
from sailboat import HOME, GEMINI_ROOT
from datetime import datetime, timedelta
from pathlib import Path

def eq_config(
        sim_direc: Path,
        tdur: int = 64800,
        dtout: int = 3600,
        min_lx: int = 192,
        nmf: float = 5e11,
        nme: float = 2e11
        ) -> None:
    
    def get_value(line: str) -> str | int | float:
        value = line.replace(' ', '').replace('\n', '').split('=')[-1]
        if value.isdigit():
            value = int(value)
        elif value.replace('e', '').isdecimal():
            value = float(value)
        return value

    cfg = read.config(sim_direc)
    sim_cfg_path = Path(cfg['nml'])
    eq_direc = Path(cfg['eq_dir'])
    eq_direc.mkdir(exist_ok=True)
    eq_cfg_path = Path(eq_direc, 'config.nml')

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
    ymd = datetime.today()
    found_ymd = False
    with open(sim_cfg_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if do_write:
                if line[1:-1] in ignore_namelists:
                    do_write = False
                elif line.startswith(tuple(ignore_variables)):
                    pass
                elif line.startswith('ymd'):
                    ymd = datetime.strptime(str(get_value(line)), '%Y,%m,%d')
                    found_ymd = True
                elif line.startswith('UTsec0'):
                    if not found_ymd:
                        raise ValueError('Please set "ymd" prior to "UTsec0" in config.nml')
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
                    altmin = float(get_value(line)) / 2
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
        sim_direc: Path,
        queue: str = 'normalq',
        num_nodes: int = 1,
        num_procs_per_node: int = 192,
        num_hours: float = 24,
        email: str = ''
        ) -> None:

    timestamp = datetime.strftime(datetime.now(), '_%Y%m%dT%H%M%S')

    sim_name = sim_direc.name
    log_direc = Path(HOME, 'logs')
    scratch_direc = Path(HOME, 'scratch')
    work_direc = Path(scratch_direc, sim_name + timestamp)

    if not sim_direc.is_dir():
        raise NotADirectoryError(f'Cannot find simulation directory: {sim_direc}')
    log_direc.mkdir(exist_ok=True)
    if not email:
        email = f'{HOME.name}@erau.edu'

    pbs_path = Path(sim_direc, 'submit.pbs')
    stdout_path = Path(log_direc, sim_name + '.${PBS_JOBID}.out')
    stderr_path = stdout_path.with_suffix('.err')
    gemini_bin_path = Path(GEMINI_ROOT, 'build', 'gemini.bin')

    pbs_pfx = '#PBS'
    shell_path = Path('/bin','bash')

    days, rem = divmod(num_hours, 24)
    hours, rem = divmod(rem, 1)
    mins, rem = divmod(rem * 60, 1)
    secs = rem * 60
    walltime_str = f'{round(days):02d}:{round(hours):02d}:{round(mins):02d}:{round(secs):02d}'

    module_list = [
        'gcc/8.5.0-gcc-8.5.0-cokvw3c',
        'openmpi/5.0.2-gcc-8.5.0-diludms',
        'netlib-lapack/3.11.0-gcc-8.5.0-hlxv33x'
        ]

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
        f.write(f'mkdir {work_direc}\n')
        f.write(f'cp -r {sim_direc}/* {work_direc}\n')
        f.write(f'cp {gemini_bin_path} {work_direc}\n')
        f.write(f'cd {work_direc}\n')
        f.write(f'mpiexec gemini.bin . > {sim_name}.out 2> {sim_name}.err\n')
        f.write(f'cp -nr {work_direc}/* {sim_direc}\n')


