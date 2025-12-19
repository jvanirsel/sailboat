from sailboat import HOME, utils as su
from gemini3d import read, utils
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

def check_sim_progress(
        sim_name: str
        ) -> None:
    # sim_direc_base = path.join(HOME, 'scratch', sim_name)
    scratch_direc = Path(HOME, 'scratch')
    sim_direcs = [d for d in scratch_direc.iterdir() if d.name == sim_name]
    if len(sim_direcs) == 1:
        sim_direc = Path(scratch_direc, sim_direcs[0])
    else:
        raise FileExistsError('Two or more simulation directories found.' \
        ' Please specify further.')

    cfg = read.config(sim_direc)
    sim_times: list[datetime] = cfg['time']

    i = 0
    found_file = False
    file_data = np.full((len(sim_times), 2), np.nan)
    for entry in sim_direc.iterdir():
        if entry.name.endswith('000000.h5'):
            found_file = True
            name = entry.stem
            date_str, sod_str = name.split('_')
            date = datetime.strptime(date_str, '%Y%m%d')
            file_time = date + timedelta(seconds=float(sod_str))
            make_time = datetime.fromtimestamp(entry.stat().st_mtime)
            file_data[i, 0] = file_time.timestamp()
            file_data[i, 1] = make_time.timestamp()
            i += 1
    
    if not found_file:
        raise FileNotFoundError('No output files in simulation')

    file_data = file_data[~np.isnan(file_data[:, 0]), :]
    file_data = file_data[file_data[:, 0].argsort()]
    latest_file_name = utils.datetime2stem(datetime.fromtimestamp(file_data[-1, 0])) + '.h5'
    latest_file_time = datetime.fromtimestamp(file_data[-1, 1])

    x = file_data[:, 0] - file_data[0, 0]
    y = file_data[:, 1] - file_data[0, 1]

    if len(x) < 2:
        raise FileNotFoundError('Need two or more output files')

    a, b = np.polyfit(x, y, 1)
    xq = sim_times[-1].timestamp() - file_data[0, 0]
    yq = a * xq + b

    completion_time = datetime.fromtimestamp(yq + file_data[0, 1])
    print('\n' + '-' * 88)
    print(f'Simulation directory:            {sim_direc}')
    print(f'Average time between files:      {np.mean(np.diff(y))/60:.1f} minutes')
    print(f'Latest simulation file:          {latest_file_name} created at {latest_file_time}')
    if not su.simulation_finished(sim_direc):
        print(f'Estimated completion time:       {completion_time}')
        print(f'Estimated time until completion: {completion_time - datetime.now()}')
        print(f'Estimated total time:            {completion_time - datetime.fromtimestamp(file_data[0, 1])}')
    else:
        print('Simulation completed.')
        print(f'Total time:                 {datetime.fromtimestamp(file_data[-1, 1]) - datetime.fromtimestamp(file_data[0, 1])}')
    print('-' * 88 + '\n')

if __name__ == '__main__':
    from sys import argv
    check_sim_progress(argv[1])