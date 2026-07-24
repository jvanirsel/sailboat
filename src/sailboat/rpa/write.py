from . import RPA, Plasma
import h5py
from pathlib import Path
import numpy as np

def config_data(
        save_path: Path,
        cfg: dict,
        rpa: RPA,
        plasma: Plasma
        ) -> None:

    # when writing config data, overwrite file, otherwise append
    with h5py.File(save_path.expanduser(), 'w') as h5f:

        group = '/simulation/'
        h5ds(h5f, group + 'num_rays', cfg['num_rays'], np.int64, 'Number of simulated rays', 'n/a')
        h5ds(h5f, group + 'num_sweep_steps', cfg['num_sweeps'], np.int64, 'Number of sweep steps', 'n/a')
        h5ds(h5f, group + 'max_steps', cfg['max_steps'], np.int64, 'Maximum number of time steps per ray', 'n/a')
        h5ds(h5f, group + 'max_distance', cfg['dx_max'], np.float64, 'Maximum distance per time step', 'millimeter')
        h5ds(h5f, group + 'time_step_factor', cfg['dt_factor'], np.float64, 'time step x curvature x speed (dt x kappa x v)', 'n/a')

        group '/rpa/'
        h5ds(h5f, group + 'noise_level', rpa.noise_level, str, '2-sigma random noise level per anode', 'nanoampere')

        group = '/rpa/geometry/'
        h5ds(h5f, group + 'aperture_shape', rpa.aperture_shape, str, 'Aperture shape', 'n/a')
        h5ds(h5f, group + 'aperture_size', rpa.aperture_size, np.float64, 'Aperture side length', 'millimeter')
        h5ds(h5f, group + 'aperture_area', rpa.aperture_area, np.float64, 'Aperture area', 'millimeter^2')
        h5ds(h5f, group + 'sensor_size', rpa.sensor_size, np.float64, 'Sensor side length', 'millimeter')
        h5ds(h5f, group + 'depth', rpa.depth, np.float64, 'Distance from aperture to sensor', 'millimeter')

        group = '/rpa/screens/'
        h5ds(h5f, group + 'locations', rpa.get_locations(), np.float64, 'Screen distances from aperture', 'millimeter')
        h5ds(h5f, group + 'voltages', rpa.get_voltages(), np.float64, 'Screen voltages', 'volt')
        h5ds(h5f, group + 'number', rpa.get_voltages().size - 1, np.int64, 'Number of screens', 'n/a')
        h5ds(h5f, group + 'opacity', rpa.screen_opacity, np.float64, 'Opactity of one screen (0 = opaque)', 'n/a')

        group = '/rpa/sweep/'
        h5ds(h5f, group + 'index', rpa.sweep_screen_id, np.int64, 'Index of sweeping screen', 'n/a')
        h5ds(h5f, group + 'length', rpa.sweep_len, np.int64, 'Number of sweep steps', 'n/a')
        h5ds(h5f, group + 'voltages', rpa.get_sweep_voltages(), np.float64, 'Sweep voltages', 'volt')

        group = '/plasma/'
        h5ds(h5f, group + 'density', plasma.N[0], np.float64, 'Ion and electron density', 'millimeter^-3')
        h5ds(h5f, group + 'temperature/ion', plasma.Ti[0], np.float64, 'Ion temperature', 'electronvolt')
        h5ds(h5f, group + 'temperature/electron', plasma.Te[0], np.float64, 'Electron temperature', 'electronvolt')
        h5ds(h5f, group + 'mass', plasma.Mi, np.float64, 'Ion temperature', 'electronvolt microsecond^2 / millimeter^2')
        h5ds(h5f, group + 'charge', plasma.Q, np.float64, 'Ion charge', 'femtocoulomb')
        h5ds(h5f, group + 'ionization_state', plasma.Z, np.int64, 'Ionization state of ions', 'elementary charge')
        h5ds(h5f, group + 'beam/velocity', plasma.V, np.float64, 'Plasma beam velocity (ux, uy, uz)', 'millimeter microsecond^-1')
        h5ds(h5f, group + 'beam/energy', plasma.K, np.float64, 'Plasma beam kinetic energy', 'electronvolt')
        h5ds(h5f, group + 'debye_length', plasma.lambdaD, np.float64, 'Total plasma Debye length', 'millimeter')
        h5ds(h5f, group + 'magnetic_field', plasma.B, np.float64, 'Background magnetic field (Bx, By, Bz)', 'microtesla')

        group = '/plasma/background/'
        h5ds(h5f, group + 'density', plasma.N[1], np.float64, 'Ion and electron density', 'millimeter^-3')
        h5ds(h5f, group + 'temperature/ion', plasma.Ti[1], np.float64, 'Ion temperature', 'electronvolt')
        h5ds(h5f, group + 'temperature/electron', plasma.Te[1], np.float64, 'Electron temperature', 'electronvolt')


def rays(
        save_path: Path,
        rays: np.ndarray,
        ray_rates: np.ndarray,
        sweep_id: int,
        num_saved_rays: int = 1000
        ) -> None:

    with h5py.File(save_path, 'a') as h5f:

        group = f'/rays/step_{sweep_id:03d}/'
        rays_saved = rays[::max(len(rays) // num_saved_rays, 1), :, :]
        ray_rates_saved = ray_rates[::max(len(rays) // num_saved_rays, 1)]
        h5ds(h5f, group + 'rays', rays_saved, np.float16, 'Sample of particle positions (num_rays x max_steps x 3)', 'millimeter')
        h5ds(h5f, group + 'ray_rates', ray_rates_saved, np.float64, 'Particle rate per ray', 'microsecond^-1')


def iv_curve(
        save_path: Path,
        rpa: RPA
        ) -> None:

    with h5py.File(save_path, 'a') as h5f:

        group = f'/iv_curve/'
        h5ds(h5f, group + 'voltages', rpa.iv_curve[:, 0], np.float64, 'Sweeping bias voltages', 'volt')
        if rpa.is_ivm:
            h5ds(h5f, group + 'currents', rpa.iv_curve[:, 1:], np.float64, 'Anode currents', 'nanoampere')
        else:
            h5ds(h5f, group + 'currents', rpa.iv_curve[:, 1], np.float64, 'Anode currents', 'nanoampere')
 

def all_currents(
        save_path: Path,
        currents: np.ndarray
        ) -> None:

    with h5py.File(save_path, 'a') as h5f:

        group = f'/currents/'
        h5ds(h5f, group + 'aperture', currents[:, 0], np.float64, 'Aperture currents', 'nanoampere')
        h5ds(h5f, group + 'bias', currents[:, 0], np.float64, 'Bias currents', 'nanoampere')
        h5ds(h5f, group + 'anode', currents[:, 0], np.float64, 'Anode currents', 'nanoampere')


def measured_parameters(
        save_path: Path,
        pars: dict[str, tuple[float, float, float, float]],
        fit_info: dict[str, float | np.ndarray],
        ) -> None:
    
    with h5py.File(save_path, 'w') as h5f:
        group = '/parameters/'
        h5ds(h5f, group + 'legend', (0, 1, 2, 3), np.int16, '(Value, Uncertainty (2\u03c3), Absolute error, Relative error (%))', 'index')
        h5ds(h5f, group + 'density/fit', pars['N1'], np.float64, 'Fit density', 'millimeter^-3')
        h5ds(h5f, group + 'density/saturation', pars['N2'], np.float64, 'Saturation density', 'millimeter^-3')
        h5ds(h5f, group + 'velocity/x', pars['U1'], np.float64, 'Beam velocity, x component', 'millimeter microsecond^-1')
        h5ds(h5f, group + 'velocity/y', pars['U2'], np.float64, 'Beam velocity, y component', 'millimeter microsecond^-1')
        h5ds(h5f, group + 'velocity/z', pars['U3'], np.float64, 'Beam velocity, z component', 'millimeter microsecond^-1')
        h5ds(h5f, group + 'temperature', pars['T'], np.float64, 'Ion temperature', 'electronvolt')

        group = '/fit_information/'
        h5ds(h5f, group + 'r_squared', fit_info['r2'], np.float64, 'Coefficient of determination', 'n/a')
        h5ds(h5f, group + 'chi_squared/regular', fit_info['chi2'], np.float64, 'Chi-squared statistic', 'n/a')
        h5ds(h5f, group + 'chi_squared/reduced', fit_info['red_chi2'], np.float64, 'Reduced chi-squared statistic', 'n/a')
        h5ds(h5f, group + 'degrees_of_freedom', fit_info['dof'], np.int64, 'Sweep points - 3', 'n/a')
        h5ds(h5f, group + 'covariance', fit_info['cov'], np.float64, 'Covariance matrix', 'n/a')
        

def h5ds(
        h5f: h5py.File,
        name: str,
        data: float | tuple[float, ...] | np.ndarray | str,
        dtype: np.dtype | type,
        description: str,
        units: str
        ) -> None:
    
    if dtype is str:
        dtype = h5py.string_dtype(encoding='utf-8')

    ds = h5f.create_dataset(name, data=data, dtype=dtype)
    ds.attrs['description'] = description
    ds.attrs['units'] = units


def config_toml(
        base_path: Path,
        out_path: Path,
        plasma_changes: dict[str, float | list[float]]
        ) -> None:
    
    with open(base_path / 'config.toml') as f:
        lines = f.readlines()
    
    new_lines = []
    change_plasma = False
    for line in lines:

        line = line.rstrip('\n')
        if line == '[plasma]':
            change_plasma = True
        elif line == '[background_plasma]':
            change_plasma = False

        name = line.replace(' ', '').split('=')[0]
        comment = line.split('#')[-1]
        comment_start_id = len(line.split('#')[0])

        if name in plasma_changes.keys() and change_plasma:
            val = plasma_changes[name]
            if isinstance(val, list):
                val_str = ''
                for v in val:
                    val_str += f'{v:.3f}, '
                val_str = '[' + val_str.rstrip(', ') + ']'
            else:
                val_str = f'{val:.3f}'

            line = f'{name} = {val_str}'
            num_chars = len(line)
            line += ' ' * (comment_start_id - num_chars) + '#' + comment + ' (auto-generated)'

        new_lines.append(f'{line}\n')
    
    with open(out_path / 'config.toml', 'w') as f:
        f.writelines(new_lines)

def batch(
        base_path: Path,
        out_path: Path,
        densities: np.ndarray | list[float],
        beam_speeds: np.ndarray | list[float],
        beam_angles: np.ndarray | list[float], # degrees
        ion_temperatures: np.ndarray | list[float],
        target_num_tests: int = 192
        ) -> None:

    out_path.mkdir(exist_ok=True)

    num_tests = len(densities) * len(beam_speeds) * len(beam_angles) * len(ion_temperatures)
    if num_tests != target_num_tests:
        raise ValueError(f'Incorrect amount of tests: {num_tests} =/= {target_num_tests}')

    ind = 0
    for n in densities:
        for u in beam_speeds:
            for a in beam_angles:
                for t in ion_temperatures:
                    u2 = u * np.sin(np.deg2rad(a))
                    u3 = u * np.cos(np.deg2rad(a))
                    path = out_path / f'{ind:03d}_n={n:04d}_u={u:02d}_a={a:02d}_t={int(1e3*t):03d}'
                    path.mkdir(exist_ok=True)
                    config_toml(base_path, path, {'density': n, 'beam_velocity': [0.0, u2, u3], 'ion_temperature': t})
                    ind += 1
