from . import RPA, Plasma
import h5py
from pathlib import Path
import numpy as np

def config_data(
        save_path: Path,
        cfg: dict,
        rpa: RPA,
        plasmas: Plasma | list[Plasma]
        ) -> None:

    if isinstance(plasmas, Plasma):
        plasmas = [plasmas]

    # when writing config data, overwrite file, otherwise append
    with h5py.File(save_path, 'w') as h5f:

        group = '/simulation/'
        h5ds(h5f, group + 'num_rays', cfg['num_rays'], np.int64, 'Number of simulated rays', 'n/a')
        h5ds(h5f, group + 'num_sweep_steps', cfg['num_sweeps'], np.int64, 'Number of sweep steps', 'n/a')
        h5ds(h5f, group + 'max_steps', cfg['max_steps'], np.int64, 'Maximum number of time steps per ray', 'n/a')
        h5ds(h5f, group + 'max_distance', cfg['dx_max'], np.float64, 'Maximum distance per time step', 'millimeter')
        h5ds(h5f, group + 'time_step_factor', cfg['dt_factor'], np.float64, 'time step x curvature x speed (dt x kappa x v)', 'n/a')

        group = '/rpa/geometry/'
        h5ds(h5f, group + 'source', rpa.source, np.float64, 'Plasma source dimensions (x, y)', 'millimeter')
        h5ds(h5f, group + 'aperture', rpa.aperture, np.float64, 'Aperture dimensions (x, y)', 'millimeter')
        h5ds(h5f, group + 'sensor', rpa.sensor, np.float64, 'Sensor dimensions (x, y)', 'millimeter')
        h5ds(h5f, group + 'depth', rpa.depth, np.float64, 'Distance from aperture to sensor', 'millimeter')

        group = '/rpa/screens/'
        h5ds(h5f, group + 'locations', rpa.get_locations(), np.float64, 'Screen distances from aperture', 'millimeter')
        h5ds(h5f, group + 'voltages', rpa.get_voltages(), np.float64, 'Screen voltages', 'volt')

        group = '/rpa/sweep/'
        h5ds(h5f, group + 'index', rpa.sweep_screen_id, np.int64, 'Index of sweeping screen', 'n/a')
        h5ds(h5f, group + 'length', rpa.sweep_len, np.int64, 'Number of sweep steps', 'n/a')
        h5ds(h5f, group + 'voltages', rpa.get_sweep_voltages(), np.float64, 'Sweep voltages', 'volt')

        i = 0
        for plasma in plasmas:
            group = '/plasmas/' + chr(65 + i) + '/'
            h5ds(h5f, group + 'density', plasma.N, np.float64, 'density', 'millimeter^-3')
            h5ds(h5f, group + 'temperature/ion', plasma.Ti, np.float64, 'Ion temperature', 'electronvolt')
            h5ds(h5f, group + 'temperature/electron', plasma.Te, np.float64, 'Electron temperature', 'electronvolt')
            h5ds(h5f, group + 'mass', plasma.M, np.float64, 'Ion temperature', 'electronvolt microsecond^2 / millimeter^2')
            h5ds(h5f, group + 'charge', plasma.Q, np.float64, 'Ion charge', 'femtocoulomb')
            h5ds(h5f, group + 'ionization_state', plasma.Z, np.int64, 'Ionization state of ions', 'elementary charge')
            h5ds(h5f, group + 'beam/velocity', plasma.V, np.float64, 'Plasma beam velocity (ux, uy, uz)', 'millimeter microsecond^-1')
            h5ds(h5f, group + 'beam/energy', plasma.K, np.float64, 'Plasma beam kinetic energy', 'electronvolt')
            h5ds(h5f, group + 'magnetic_field', plasma.B, np.float64, 'Background magnetic field (Bx, By, Bz)', 'microtesla')
            i += 1


def rays(
        save_path: Path,
        rays: np.ndarray,
        ray_rates: np.ndarray,
        sweep_id: int
        ) -> None:

    with h5py.File(save_path, 'a') as h5f:

        group = f'/rays/step_{sweep_id:03d}/'
        h5ds(h5f, group + 'rays', rays, np.float16, 'Particle positions (num_rays x max_steps x 3)', 'millimeter')
        h5ds(h5f, group + 'ray_rates', ray_rates, np.float64, 'Particle rate per ray', 'microsecond^-1')


def iv_curve(
        save_path: Path,
        iv_curve: np.ndarray
        ) -> None:

    with h5py.File(save_path, 'a') as h5f:

        group = f'/iv_curve/'
        h5ds(h5f, group + 'voltages', iv_curve[:, 0], np.float64, 'Sweeping bias voltages', 'volt')
        h5ds(h5f, group + 'currents', iv_curve[:, 1], np.float64, 'Anode currents', 'nanoamperes')


def h5ds(
        h5f: h5py.File,
        name: str,
        data: float | tuple[float, ...] | np.ndarray,
        dtype: np.dtype | type,
        description: str,
        units: str
        ):

        ds = h5f.create_dataset(name, data=data, dtype=dtype)
        ds.attrs['description'] = description
        ds.attrs['units'] = units