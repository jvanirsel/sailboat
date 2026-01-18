from . import RPA, Plasma, Screen
from .. import utils
import numpy as np
from pathlib import Path
import time

def run(
        rpa_direc: Path,
        save: bool = True,
        debug: bool = False
        ) -> None:

    from . import read, plot, sim, write

    if not rpa_direc.is_dir():
        raise FileNotFoundError(f'RPA directory {rpa_direc} not found')
    
    cfg = read.config(rpa_direc)
    rpa = read.rpa(rpa_direc)
    plasma = read.plasma(rpa_direc)
    plasma_bg = read.background_plasma(rpa_direc)

    cfg_id = cfg['config_id']
    plot_direc = rpa_direc / 'plots'
    h5_path = rpa_direc / f'config_{cfg_id:02d}_data.h5'
    num_sweep_ids = cfg['num_sweeps']

    if h5_path.is_file():
        while True:
            overwrite = input(f'Overwrite {h5_path} (Y/n)?: ')
            if overwrite == 'Y':
                save = True
                break
            if overwrite == 'n':
                save = False
                print('\n' + 80 * '!')
                print(' NOT SAVING DATA '.center(80, '!'))
                print(80 * '!' + '\n')
                break

    if save:
        write.config_data(h5_path, cfg, rpa, [plasma, plasma_bg])

    for sweep_id in range(num_sweep_ids):
        print(f' Sweep id: {sweep_id + 1} / {num_sweep_ids} '.center(80, '-'))

        t0 = time.perf_counter()
        rays, ray_rates = sim.rays(
            num_rays = cfg['num_rays'],
            max_steps = cfg['max_steps'],
            rpa = rpa,
            plasmas = [plasma, plasma_bg],
            dt_factor = cfg['dt_factor'],
            dx_max = cfg['dx_max'],
            debug = debug
        )
        t1 = time.perf_counter()
        print(f'Ray calculation time: {t1-t0:.2f} seconds')

        if save:
            write.rays(h5_path, rays, ray_rates, sweep_id)

        currents = plasma.Q * get_currents(rays, ray_rates, [0.0, rpa.screens[rpa.sweep_screen_id].location, rpa.depth])
        rpa.update_iv_curve(currents[-1])

        png_path = plot_direc / f'config_{cfg_id:02d}_step_{sweep_id:03d}.png'
        print('Plotting rays...', end='\r')
        plot.rays(png_path, rays, ray_rates, currents, rpa, [plasma, plasma_bg])
        print('Plotting rays... done')

        rpa.step_sweep()
    if save:
        write.iv_curve(h5_path, rpa.iv_curve)
    
    gif_path = rpa_direc / f'config_{cfg_id:02d}.gif'
    utils.make_gif(plot_direc, prefix=f'config_{cfg_id:02d}_step', filename=gif_path)


def rays(
        num_rays: int,
        max_steps: int,
        rpa: RPA,
        plasmas: Plasma | list[Plasma],
        dt_factor: float,
        dx_max: float,
        debug: bool = False,
        ) -> tuple[np.ndarray, np.ndarray]:
    
    '''
    Generate a collection of plasma ion ray traces, each evolving through RPA defined forces.
    Rays will terminate once outside of a predetermined volume.
    Also generates the particle rates of each ray for later current calculations.
    
    :param num_rays: Number of rays to simulate
    :type num_rays: int
    :param max_steps: Maximum number of time steps per ray
    :type max_steps: int
    :param rpa: Retarding Potential Analyzer
    :type rpa: sailboat.rpa.RPA
    :param plasma: Plasma parameters
    :type plasma: sailboat.rpa.Plasma or list[sailboat.rpa.Plasma]
    :param dt: Time step [microsecond]
    :type dt: float
    :return: Collection of rays (num_rays, num_steps, 3) and their particle rates (num_rays,) [particle microsecond^-1]
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    '''

    if not isinstance(plasmas, list):
        plasmas = [plasmas]

    x_range_sim = (-rpa.sensor[0], rpa.sensor[0])
    y_range_sim = (-rpa.sensor[1], rpa.sensor[1])
    z_range_sim = (-rpa.depth, rpa.depth + dx_max)
    x_range_sensor = (-rpa.sensor[0] / 2, rpa.sensor[0] / 2)
    y_range_sensor = (-rpa.sensor[1] / 2, rpa.sensor[1] / 2)
    x_range_aperture = (-rpa.aperture[0] / 2, rpa.aperture[0] / 2)
    y_range_aperture = (-rpa.aperture[1] / 2, rpa.aperture[1] / 2)
    x_range_source = (-rpa.source[0] / 2, rpa.source[0] / 2)
    y_range_source = (-rpa.source[1] / 2, rpa.source[1] / 2)

    N_tot = np.sum([p.N for p in plasmas])
    Ne = N_tot
    linear_aperture_particle_density = N_tot * rpa.source[0] * rpa.source[1] / num_rays # particles / millimeter
    debye_length = 7.43e6 * (Ne / plasmas[0].Te + np.sum([p.N / p.Ti for p in plasmas]))**(-0.5) # millimeter
    print(debye_length)

    print(f'Allocating {(num_rays * max_steps * 3 * 8)/(1024**3):.2f} GB ray matrix...', end='\r')
    rays = np.full((num_rays, max_steps, 3), np.nan, dtype=float)
    ray_rates = np.full(num_rays, np.nan)
    print(f'Allocating {(num_rays * max_steps * 3 * 8)/(1024**3):.2f} GB ray matrix... done')
        
    early_terminations = 0
    sid_avg = 0
    for rid in range(num_rays):
        if rid % (num_rays // 80) == 0 or rid == num_rays - 1:
            utils.load_bar(rid, num_rays-1, 'Generating rays')
        x, y, z, vx, vy, vz = initial_phase(x_range_source, y_range_source, z_range_sim[0], plasmas)
        ray_rates[rid] = linear_aperture_particle_density * (vx**2 + vy**2 + vz**2)**0.5 # particles / microsecond
        
        sid = 0
        for sid in range(max_steps):
            rays[rid, sid, :] = [x, y, z]
            phase = np.array([x, y, z, vx, vy, vz])
            x, y, z, vx, vy, vz = update_phase(phase, rpa.screens, plasmas[0], debye_length, dt_factor, dx_max, debug=debug)

            # out of simulation z range
            if z < z_range_sim[0] or z > z_range_sim[1]:
                break

            # left of aperture
            if z < 0:
                # in the plane of aperture
                if z > -2 * dx_max:
                    # does not pass through aperture, hit aperture shield
                    if x < x_range_aperture[0] or x > x_range_aperture[1]:
                        break
                    if y < y_range_aperture[0] or y > y_range_aperture[1]:
                        break
                # out of simulation x or y range
                if x < x_range_sim[0] or x > x_range_sim[1]:
                    break
                if y < y_range_sim[0] or y > y_range_sim[1]:
                    break
            # right of aperture
            else:
                # hit sensor walls
                if x < x_range_sensor[0] or x > x_range_sensor[1]:
                    break
                if y < y_range_sensor[0] or y > y_range_sensor[1]:
                    break

        sid_avg += sid
        if sid == max_steps - 1:
            early_terminations += 1
    
    print()
    print(f'Average number of steps: {sid_avg / num_rays:.2f}')
    print(f'Early terminations: {100 * early_terminations / num_rays:.2f} %')
    return rays, ray_rates


def initial_phase(
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        z0: float,
        plasma: Plasma | list[Plasma]
    ) -> np.ndarray:

    '''
    Generate initial phase of a sampled ray given RPA geometry and plasma parameters.

    :param rpa: Retarding Potential Analyzer
    :type rpa: sailboat.rpa.RPA
    :param plasma: Plasma parameters
    :type plasma: sailboat.rpa.Plasma
    :return: Initial phase space vector (6,) [3 x millimeter, 3 x millimeter microsecond^-1]
    :rtype: numpy.ndarray
    '''

    # randomly choose plasma to source from with density as weights
    if isinstance(plasma, list):
        weights = np.array([p.N for p in plasma], dtype=float)
        weights /= weights.sum()
        plasma_id = np.random.choice(range(len(plasma)), p=weights)
        plasma = plasma[plasma_id]

    assert(isinstance(plasma, Plasma))

    sigma = np.sqrt(plasma.Ti / plasma.M) # millimeters / microsecond

    x0 = np.random.uniform(-x_range[0], x_range[0])
    y0 = np.random.uniform(-y_range[0], y_range[0])

    vx0 = np.random.normal(loc=plasma.V[0], scale=sigma)
    vy0 = np.random.normal(loc=plasma.V[1], scale=sigma)
    vz0 = np.random.normal(loc=plasma.V[2], scale=sigma)

    return np.array([x0, y0, z0, vx0, vy0, vz0])


def update_phase(
        phase: np.ndarray,
        screens: list[Screen],
        plasma: Plasma,
        debye_length: float,
        dt_factor: float,
        dx_max: float,
        debug: bool = False
    ) -> np.ndarray:
    
    '''
    Update phase space vector using Velocity Verlet integration.
    
    :param phase: Input phase space vector (6,) [3 x millimeter, 3 x millimeter microsecond^-1]
    :type phase: numpy.ndarray
    :param dt: Time step size [microsecond]
    :type dt: float
    :param screens: RPA screens
    :type screens: list[sailboat.rpa.Screen]
    :param plasma: Plasma parameters
    :type plasma: srpa.Plasma
    :return: Updated phase space vector (6,) [3 x millimeter, 3 x millimeter microsecond^-1]
    :rtype: numpy.ndarray
    '''

    x, y, z, vx, vy, vz = phase

    # calculate acceleration and update position
    ax, ay, az = acceleration(phase, screens, plasma, debye_length)

    dx_max = 0.2
    speed = (vx**2 + vy**2 + vz**2)**0.5 # ||v||
    kappa = ((vy * az - vz * ay)**2 + (vz * ax - vx * az)**2 + (vx * ay - vy * ax))**0.5 # ||v x a||
    dt = dx_max / speed
    r = 0
    if kappa > 1e-8:
        dt = min(dt, dt_factor * speed**2 / kappa) # ||v||^2 / ||v x a|| = 1 / ( ||v|| x kappa ), microseconds)
        r = speed**3 / kappa # radius of curvature
    
    if debug:
        print(f'{z:.3f},{r:.3f},{speed:.3f},{dt:.3f},{dt * speed:.3f}')

    x += vx * dt + 0.5 * ax * dt**2
    y += vy * dt + 0.5 * ay * dt**2
    z += vz * dt + 0.5 * az * dt**2

    # calculate acceleration at new position and update velocity
    phase = np.array([x, y, z, vx, vy, vz])
    ax, ay, az = acceleration(phase, screens, plasma, debye_length)
    vx += ax * dt
    vy += ay * dt
    vz += az * dt

    return np.array([x, y, z, vx, vy, vz])


def acceleration(
        phase: np.ndarray,
        screens: list[Screen],
        plasma: Plasma,
        debye_length: float
    ) -> tuple[float, float, float]:

    '''
    Phase space dependent acceleration.
    
    :param phase: Input phase space vector (6,) [3 x millimeter, 3 x millimeter microsecond^-1]
    :type phase: numpy.ndarray
    :param screens: RPA screens
    :type screens: list[sailboat.rpa.Screen]
    :param plasma: Plasma parameters
    :type plasma: sailboat.rpa.Plasma
    :return: Acceleration vector (3,) [millimeter microsecond^-2]
    :rtype: tuple[float, float, float]
    '''

    x, y, z, vx, vy, vz = phase
    Bx, By, Bz = plasma.B # tesla or volt microsecond millimeter^-2
    specific_charge = plasma.Z / plasma.M # volt^-1 microsecond^-2 millimeter^2

    sid = 0
    while sid < len(screens) and z >= screens[sid].location:
        sid += 1
    
    if sid == len(screens):
        return 0, 0, 0
    
    if sid == 0:
        ### ADD SHEATH ACCELERATION FROM SPACECRAFT POTENTIAL
        return 0, 0, 0
    
    s0 = screens[sid - 1]
    s1 = screens[sid]

    # electric field
    Ex = 0
    Ey = 0
    Ez = (s0.voltage - s1.voltage) / (s1.location - s0.location)

    # Lorentz force
    ax = specific_charge * (Ex + vy * Bz - vz * By)
    ay = specific_charge * (Ey + vz * Bx - vx * Bz)
    az = specific_charge * (Ez + vx * By - vy * Bx)

    ay += -9.8e-9 # millimeter / microseconds^2 heheh

    return ax, ay, az # millimeter / microseconds^2


def collect_punctures(
        rays: np.ndarray,
        surface: float,
        get_coords: bool = False
        ) -> np.ndarray:

    '''
    Collect ray indices that puncture a given surface.
    A puncture is defined as a ray whose last recorded z-position is passed surface.
    This means an odd (even) number of punctures is counted as one (zero).
    
    :param rays: Collection of rays (num_rays, num_steps, 3)
    :type rays: numpy.ndarray
    :param surface: z-position of surface to check for punctures [millimeter]
    :type surface: float
    :param get_coords: Whether to return puncture x- and y-coordinates or just indices
    :type get_coords: bool
    :return: Indices (:,), or puncture x- and y-coordinates and indeces (:, 3) [millimeter, millimeter, index]
    :rtype: numpy.ndarray
    '''

    if get_coords:
        punctures = np.full((len(rays), 3), np.nan)
    else:
        punctures = np.full(len(rays), np.nan)

    for i in range(len(rays)):
        ray = rays[i] # millimeter
        ray = ray[~np.isnan(ray[:, 2]), :]
        if ray[-1, 2] > surface:
            if get_coords:
                punctures[i, :2] = ray[-1, :2] # millimeter
                punctures[i, 2] = float(i) # index
            else:
                punctures[i] = i # index

    if get_coords:
        return punctures[~np.isnan(punctures[:, 0]), :].astype(float)
    else:
        return punctures[~np.isnan(punctures)].astype(int)


def get_currents(
        rays: np.ndarray,
        ray_rates: np.ndarray,
        surfaces: float | list[float]
        ) -> np.ndarray:
    
    '''
    Calculate currents at given surfaces based on ray punctures and ray rates.
    If surface is 0.0, current is calculated as total ray rate.

    :param rays: Collection of rays (num_rays, num_steps, 3)
    :type rays: numpy.ndarray
    :param ray_rates: Particle rates at each ray (num_rays,) [particle microsecond^-1]
    :type ray_rates: numpy.ndarray
    :param surfaces: List of surface z-positions at which to collect currents [millimeter]
    :type surfaces: float or list[float]
    :return: Currents at each surface [particle microsecond^-1]
    :rtype: numpy.ndarray or float
    '''

    if isinstance(surfaces, float):
        float_in = True
        surfaces = [surfaces]
    else:
        float_in = False
    assert(isinstance(surfaces, list))

    currents = np.full(len(surfaces), np.nan)
    for i, surface in enumerate(surfaces):
        # if surface == -:
        #     currents[i] = np.sum(ray_rates) # particle microsecond^-1
        # else:
        punctures = collect_punctures(rays, surface) # indeces
        currents[i] = np.sum(ray_rates[punctures]) # particle microsecond^-1
    
    return currents[0] if float_in else currents


if __name__ == '__main__':
    import sys
    run(Path(sys.argv[1]))