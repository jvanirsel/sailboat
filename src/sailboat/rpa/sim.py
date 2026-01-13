from . import RPA, Plasma, Screen
from .. import utils
import numpy as np
from pathlib import Path


def run(
        rpa_direc: Path,
        ) -> None:

    from . import read, plot, sim

    if not rpa_direc.is_dir():
        raise FileNotFoundError(f'RPA directory {rpa_direc} not found')
    
    cfg = read.config(rpa_direc)
    rpa = read.rpa(rpa_direc)
    plasma = read.plasma(rpa_direc)

    plot_direc = rpa_direc / 'plots'
    gif_filename = 'simulation.gif'

    for sweep_id in range(cfg['num_sweeps']):

        rays, ray_rates = sim.rays(
            num_rays = cfg['num_rays'],
            max_steps = cfg['max_steps'],
            rpa = rpa,
            plasma = plasma,
            dt = 0.01
        )

        current = plasma.Q * get_currents(rays, ray_rates, rpa.depth)
        assert(isinstance(current, float))
        rpa.update_iv_curve(current)

        filename = f'step_{sweep_id:03d}.png'
        
        plot.rays(rays, ray_rates, rpa, plasma, plot_direc=plot_direc, filename=filename)
        rpa.step_sweep()

    utils.make_gif(plot_direc, prefix='step', filename=gif_filename)


def rays(
        num_rays: int,
        max_steps: int,
        rpa: RPA,
        plasma: Plasma,
        dt: float = 0.01
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
    :type plasma: sailboat.rpa.Plasma
    :param dt: Time step [microsecond]
    :type dt: float
    :return: Collection of rays (num_rays, num_steps, 3) and their particle rates (num_rays,) [particle microsecond^-1]
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    '''

    x_range = [-rpa.sensor[0] / 2, rpa.sensor[0] / 2]
    y_range = [-rpa.sensor[1] / 2, rpa.sensor[1] / 2]
    z_range = [-rpa.depth, rpa.depth]

    linear_aperture_particle_density =  plasma.N * rpa.aperture[0] * rpa.aperture[1] / num_rays # particles / millimeter

    rays = np.full((num_rays, max_steps, 3), np.nan)
    ray_rates = np.full(num_rays, np.nan)
    for rid in range(num_rays):
        x, y, z, vx, vy, vz = initial_phase(rpa, plasma)
        ray_rates[rid] = linear_aperture_particle_density * vz # particles / microsecond
        for sid in range(max_steps):
            rays[rid, sid, :] = [x, y, z]
            x, y, z, vx, vy, vz = update_phase(np.array([x, y, z, vx, vy, vz]), dt, rpa.screens, plasma)
            if x < x_range[0] or x > x_range[1] or y < y_range[0] or y > y_range[1] or z < z_range[0] or z > z_range[1]:
                    break
    
    return rays, ray_rates


def initial_phase(
        rpa: RPA,
        plasma: Plasma
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

    x_range = rpa.aperture[0]
    y_range = rpa.aperture[1]
    sigma = np.sqrt(plasma.Ti / plasma.M) # millimeters / microsecond

    x0 = np.random.uniform(-x_range / 2, x_range / 2)
    y0 = np.random.uniform(-y_range / 2, y_range / 2)
    z0 = 0.0

    vx0 = np.random.normal(loc=plasma.V[0], scale=sigma)
    vy0 = np.random.normal(loc=plasma.V[1], scale=sigma)
    vz0 = np.random.normal(loc=plasma.V[2], scale=sigma)

    return np.array([x0, y0, z0, vx0, vy0, vz0])


def update_phase(
        phase: np.ndarray,
        dt: float,
        screens: list[Screen],
        plasma: Plasma
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
    ax, ay, az = acceleration(phase, screens, plasma)
    x += vx * dt + 0.5 * ax * dt**2
    y += vy * dt + 0.5 * ay * dt**2
    z += vz * dt + 0.5 * az * dt**2

    # calculate acceleration at new position and update velocity
    phase = np.array([x, y, z, vx, vy, vz])
    ax, ay, az = acceleration(phase, screens, plasma)
    vx += ax * dt
    vy += ay * dt
    vz += az * dt

    return np.array([x, y, z, vx, vy, vz])


def acceleration(
        phase: np.ndarray,
        screens: list[Screen],
        plasma: Plasma
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
    specific_charge = plasma.Z / plasma.M

    sid = 0
    while sid < len(screens) and z >= screens[sid].location:
        sid += 1
    
    if sid == 0 or sid == len(screens):
        return 0, 0, 0
    ### ADD SHEATH ACCELERATION FROM SPACECRAFT POTENTIAL
    
    s0 = screens[sid - 1]
    s1 = screens[sid]

    # electric field
    ax = 0
    ay = -9.8e-9 # millimeter / microseconds^2 heheh
    az = specific_charge * (s0.voltage - s1.voltage) / (s1.location - s0.location)

    # magnetic field
    ax += specific_charge * (Bz * vy - By * vz)
    ay += specific_charge * (Bx * vz - Bz * vx)
    az += specific_charge * (By * vx - Bx * vy)

    return ax, ay, az # millimeter / microseconds^2


def collect_punctures(
        rays: np.ndarray,
        surface: float,
        tolerance: float = 0.05,
        get_coords: bool = False
        ) -> np.ndarray:

    '''
    Collect ray indices that puncture a given surface.
    A puncture is defined as a ray whose last recorded z-position is passed surface minus tolerance.
    
    :param rays: Collection of rays (num_rays, num_steps, 3)
    :type rays: numpy.ndarray
    :param surface: z-position of surface to check for punctures [millimeter]
    :type surface: float
    :param tolerance: Tolerance for puncture detection [millimeter]
    :type tolerance: float
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
        if ray[-1, 2] > surface - tolerance:
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
        if surface == 0.0:
            currents[i] = np.sum(ray_rates) # particle microsecond^-1
        else:
            punctures = collect_punctures(rays, surface) # indeces
            currents[i] = np.sum(ray_rates[punctures]) # particle microsecond^-1
    
    return currents[0] if float_in else currents


if __name__ == '__main__':
    import sys
    run(Path(sys.argv[1]))