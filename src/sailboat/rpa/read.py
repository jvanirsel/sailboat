from . import RPA, Sweep, Screen, RPAGeometry, Plasma, _Q_ELEM, _MU
import tomllib
from pathlib import Path
import numpy as np

def toml(
        rpa_direc: Path
    ) -> dict:
    
    toml_path = rpa_direc / 'config.toml'

    if not toml_path.is_file():
        raise FileNotFoundError(f'No config.toml found in {rpa_direc}')

    with open(toml_path, 'rb') as toml_file:
        return tomllib.load(toml_file)


def config(
        rpa_direc: Path
        ) -> dict:
    
    toml_dict = toml(rpa_direc)
    cfg = toml_dict['simulation']
    cfg['num_sweeps'] = int(toml_dict['rpa']['sweep_steps'])
    cfg['num_rays'] = int(cfg['num_rays'])
    cfg['max_steps'] = int(cfg['max_steps'])

    return cfg


def rpa(
        rpa_direc: Path
        ) -> RPA:
    
    cfg_rpa = toml(rpa_direc)['rpa']
    screen_locations = np.array(cfg_rpa['screen_locations']).astype(float) # millimeter
    screen_voltages =  np.array(cfg_rpa['screen_voltages']).astype(float) # volt
    num_screens = len(screen_locations)
    floating_potential = float(cfg_rpa['floating_potential'])
    screen_voltages += floating_potential
    sweep_id = int(cfg_rpa['sweep_id'])
    is_ivm = bool(cfg_rpa['is_ivm'])

    screens = []
    for sid in range(num_screens):
        if sid == sweep_id:
            v = Sweep(
                min_voltage = float(cfg_rpa['sweep_limits'][0]),
                max_voltage = float(cfg_rpa['sweep_limits'][1]),
                num_steps = int(cfg_rpa['sweep_steps']),
                rising = bool(cfg_rpa['sweep_rising'])
            )
        else:
            v = screen_voltages[sid]
        
        screens.append(
            Screen(
                location = screen_locations[sid],
                voltages = v
            )
        )

    return RPA(
        screens = screens,
        geometry = RPAGeometry(
            sensor = (
                float(cfg_rpa['sensor_size'][0]),
                float(cfg_rpa['sensor_size'][1])
            ), # millimeters
            aperture = (
                float(cfg_rpa['aperture_size'][0]),
                float(cfg_rpa['aperture_size'][1])
            ) # millimeters
            ),
        floating_potential = floating_potential,
        is_ivm = is_ivm
    )


def plasma(
        rpa_direc: Path
        ) -> Plasma:
    
    cfg_plasma = toml(rpa_direc)['plasma']
    cfg_plasma_bg = toml(rpa_direc)['background_plasma']

    velocity = (
        float(cfg_plasma['beam_velocity'][0]),
        float(cfg_plasma['beam_velocity'][1]),
        float(cfg_plasma['beam_velocity'][2])
        ) # millimeters / microsecond
    ion_temperature = (
        float(cfg_plasma['ion_temperature']),
        float(cfg_plasma_bg['ion_temperature'])
        ) # electronvolts
    electron_temperature = (
        float(cfg_plasma['electron_temperature']),
        float(cfg_plasma_bg['electron_temperature'])
        ) # electronvolts
    density = (
        float(cfg_plasma['density']),
        float(cfg_plasma_bg['density'])
        ) # millimeter^-3
    ionization_state = int(cfg_plasma['ionization_state']) # elementary charges
    charge = ionization_state * _Q_ELEM # femtocoulombs
    mass = float(cfg_plasma['ion_mass']) * _MU # electronvolts microseconds^2 / millimeter^2
    magnetic_field = (
        float(cfg_plasma['magnetic_field'][0]),
        float(cfg_plasma['magnetic_field'][1]),
        float(cfg_plasma['magnetic_field'][2])
        ) # microtesla

    return Plasma(
        velocity = velocity,
        ion_temperature = ion_temperature,
        electron_temperature = electron_temperature,
        density = density,
        ionization_state = ionization_state,
        charge = charge,
        mass = mass,
        magnetic_field = magnetic_field
    )

# def background_plasma(
#         rpa_direc: Path
#         ) -> Plasma:    

#     cfg_plasma = toml(rpa_direc)['plasma']
#     cfg_plasma_bg = toml(rpa_direc)['background_plasma']
#     velocity = (0.0, 0.0, 0.0)
#     ion_temperature = float(cfg_plasma_bg['ion_temperature']) # electronvolts
#     electron_temperature = float(cfg_plasma['electron_temperature']) # electronvolts
#     density = float(cfg_plasma_bg['density']) # millimeter^-3
#     ionization_state = int(cfg_plasma['ionization_state']) # elementary charges
#     charge = ionization_state * _Q_ELEM # femtocoulombs
#     mass = float(cfg_plasma['ion_mass']) * _MU # electronvolts microseconds^2 / millimeter^2
#     magnetic_field = (
#         float(cfg_plasma['magnetic_field'][0]),
#         float(cfg_plasma['magnetic_field'][1]),
#         float(cfg_plasma['magnetic_field'][2])
#         ) # microtesla

#     return Plasma(
#         velocity = velocity,
#         ion_temperature = ion_temperature,
#         electron_temperature = electron_temperature,
#         density = density,
#         ionization_state = ionization_state,
#         charge = charge,
#         mass = mass,
#         magnetic_field = magnetic_field
#     )