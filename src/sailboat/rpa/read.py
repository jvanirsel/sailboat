from . import RPA, Sweep, Screen, RPAGeometry, Plasma, _Q_ELEM, _MU
import tomllib
from pathlib import Path


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
    cfg['num_sweeps'] = toml_dict['rpa']['sweep_steps']

    return cfg


def rpa(
        rpa_direc: Path
        ) -> RPA:
    
    cfg_rpa = toml(rpa_direc)['rpa']
    screen_locations = cfg_rpa['screen_locations'] # millimeters
    screen_voltages = cfg_rpa['screen_voltages'] # volts
    num_screens = len(screen_locations)

    screens = []
    for sid in range(num_screens):
        if sid == cfg_rpa['sweep_id']:
            v = Sweep(
                min_voltage = cfg_rpa['sweep_limits'][0],
                max_voltage = cfg_rpa['sweep_limits'][1],
                num_steps = cfg_rpa['sweep_steps'],
                rising = cfg_rpa['sweep_rising']
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
            sensor = (cfg_rpa['sensor_size'][0], cfg_rpa['sensor_size'][1]), # millimeters
            aperture = (cfg_rpa['aperture_size'][0], cfg_rpa['aperture_size'][1]) # millimeters
            )
    )


def plasma(
        rpa_direc: Path
        ) -> Plasma:
    
    cfg_plasma = toml(rpa_direc)['plasma']
    velocity = (cfg_plasma['beam_velocity'][0], cfg_plasma['beam_velocity'][1], cfg_plasma['beam_velocity'][2]) # millimeters / microsecond
    ion_temperature = cfg_plasma['ion_temperature'] # electronvolts
    electron_temperature = cfg_plasma['electron_temperature'] # electronvolts
    density = cfg_plasma['density'] # millimeter^-3
    ionization_state = cfg_plasma['z'] # elementary charges
    charge = ionization_state * _Q_ELEM # femtocoulombs
    mass = cfg_plasma['m'] * _MU # electronvolts microseconds^2 / millimeter^2
    magnetic_field = cfg_plasma['magnetic_field'] # microtesla

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

