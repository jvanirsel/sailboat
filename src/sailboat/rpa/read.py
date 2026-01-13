import tomllib
from pathlib import Path
import sailboat.rpa as srpa


def toml(
        sim_direc: Path
    ) -> dict:
    
    toml_path = sim_direc / 'config.toml'

    if not toml_path.is_file():
        raise FileNotFoundError(f'No config.toml found in {sim_direc}')

    with open(toml_path, 'rb') as toml_file:
        return tomllib.load(toml_file)


def rpa(
        sim_direc: Path
        ) -> srpa.RPA:
    
    cfg = toml(sim_direc)['rpa']
    screen_locations = cfg['screen_locations'] # millimeters
    screen_voltages = cfg['screen_voltages'] # volts
    num_screens = len(screen_locations)

    screens = []
    for sid in range(num_screens):
        if sid == cfg['sweep_id']:
            v = srpa.Sweep(
                min_voltage = cfg['sweep_limits'][0],
                max_voltage = cfg['sweep_limits'][1],
                num_steps = cfg['sweep_steps'],
                rising = cfg['sweep_rising']
            )
        else:
            v = screen_voltages[sid]
        
        screens.append(
            srpa.Screen(
                location = screen_locations[sid],
                voltages = v
            )
        )

    return srpa.RPA(
        screens = screens,
        geometry = srpa.RPAGeometry(
            sensor = (cfg['sensor_size'][0], cfg['sensor_size'][1]), # millimeters
            aperture = (cfg['aperture_size'][0], cfg['aperture_size'][1]) # millimeters
            )
    )


def plasma(
        sim_direc: Path
        ) -> srpa.Plasma:
    
    cfg = toml(sim_direc)['plasma']
    velocity = (cfg['beam_velocity'][0], cfg['beam_velocity'][1], cfg['beam_velocity'][2]) # millimeters / microsecond
    temperature = cfg['temperature'] # electronvolts
    density = cfg['density'] # millimeter^-3
    ionization_state = cfg['z'] # elementary charges
    charge = ionization_state * srpa._Q_ELEM # femtocoulombs
    mass = cfg['m'] * srpa._MU # electronvolts microseconds^2 / millimeter^2

    return srpa.Plasma(
        velocity = velocity,
        temperature = temperature,
        density = density,
        ionization_state = ionization_state,
        charge = charge,
        mass = mass
    )

