import numpy as np

_MU = 0.0103651 # electronvolt microsecond^2 millimeter^-2
_ME = 5.68563e-6 # electronvolt microsecond^2 millimeter^-2
_Q_ELEM = 1.60218e-4 # femtocoulomb
_EPS0 = 5.52635e5 # elementary charge^2 electronvolt^-1 millimeter^-1
__version__ = '0.1.0'

class Sweep:
    def __init__(
            self,
            min_voltage: float,
            max_voltage: float,
            num_steps: int,
            rising: bool = True
            ):
        
        if num_steps < 2:
            raise ValueError('num_steps must be >= 2')
        
        self.range = max_voltage - min_voltage
        voltages = np.linspace(min_voltage, max_voltage, num_steps)
        self.voltages = voltages if rising else voltages[::-1]
    
    def __len__(
            self
            ) -> int:
        
        return len(self.voltages)
    
    def __getitem__(
            self,
            index: int
            ) -> float:
        return self.voltages[index]
    
    def __min__(
            self
            ):
        return min(self.voltages)
    
    def __max__(
            self
            ):
        return max(self.voltages)


class Screen:
    def __init__(
            self,
            location: float,
            voltages: float | Sweep
            ):

        sweeping = isinstance(voltages, Sweep)
        voltage = voltages[0] if sweeping else voltages # volts
        self.voltage = voltage
        self.location = location # millimeters
        self.sweeping = sweeping
        if sweeping:
            self.sweep: Sweep = voltages
            self.sweep_len = len(voltages)
            self.sweep_id = 0

    def step_sweep(
            self
            ) -> None:
        
        if not self.sweeping:
            raise TypeError('Attempting to sweep a non-sweeping screen')
        
        sweep = self.sweep
        self.sweep_id += 1
        self.voltage = sweep[self.sweep_id % len(sweep)]


class RPAGeometry:
    def __init__(
            self,
            sensor: tuple[float, float],
            aperture: tuple[float, float]
            ):
        
        self.sensor = sensor
        self.aperture = aperture
        self.source = (sensor[0] + 2 * aperture[0], sensor[1] + 2 * aperture[1])


class RPA:
    def __init__(
            self,
            screens: list[Screen],
            geometry: RPAGeometry,
            floating_potential: float,
            is_ivm: bool = True
            ):

        if not screens:
            raise ValueError('screens list must not be empty')

        self.screens = sorted(screens, key=lambda s: s.location)
        self.sensor = geometry.sensor
        self.aperture = geometry.aperture
        self.source = geometry.source
        self.depth = self.screens[-1].location
        sweep_len = 0
        sweep_screen_id = -1
        for sid, screen in enumerate(self.screens):
            if screen.sweeping:
                sweep_len = screen.sweep_len
                sweep_screen_id = sid
                break
        self.sweep_screen_id = sweep_screen_id
        self.sweep_len = sweep_len
        self.sweep_id = 0
        if is_ivm:
            self.iv_curve = np.full((sweep_len, 5), np.nan)
        else:
            self.iv_curve = np.full((sweep_len, 2), np.nan)
        self.floating_potential = floating_potential
        self.is_ivm = is_ivm

    def __len__(
            self
            ) -> int:
        
        return len(self.screens)

    def step_sweep(
            self
            ) -> None:
        
        for screen in self.screens:
            if screen.sweeping:
                screen.step_sweep()
                self.sweep_id = screen.sweep_id

    def get_sweep_voltages(
            self
            ) -> np.ndarray:
        
        for screen in self.screens:
            if screen.sweeping:
                return screen.sweep.voltages
        return np.array([])     
    
    def update_iv_curve(
            self,
            current: float | np.ndarray
            ) -> None:

        v = self.screens[self.sweep_screen_id].voltage
        if isinstance(current, float):
            self.iv_curve[self.sweep_id, :] = [v, current]
        else:
            self.iv_curve[self.sweep_id, :] = np.concatenate(([v], current))


    def get_locations(
            self
            ) -> np.ndarray:
        
        return np.array([s.location for s in self.screens])

    def get_voltages(
            self
            ) -> np.ndarray:
        
        return np.array([s.voltage for s in self.screens])


class Plasma:
    def __init__(
            self,
            velocity: tuple[float, float, float],
            ion_temperature: tuple[float, float],
            electron_temperature: tuple[float, float],
            density: tuple[float, float],
            ionization_state: float,
            charge: float,
            mass: float,
            magnetic_field: tuple[float, float, float]
            ):
        
        lambdaD = (
            234.957 * (electron_temperature[0] / density[0])**0.5,
            234.957 * (electron_temperature[1] / density[1])**0.5,
            234.957 * (ion_temperature[0] / density[0])**0.5,
            234.957 * (ion_temperature[1] / density[1])**0.5,
        )

        self.V = velocity
        self.Ti = ion_temperature
        self.Te = electron_temperature
        self.N = density
        self.Z = ionization_state
        self.Q = charge
        self.Mi = mass
        self.Me = _ME
        self.jz = charge * density[0] * velocity[2] # nanoamperes millimeter^-2
        self.lambdaD = (lambdaD[0]**-2 + lambdaD[1]**-2 + lambdaD[2]**-2 + lambdaD[3]**-2)**-0.5
        self.K = mass * (velocity[0]**2 + velocity[1]**2 + velocity[2]**2) / 2
        self.B = magnetic_field


from . import sim, plot, read