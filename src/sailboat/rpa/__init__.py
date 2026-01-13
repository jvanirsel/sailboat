import numpy as np
from . import plot, read, sim

_MU = 0.0103651 # electronvolts microseconds^2 / millimeter^2
_Q_ELEM = 1.60218e-4 # femtocoulombs
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


class Screen:
    def __init__(
            self,
            location: float,
            voltages: float | Sweep
            ):
        
        sweeping = isinstance(voltages, Sweep)
        self.voltage = voltages[0] if sweeping else voltages # volts
        self.location = location # millimeters
        self.sweeping = sweeping
        if sweeping:
            self.sweep: Sweep = voltages
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
        
        self.sensor = sensor # millimeters
        self.aperture = aperture # millimeters


class RPA:
    def __init__(
            self,
            screens: list[Screen],
            geometry: RPAGeometry,
            ):

        if not screens:
            raise ValueError('screens list must not be empty')

        self.screens = sorted(screens, key=lambda s: s.location)
        self.sensor = geometry.sensor
        self.aperture = geometry.aperture
        self.depth = self.screens[-1].location

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
            temperature: float,
            density: float,
            ionization_state: float,
            charge: float,
            mass: float
            ):
        
        self.V = velocity
        self.T = temperature
        self.N = density
        self.Z = ionization_state
        self.Q = charge
        self.M = mass