from os import getenv
from pathlib import Path

__version__ = '0.1.0'

_WGS84_A: float = 6_378_137.0 # semi-major axis (meters)
_WGS84_F: float = 1.0 / 298.257_223_563 # flattening
_WGS84_E2: float = _WGS84_F * (2.0 - _WGS84_F) # eccentricity^2
_RE: float = _WGS84_A

HOME = Path(getenv('HOME'))
SAILBOAT_ROOT = Path(getenv('SAILBOAT_ROOT'))
GEMINI_ROOT = Path(getenv('GEMINI_ROOT'))
GEMINI_SIM_ROOT = Path(getenv('GEMINI_SIM_ROOT'))

if not SAILBOAT_ROOT:
    raise ValueError('Environment variables not found: SAILBOAT_ROOT. ' \
    'Please set SAILBOAT_ROOT to location of\n\n' \
    '  https://github.com/jvanirsel/sailboat/tree/main/src/sailboat\n')
if not GEMINI_ROOT:
    raise ValueError('Environment variables not found: GEMINI_ROOT. ' \
    'Please set GEMINI_ROOT to location of\n\n  gemini3d/build/gemini.bin\n')
if not GEMINI_SIM_ROOT:
    raise ValueError('Environment variables not found: GEMINI_SIM_ROOT. ' \
    'Please set GEMINI_SIM_ROOT to your location of gemini simulations')
