from os import getenv

__version__ = '0.1.0'

RE = 6370e3
GEMINI_SIM_ROOT = getenv('GEMINI_SIM_ROOT')
if GEMINI_SIM_ROOT is None:
    raise KeyError('Environment variable GEMINI_SIM_ROOT not set')
