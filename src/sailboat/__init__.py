from os import getenv

__version__ = '0.1.0'

RE = 6370e3
HOME = getenv('HOME')
GEMINI_ROOT = getenv('GEMINI_ROOT')
GEMINI_SIM_ROOT = getenv('GEMINI_SIM_ROOT')

if not GEMINI_ROOT:
    raise ValueError('Environment variables not found: GEMINI_ROOT. ' \
    'Please set GEMINI_ROOT to location of build/gemini.bin')
if not GEMINI_SIM_ROOT:
    raise ValueError('Environment variables not found: GEMINI_SIM_ROOT. ' \
    'Please set GEMINI_SIM_ROOT to location of gemini simulations')
