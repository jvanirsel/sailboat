from gemini3d import read

def grid(sim_path: str) -> dict:
    return read.grid(sim_path, var=[
        'lx', 'x1', 'x2', 'x3',
        'x', 'y', 'z', 'glon', 'glat', 'alt',
        ])