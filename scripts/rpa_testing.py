from sailboat import SAILBOAT_ROOT
import sailboat.rpa as srpa


sim_direc = SAILBOAT_ROOT / 'data' / 'rpa' / 'test'

srpa.sim.run(sim_direc)

# rpa = srpa.read.rpa(sim_direc)
# plasma = srpa.read.plasma(sim_direc)

# rays, ray_rates = srpa.sim.rays(
#     num_rays = 4000,
#     max_steps = 1000,
#     rpa = rpa,
#     plasma = plasma,
#     dt = 0.01
# )

# srpa.plot.rays(rays, ray_rates, rpa, plasma)
# srpa.plot.rays_3d(rays, rpa)