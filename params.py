# Parameters file
import numpy as np

class System:
   
    # Physical constants
    boltz=1.38064852e-23  # J/k
   
    # System parameters
    box_size = 1000    # mu_m
    cell_radius = 1.0  # mu_m
    temperature = 300  # K
    viscosity = 0.01  # g/cm s (1 g/cm s = 1 Poise = 0.1 kg/m s)
    
    # Diffusion constants (Stokes-Einstein relation)
    diffusion_constant = (boltz*temperature)/(6.0*np.pi*0.1*viscosity*1e-6*cell_radius)  # m^2/s
    diffusion_constant *= 1e12   # mu_m^2/s
    rot_diffusion_constant = (boltz*temperature)/(8.0*np.pi*0.1*viscosity*(1e-6*cell_radius)**3)  # rad^2/s

    # Time
    max_time = 34     # s
    time_step = 0.02  # s
    total_steps = int(max_time / time_step)
    timesteps = np.linspace(0, max_time, num=total_steps+1, endpoint=True)

    # Random number seed
    seed = 98
    np.random.seed(seed)

    # Graph header
    title = "Time = {}s, step size = {}s, seed = {}".format(max_time, time_step, seed)

