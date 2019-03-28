# Parameters file
import numpy as np

class System:
   
    # Physical constants
    boltz=1.38064852e-23  # J/k
   
    # System parameters
    box_size = 1000    # mu
    temperature = 300.0  # K
    viscosity = 0.01  # g/cm s (1 g/cm s = 1 Poise = 0.1 kg/m s)
    cell_radius = 1.0  # mu
    tumble_prob = 0.05  # per timestep
    mean_speed = 20  # mu/s

    # Diffusion constants (Stokes-Einstein relation)
    diffusion_constant = (boltz*temperature)/(6.0*np.pi*0.1*viscosity*1e-6*cell_radius)  # m^2/s
    diffusion_constant *= 1e12   # mu^2/s
    rot_diffusion_constant = (boltz*temperature)/(8.0*np.pi*0.1*viscosity*(1e-6*cell_radius)**3)  # rad^2/s

    # Time
    max_time = 300.0     # s
    time_step = 0.02  # s
    total_steps = int(max_time / time_step)
    timesteps = np.linspace(0, max_time, num=total_steps+1, endpoint=True)

    # Random number seed
    seed = 98
    np.random.seed(seed)

    # Simulation flags
    cell_run = False
    cell_tumble = False
    cell_rbm = False
    cell_tbm = True
        
    # Data analysis flags
    run_ang_corr = False
    run_delay_time = True

    # Graph header
    title = "tmax={}s, t={}s, seed={}".format(max_time, time_step, seed)

    # Parameter string
    paramstring = "T={0:5.1f} K, eta={1:5.3f} g/cm s, D={2:6.4f} mu^2/s, D_r={3:6.4f} rad^2/s, "\
    "tmax={4:3.1f} s, dt={5:5.3f} s, seed={6:2d}, <v>={7:4.1f} mu/s, lambda_T={8:5.2f}, "\
    "run={9:s}, tumble={10:s}, tbm={11:s}, rbm={12:s}".format(temperature, viscosity, 
    diffusion_constant, rot_diffusion_constant, max_time, time_step, seed, mean_speed, 
    tumble_prob, str(cell_run), str(cell_tumble), str(cell_rbm), str(cell_tbm))
