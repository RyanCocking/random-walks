"""
RandomWalks - A code to simulate run-and-tumble swimming and Brownian motion
    
Copyright (C) 2019  R.C.T.B. Cocking

Email: rctc500@york.ac.uk

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

# Parameters file
import numpy as np

class System:
   
    # Physical constants
    boltz=1.38064852e-23  # J/k
   
    # Time (s)
    max_time = 200.0
    time_step = 0.02
    total_steps = int(max_time / time_step)
    timesteps = np.linspace(0, max_time, num=total_steps+1, endpoint=True)
   
    # System parameters
    box_size = 1000      # mu (has no effect on simulation)
    temperature = 300.0  # K
    viscosity = 0.01     # g/cm s (1 g/cm s = 1 Poise = 0.1 kg/m s)
    cell_radius = 1.0    # mu
    mean_speed = 20      # mu/s
    tumble_prob = 0.02   # per timestep
    mean_run_dur = time_step / tumble_prob  # Poisson interval distribution: <t> = 1/lambda (scaled with timestep)

    # Diffusion constants (Stokes-Einstein equation)
    diffusion_constant = (boltz*temperature)/(6.0*np.pi*0.1*viscosity*1e-6*cell_radius)  # m^2/s
    diffusion_constant *= 1e12   # mu^2/s
    swim_diffusion_constant = diffusion_constant + (mean_speed**2 * mean_run_dur / 3.0) # mu^2/s, valid at large times (Lauga paper)
    rot_diffusion_constant = (boltz*temperature)/(8.0*np.pi*0.1*viscosity*(1e-6*cell_radius)**3)  # rad^2/s

    # Random number seed
    seed = 98
    np.random.seed(seed)

    # Simulation flags
    cell_run = True
    cell_tumble = True
    cell_tbm = True
    cell_rbm = True
    
    # Data analysis flags
    run_expt = True
    run_ang_corr = True
    run_delay_time = True

    # File path of experimental data (relative to main.py)
    expt_file = "track34sm.txt"

    # Graph header
    title = "tmax={}s, t={}s, seed={}".format(max_time, time_step, seed)

    # Parameter string
    param_string = "T={0:5.1f} K, eta={1:5.3f} g/cm s, D={2:6.4f} mu^2/s, D_r={3:6.4f} rad^2/s, "\
    "tmax={4:3.1f} s, dt={5:5.3f} s, seed={6:2d}, <v>={7:4.1f} mu/s, <t>={8:5.2f} s, lambda_T={9:5.3f}, "\
    "run={10:s}, tumble={11:s}, tbm={12:s}, rbm={13:s}".format(temperature, viscosity, 
    diffusion_constant, rot_diffusion_constant, max_time, time_step, seed, mean_speed, 
    mean_run_dur, tumble_prob, str(cell_run), str(cell_tumble), str(cell_tbm), str(cell_rbm))

    # Simulation name
    sim_name = ""
    if cell_run:
        sim_name += "Run-"
    if cell_tumble:
        sim_name += "Tumble-"
    if cell_tbm:
        sim_name += "TBM-"
    if cell_rbm:
        sim_name += "RBM-"
        
    sim_name = sim_name[:-1]
    
    # Unique file ID - used for naming files
    file_id = "_{0:s}_{1:03.0f}s".format(sim_name,max_time)
