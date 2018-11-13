import matplotlib.pyplot as plt
import numpy as np

from cell_3d import Cell3D
from cell_2d import Cell2D
import figures as fg

class System:
    
    total_cells = 1  # number of swimmers (non-interacting)

    # time (seconds)
    max_time = 1e2
    step_size = 1
    total_steps = int(max_time / step_size)
    timesteps = np.linspace(0,max_time,num=total_steps+1,
            endpoint=True)  # includes t=0
    
    # Random number seed - can be used to replicate trajectories
    seed = 98
    np.random.seed(seed)
    
    box_size = 800  # cell container (micrometres)

class Data:

    def mean_square(data, axis):
        """Square every element of a 1D dataset and calculate the mean"""
        return np.mean(np.square(data),axis=0)

    def root_mean_square(data, axis):
        """Square every element of a 1D dataset and calculate the square
        root of the mean"""
        return np.sqrt(np.mean(np.square(data),axis=0))

    def delay_time_mean_square(data, segment_size, step_size):
        """For a given segment of time (delay time), compute the mean square 
        of a 1D dataset. The segment is moved through a dataset and is used 
        to gain statistics equivalent to averaging over many cells, when 
        only one has been simulated.

        Returns mean square of dataset and delay time."""

        N = len(data) - segment_size  # Number of segment moves
        segment_data = np.zeros(N)

        for i in range(0,N):
            segment_data[i] = data[i+segment_size] - data[i]
        
        delay_time = segment_size*step_size  # segment size (s)

        return np.mean(np.square(segment_data),axis=0), delay_time


class ErrorChecks:
    
    # length of position array = length of timesteps array

    def example():
        pass

#Run code##################################

# Create many swimming cells
swimmers = []
for i in range(System.total_cells):
    swimmers.append(Cell3D(name='Escherichia coli', 
        position=np.array([0.0,0.0,0.0]), speed=20, 
        direction=np.array([1.0,1.0,1.0]), tumble_chance=0.1))

# Step through time in range 1 <= t <= tmax
for time in System.timesteps[1:]:   
    # Update every cell
    for swimmer in swimmers:
        swimmer.update(System.step_size, 2*np.pi)

# Create list of cell trajectories
positions = []
for swimmer in swimmers:
    positions.append(np.array(swimmer.position_history))

positions = np.array(positions)
#Plot data################################
title = "Time = {}s, step size = {}s, seed = {}".format(System.max_time, 
        System.step_size, System.seed)

#fg.trajectory(positions, System.box_size, title)

# Positions of cell 1
x = positions[0,:,0]
y = positions[0,:,1]
z = positions[0,:,2]

# Displacement of cell 1
r = np.linalg.norm(positions,axis=2)[0]

# Trajectory plots - 3D IMPLEMENTATION MISSING
fg.trajectory(positions, System.box_size, title)

# Delay time
tau_values = System.timesteps[1:-1]  # tau time segments in range 1 <= tau < tmax
segments = np.linspace(1,len(tau_values),num=len(tau_values), endpoint=True, 
        dtype='int')  # width (in integer steps) of tau segments

msq_x_tau = np.zeros(len(tau_values))
msq_y_tau = np.copy(msq_x_tau)
msq_z_tau = np.copy(msq_x_tau)
msq_r_tau = np.copy(msq_x_tau)

# Loop over tau
for i,segment in enumerate(segments,0):
    msq_x_tau[i], tau = Data.delay_time_mean_square(x, segment, System.step_size) 
    msq_y_tau[i], tau = Data.delay_time_mean_square(y, segment, System.step_size) 
    msq_z_tau[i], tau = Data.delay_time_mean_square(z, segment, System.step_size) 

    msq_r_tau[i], tau = Data.delay_time_mean_square(r, segment,
            System.step_size)  # delay time mean


# tau vs. mean square plots for xyz and r
fg.scatter([tau_values,msq_x_tau], ["$\\tau$ (s)","$\langle x^2_{\\tau} \\rangle$"], 'delay_time_VS_msq_x', title)

fg.scatter([tau_values,msq_y_tau], ["$\\tau$ (s)","$\langle y^2_{\\tau} \\rangle$"], 'delay_time_VS_msq_y', title)

fg.scatter([tau_values,msq_z_tau], ["$\\tau$ (s)","$\langle z^2_{\\tau} \\rangle$"], 'delay_time_VS_msq_z', title)

fg.scatter([tau_values,msq_r_tau], ["$\\tau$ (s)","$\langle r^2_{\\tau} \\rangle$"], 'delay_time_VS_msq_r', title)
