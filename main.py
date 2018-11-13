import matplotlib.pyplot as plt
import numpy as np

from cell_3d import Cell3D
from cell_2d import Cell2D
from cell_1d import Cell1D
import figures as fg

class System:
    
    total_cells = 1  # number of swimmers (non-interacting)
    
    ndim = 2  # dimensionality of the system

    # time (seconds)
    max_time = 1e2
    step_size = 1
    total_steps = int(max_time / step_size)
    timesteps = np.linspace(0,max_time,num=total_steps+1,
            endpoint=True)  # includes t=0
    
    # Random number seed - can be used to replicate trajectories
    seed = 100
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
    swimmers.append(Cell2D(name='Escherichia coli', 
        position=np.array([0.0,0.0]), speed=20, 
        direction=np.array([1.0,1.0]), tumble_chance=0.1))

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

fg.trajectory(positions, System.box_size, title)

# Positions of cell 1
x = positions[0,:,0]
y = positions[0,:,1]

# Delay time
tau_values = System.timesteps[1:-1]  # tau time segments in range 1 <= tau < tmax
segments = np.linspace(1,len(tau_values),num=len(tau_values), endpoint=True, 
        dtype='int')  # width (in integer steps) of tau segments
msq_r_tau = np.zeros(len(tau_values))

# Loop over tau
for i,segment in enumerate(segments,0):
    msq_r_tau[i], tau = Data.delay_time_mean_square(x, segment,
            System.step_size)  # delay time mean

fg.scatter([tau_values,msq_r_tau], ["$\\tau$ (s)","$\langle x^2_{\\tau} \\rangle$"], 'delay_time', title)

quit()

# take rms and mean square, averaged over all particles
rms_x = Data.root_mean_square(positions[:,:,0],axis=0)
rms_y = Data.root_mean_square(positions[:,:,1],axis=0)

msq_x = Data.mean_square(positions[:,:,0],axis=0)
msq_y = Data.mean_square(positions[:,:,1],axis=0)

fg.scatter([np.sqrt(System.timesteps),rms_x], ["$\sqrt{t}$","$x_{RMS}$ (m)"],
        'sqrt_t_VS_x_rms', title)

fg.scatter([System.timesteps,msq_x], ["t","$\overline{x^2}$ ($m^2$)"],
        't_VS_x_mean_sq', title)
