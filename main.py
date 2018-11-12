import matplotlib.pyplot as plt
import numpy as np

from cell_3d import Cell3D
from cell_2d import Cell2D
from cell_1d import Cell1D
import figures as fg

class System:
    
    total_cells = 5  # number of swimmers (non-interacting)
    
    ndim = 2  # dimensionality of the system

    max_time = 1e2  # seconds
    step_size = 1
    total_steps = int(max_time / step_size)
    timesteps = np.linspace(0,max_time,num=total_steps)
    
    # Random number seed - can be used to replicate trajectories
    seed = 100
    np.random.seed(seed)
    
    box_size = 800  # micrometres

class Data:

    def mean_square(data, axis):
        """Square every element of a dataset and calculate the mean"""
        return np.mean(np.square(data),axis=axis)

    def root_mean_square(data, axis):
        """Square every element of a dataset and calculate the square
        root of the mean"""
        return np.sqrt(np.mean(np.square(data),axis=axis))

    def delay_time_mean_square(data, axis, delay_time):
        """For a given delay time, compute the mean square of a dataset. The 
        delay time is a segment of time that is moved through a dataset and 
        is used to gain statistics equivalent to averaging over many cells, 
        when only one has been simulated.
        
        Delay time is in time steps."""

        N = len(data) - delay_time  # Number of delay time segments
        segment_data = np.zeros(N)

        for i in range(0,N):
            segment_data[i] = data[i+delay_time] - data[i]

        return Data.mean_square(segment_data, axis)

#Run code##################################

# Create many swimming cells
swimmers = []
for i in range(System.total_cells):
    swimmers.append(Cell2D(name='Escherichia coli', position=np.array([0,0]),
        speed=20, direction=np.array([1,1]), tumble_chance=0.1))

# Step through time
for time in System.timesteps:   
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

# take rms and mean square, averaged over all particles
rms_x = Data.root_mean_square(positions[:,:,0],axis=0)
rms_y = Data.root_mean_square(positions[:,:,1],axis=0)

msq_x = Data.mean_square(positions[:,:,0],axis=0)
msq_y = Data.mean_square(positions[:,:,1],axis=0)

fg.scatter([np.sqrt(System.timesteps),rms_x], ["$\sqrt{t}$","$x_{RMS}$ (m)"],
        'sqrt_t_VS_x_rms', title)

fg.scatter([System.timesteps,msq_x], ["t","$\overline{x^2}$ ($m^2$)"],
        't_VS_x_mean_sq', title)
