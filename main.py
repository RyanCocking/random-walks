import matplotlib.pyplot as plt
import numpy as np

from cell_3d import Cell3D
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


class IO:

    def load_track(filename):
        """Load cell tracking data from a text file. Return time (s), 
        positions (mu_m) and smoothed positions (mu_m)."""

        t, x, y, z, xs, ys, zs = np.loadtxt(filename,unpack=True,usecols=[0,1,2,3,4,5,6])

        # output track data in same array format as model
        pos = []
        pos_s = []
        for i in range(0,len(x)):
            pos.append(np.array([x[i],y[i],z[i]]))
            pos_s.append(np.array([xs[i],ys[i],zs[i]]))

        pos = np.array(pos)
        pos_s = np.array(pos_s)

        return t, pos, pos_s

class ErrorChecks:
    
    # length of position array = length of timesteps array

    def example():
        pass

# Instantiate cell classes
swimmers = []
for i in range(System.total_cells):
    swimmers.append(Cell3D(name='Escherichia coli', 
        position=np.array([0.0,0.0,0.0]), speed=20, 
        direction=np.array([1.0,0.0,0.0]), tumble_chance=0.1))

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

# Model data
x = positions[0,:,0]
y = positions[0,:,1]
z = positions[0,:,2]
r = np.linalg.norm(positions,axis=2)[0]

# Tracking data
t_track, pos_track, pos_s_track = IO.load_track('tracks/track34sm.txt')
x_track = pos_track[:,0]
y_track = pos_track[:,1]
z_track = pos_track[:,2]
r_track  = np.linalg.norm(pos_track,axis=1)
rs_track = np.linalg.norm(pos_s_track,axis=1)

# System info for plot titles
title = "Time = {}s, step size = {}s, seed = {}".format(System.max_time, 
        System.step_size, System.seed)

# Trajectory plots (model)
fg.trajectory(positions[0], System.box_size, title, tag='model_')

# Tracking plots (experiment)
fg.trajectory(pos_track, System.box_size, title, tag='expt_')
fg.scatter([t_track,x_track],["t (s)","x ($\mu m$)"],'t_vs_x_track','',tag='expt_')
fg.scatter([t_track,y_track],["t (s)","y ($\mu m$)"],'t_vs_y_track','',tag='expt_')
fg.scatter([t_track,z_track],["t (s)","z ($\mu m$)"],'t_vs_z_track','',tag='expt_')

# Delay time averaging (model)
tau_values = System.timesteps[1:-1]  # segments in range 1 <= tau < tmax
segments = np.linspace(1,len(tau_values),num=len(tau_values), endpoint=True, 
        dtype='int')  # width (in integer steps) of tau segments

msq_x_tau = np.zeros(len(tau_values))
msq_y_tau = np.copy(msq_x_tau)
msq_z_tau = np.copy(msq_x_tau)
msq_r_tau = np.copy(msq_x_tau)

# Loop over tau, compute mean squares
for i,segment in enumerate(segments,0):
    msq_x_tau[i], tau = Data.delay_time_mean_square(x, segment, 
            System.step_size) 
    msq_y_tau[i], tau = Data.delay_time_mean_square(y, segment, 
            System.step_size) 
    msq_z_tau[i], tau = Data.delay_time_mean_square(z, segment, 
            System.step_size) 

    msq_r_tau[i], tau = Data.delay_time_mean_square(r, segment,
            System.step_size)  # delay time mean


# tau vs. mean square plots for xyz and r
fg.scatter([tau_values,msq_x_tau], 
        ["$\\tau$ (s)","$\langle x^2_{\\tau} \\rangle$ $(\mu m^2)$"],
        'tau_VS_msq_x', title,'model_')
fg.scatter([tau_values,msq_y_tau],
        ["$\\tau$ (s)","$\langle y^2_{\\tau} \\rangle$ $(\mu m^2)$"],
        'tau_VS_msq_y', title,'model_')
fg.scatter([tau_values,msq_z_tau],
        ["$\\tau$ (s)","$\langle z^2_{\\tau} \\rangle$ $(\mu m^2)$"],
        'tau_VS_msq_z', title,'model_')
fg.scatter([tau_values,msq_r_tau],
        ["$\\tau$ (s)","$\langle r^2_{\\tau} \\rangle$ $(\mu m^2)$"],
        'tau_VS_msq_r', title, tag='model_')
fg.scatter([np.log10(tau_values),np.log10(msq_r_tau)],
        ["log$_{10}[\\tau]$","log$_{10}[\langle r^2_{\\tau} \\rangle]$"],
        'log10_tau_VS_log10_msq_r', title, tag='model_',regress=True)

# x vs. t
fg.scatter([System.timesteps,x],["t (s)","x ($\mu m$)"],'t_VS_x',title,tag='model_')
