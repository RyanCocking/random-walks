import matplotlib.pyplot as plt
import numpy as np

from cell_3d import Cell3D
from cell_2d import Cell2D
from cell_1d import Cell1D
import plot

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

# Plot trajectory of first cell
print(len(positions))
plot.trajectory(positions, System.box_size)

quit()

plt.xlabel('x ($\mu$m)')
plt.ylabel('y ($\mu$m)')
plt.xlim(-0.5*System.box_size,0.5*System.box_size)
plt.ylim(-0.5*System.box_size,0.5*System.box_size)
plt.plot([-0.5*System.box_size,0.5*System.box_size],[0,0],color='k',ls='--',
         lw=0.5)
plt.plot([0,0],[-0.5*System.box_size,0.5*System.box_size],color='k',ls='--',
         lw=0.5)
plt.savefig('xy_trajectory.png')
