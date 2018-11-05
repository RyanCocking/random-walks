import matplotlib.pyplot as plt
import numpy as np

from cell_3d import Cell3D
from cell_2d import Cell2D
from cell_1d import Cell1D

class System:
    
    max_time = 1e3  # seconds
    step_size = 1
    total_steps = int(max_time / step_size)
    timesteps = np.linspace(0,max_time,num=total_steps)
    
    # Random number seed - can be used to replicate trajectories
    seed = 100
    np.random.seed(seed)
    
    box_size = 800  # micrometres


swimmer2D = Cell2D(name='Escherichia coli', position=np.array([0,0]), speed=20,
               direction=np.array([1,1]), tumble_chance=0.1)

for time in System.timesteps:   
    swimmer2D.update(System.step_size, 2*np.pi)
 
positions = np.array(swimmer2D.position_history)
x_pos = positions[:,0]
y_pos = positions[:,1]
displacements = np.array(np.linalg.norm(positions,ord=None,axis=1))

# plot x,y
plt.plot(x_pos,y_pos)
plt.xlabel('x ($\mu$m)')
plt.ylabel('y ($\mu$m)')
plt.xlim(-0.5*System.box_size,0.5*System.box_size)
plt.ylim(-0.5*System.box_size,0.5*System.box_size)
plt.plot([-0.5*System.box_size,0.5*System.box_size],[0,0],color='k',ls='--',
         lw=0.5)
plt.plot([0,0],[-0.5*System.box_size,0.5*System.box_size],color='k',ls='--',
         lw=0.5)
plt.savefig('xy_traj.png')
plt.close()

# plot t,<r^2>
plt.plot(System.timesteps, displacements)
plt.ylabel('|r| ($\mu$m)')
plt.xlabel('t (s)')
plt.xlim(0,System.max_time)
plt.savefig('norm_r_vs_t.png')
plt.close()

# plot x,y distributions
plt.hist(np.unique(x_pos), bins='auto', density=False)
plt.xlabel('x ($\mu$m)')
plt.ylabel('Frequency')
plt.savefig('x_hist.png')
plt.close()

plt.hist(np.unique(y_pos), bins='auto', density=False)
plt.xlabel('y ($\mu$m)')
plt.ylabel('Frequency')
plt.savefig('y_hist.png')
plt.close()
