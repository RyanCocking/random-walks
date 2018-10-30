import matplotlib.pyplot as plt
import numpy as np

from Cell3D import Cell3D
from Cell2D import Cell2D
from Cell1D import Cell1D

class System:
    
    max_time = 1000
    step_size = 1
    total_steps = int(max_time / step_size)
    timesteps = np.linspace(0,max_time,num=total_steps)  # seconds
    
    seed = 100
    np.random.seed(seed)
    pi = np.pi
    
    box_size = 1000  # micrometres

    
def rotation_matrix_x(angle):
    sin = np.sin(angle)
    cos = np.cos(angle)
    
    return np.array([[1,0,0],
                     [0,cos,-sin],
                     [0,sin,cos]])

def rotation_matrix_y(angle):
    sin = np.sin(angle)
    cos = np.cos(angle)
    
    return np.array([[cos,0,sin],
                     [0,1,0],
                     [-sin,0,cos]])
    
def rotation_matrix_z(angle):
    sin = np.sin(angle)
    cos = np.cos(angle)
    
    return np.array([[cos,-sin,0],
                     [sin,cos,0],
                     [0,0,1]])

    
def compute_mode_durations(mode_history, step_size):
    """Calculate the durations of intervals of consecutive runs or tumbles."""

    count = 1
    run_durations = np.array([])
    tumble_durations = np.array([])
    for i in range(1, len(mode_history)):
        # Matching elements are part of the same interval
        if mode_history[i] == mode_history[i-1]:
            count += 1
        # Differing elements indicates the start of a new interval
        elif mode_history[i] != mode_history[i-1]:
            time_interval = count * step_size
            if mode_history[i-1] == True:
                run_durations = np.append(run_durations,time_interval)
            elif mode_history[i-1] == False:
                tumble_durations = np.append(tumble_durations,time_interval)
            count = 1

    return run_durations, tumble_durations


swimmer = Cell3D(name='Escherichia coli', position=np.array([0,0,0]), speed=20,
               direction=np.array([1,1,0]), run_chance=0.3, tumble_chance=0.1)


for time in System.timesteps:   
    swimmer.update(System.step_size, System.pi, rotation_matrix_x,
                   rotation_matrix_y, rotation_matrix_z)    
 
positions = np.array(swimmer.position_history)
x_pos = positions[:,0]
y_pos = positions[:,1]
z_pos = positions[:,2]
displacement = np.array(np.linalg.norm(positions,ord=None,axis=1))

run_intervals, tumble_intervals = compute_mode_durations(swimmer.mode_history,
                                                         System.step_size)

print(run_intervals.size)
print(tumble_intervals.size)

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
plt.savefig('xy_run_tumble.png')
plt.close()

# plot t,<r^2>
plt.plot(System.timesteps,displacement)
plt.ylabel('|r| ($\mu$m)')
plt.xlabel('t (s)')
plt.xlim(0,System.max_time)
plt.savefig('norm_r_vs_t.png')
plt.close()

# plot x,y,z distributions
plt.hist(np.unique(x_pos), bins='auto', density=False)
#plt.plot([0,0],[0,1],lw=0.5,ls='--',color='k')
plt.savefig('x_hist.png')
plt.close()
plt.hist(np.unique(y_pos), bins='auto', density=False)
#plt.plot([0,0],[0,1],lw=0.5,ls='--',color='k')
plt.savefig('y_hist.png')
plt.close()
plt.hist(np.unique(z_pos), bins='auto', density=False)
#plt.plot([0,0],[0,1],lw=0.5,ls='--',color='k')
plt.savefig('z_hist.png')
plt.close()

# plot durations of runs and tumbles
plt.hist(run_intervals,bins='auto',color='r')
plt.hist(tumble_intervals,bins='auto',color='b')
plt.show()
plt.savefig('intervals.png')
plt.close()

