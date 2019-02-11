# external libraries
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

# custom classes
from cell_3d import Cell3D
from data import Data
import figures as fg

class System:
   
    total_cells = 1  # non-interacting
    box_size = 2000   # (micrometres (mu_m))

    # time (s)
    max_time = 34
    time_step = 0.02
    total_steps = int(max_time / time_step)
    timesteps = np.linspace(0,max_time,num=total_steps+1,
            endpoint=True)  # includes t=0

    diffusion_constant  = 1e-5  # water (cm^2 / s)
    diffusion_constant *= 1e8   # (mu_m^2 / s)

    # random number seed
    seed = 98
    np.random.seed(seed)

    title = "Time = {}s, step size = {}s, seed = {}".format(max_time,
        time_step, seed)



class IO:

    def load_track(filename):
        """Load cell tracking data from a text file. Return time (s), 
        positions (mu_m) and smoothed positions (mu_m)."""

        t, x, y, z, xs, ys, zs = np.loadtxt(filename,unpack=True,
                usecols=[0,1,2,3,4,5,6])

        # output track data in same array format as model
        pos = []
        pos_s = []
        for i in range(0,len(x)):
            pos.append(np.array([x[i],y[i],z[i]]))
            pos_s.append(np.array([xs[i],ys[i],zs[i]]))

        pos = np.array(pos)
        pos_s = np.array(pos_s)

        return t, pos, pos_s


# Instantiate cell classes
print('Creating cells...')
swimmers = []
for i in range(System.total_cells):
    swimmers.append(Cell3D(name='Escherichia coli', 
        position=np.array([0.0,0.0,0.0]), speed=20, 
        direction=np.array([1.0,0.0,0.0]), tumble_chance=0.1, 
        time_step=System.time_step))

print('Done')

# Step through time in range 1 <= t <= tmax
print('Computing cell trajectories...')
for time in System.timesteps[1:]:   
    # Update every cell
    for swimmer in swimmers:
        swimmer.update(System.diffusion_constant, System.time_step, 2*np.pi)

print('Done')

# PLACE THIS IN A FUNCTION========
plt.title(System.title)
plt.hist(swimmer.run_durations, bins='auto', density=True, edgecolor='black')
x=np.linspace(1,max(swimmer.run_durations),num=50)
fit=0.1*np.exp(-0.1*x)
plt.plot(x,fit,'r',lw=2,label='$\lambda e^{-\lambda t},\ \lambda=0.1$')
plt.yscale('log')
plt.ylim(0.001,1)
plt.xlim(min(x),max(x))
plt.ylabel('Probability density')
plt.xlabel('Run duration (s)')
#plt.grid(True)
plt.legend()
plt.savefig('test.png')
plt.yscale('linear')
plt.ylim(0,0.2)
plt.savefig('test2.png')
plt.close()

ang=np.array(swimmer.tumble_angles)*(180.0/3.14159)
plt.hist(ang, bins='auto', density=True, edgecolor='black')
x=np.linspace(0,180,num=50)
fit=mlab.normpdf(x,90,39)
plt.plot(x,fit,'r',lw=2,label='Gaussian; $\mu=90^\circ$, $\sigma=39^\circ$')
plt.xlim(min(ang),max(ang))
plt.ylim(0,0.012)
plt.ylabel('Probability density')
plt.xlabel('Tumble angle between subsequent runs (deg)')
plt.legend()
plt.savefig('angle.png')
plt.close()
#==========================================================================
#quit()

# Create list of cell trajectories
print('Extracting model data...')
brownian_positions = []
positions = []
for swimmer in swimmers:
    brownian_positions.append(np.array(swimmer.brownian_history))
    positions.append(np.array(swimmer.swim_history))

brownian_positions = np.array(brownian_positions)
positions = np.array(positions)

# Model data
# brownian
xb = brownian_positions[0,:,0]
yb = brownian_positions[0,:,1]
zb = brownian_positions[0,:,2]
rb = np.linalg.norm(brownian_positions,axis=2)[0]
# swimming
x = positions[0,:,0]
y = positions[0,:,1]
z = positions[0,:,2]
r = np.linalg.norm(positions,axis=2)[0]
print('Done')

# Tracking data
print('Extracting experimental data...')
t_track, pos_track, pos_s_track = IO.load_track('tracks/track34sm.txt')
xt = pos_track[:,0]
yt = pos_track[:,1]
zt = pos_track[:,2]
rt  = np.linalg.norm(pos_track,axis=1)
rt_s = np.linalg.norm(pos_s_track,axis=1)
print('Done')

# Delay time averaging (model)
print('Averaging data...')
tau_values = System.timesteps[1:-1]  # segments in range 1 <= tau < tmax
segments = np.linspace(1,len(tau_values),num=len(tau_values), endpoint=True, 
        dtype='int')  # width (in integer steps) of tau segments

datasets = [x,y,z,r,xb,yb,zb,rb]
data_tau, mean, msq, rms = Data.delay_time_loop(datasets, segments,
    System.time_step)

# swimming data
x_tau = data_tau[0]
y_tau = data_tau[1]
z_tau = data_tau[2]
r_tau = data_tau[3]
mean_x = mean[0]
mean_y = mean[1]
mean_z = mean[2]
mean_r = mean[3]
msq_x  = msq[0]
msq_y  = msq[1]
msq_z  = msq[2]
msq_r  = msq[3]

# brownian data
xb_tau = data_tau[4]
yb_tau = data_tau[5]
zb_tau = data_tau[6]
rb_tau = data_tau[7]
mean_xb = mean[4]
mean_yb = mean[5]
mean_zb = mean[6]
mean_rb = mean[7]
msq_xb  = msq[4]
msq_yb  = msq[5]
msq_zb  = msq[6]
msq_rb  = msq[7]
print('Done')

#--------------------------------#
#----------PLOTTING--------------#
#--------------------------------#
print('Plotting graphs...')

# Trajectory plots (model)
fg.trajectory(positions[0], System.box_size, System.title, tag='model_')
fg.trajectory(brownian_positions[0], System.box_size, System.title, tag='bm_')

# Trajectory plots (experiment)
fg.trajectory(pos_track, System.box_size, System.title, tag='expt_')
fg.scatter([t_track,xt],["t (s)","x ($\mu m$)"],'t_vs_x','',tag='expt_')
fg.scatter([t_track,yt],["t (s)","y ($\mu m$)"],'t_vs_y','',tag='expt_')
fg.scatter([t_track,zt],["t (s)","z ($\mu m$)"],'t_vs_z','',tag='expt_')

# tau vs. mean square plots for xyz and r
fit_xyz = 2*System.diffusion_constant*tau_values
fit_r = 6*System.diffusion_constant*tau_values
fg.scatter([tau_values,msq_xb], 
        ["$\\tau$ (s)","$\langle x^2_{\\tau} \\rangle$ $(\mu m^2)$"],
        'tau_VS_msq_x', System.title,tag='bm_', fit=True, fitdata=[tau_values,fit_xyz])
fg.scatter([tau_values,msq_yb],
        ["$\\tau$ (s)","$\langle y^2_{\\tau} \\rangle$ $(\mu m^2)$"],
        'tau_VS_msq_y', System.title,tag='bm_', fit=True, fitdata=[tau_values,fit_xyz])
fg.scatter([tau_values,msq_zb],
        ["$\\tau$ (s)","$\langle z^2_{\\tau} \\rangle$ $(\mu m^2)$"],
        'tau_VS_msq_z', System.title,tag='bm_', fit=True, fitdata=[tau_values,fit_xyz])
fg.scatter([tau_values,msq_rb],
        ["$\\tau$ (s)","$\langle r^2_{\\tau} \\rangle$ $(\mu m^2)$"],
        'tau_VS_msq_r', System.title, tag='bm_', fit=True, fitdata=[tau_values,fit_r])

# histograms
# brownian displacements
fg.distribution(xb_tau[0],'$x(\\tau=1)$ $(\mu m)$','x_VS_px_tau1',System.title,tag='bm_')
fg.distribution(yb_tau[0],'$y(\\tau=1)$ $(\mu m)$','y_VS_py_tau1',System.title,tag='bm_')
fg.distribution(zb_tau[0],'$z(\\tau=1)$ $(\mu m)$','z_VS_pz_tau1',System.title,tag='bm_')
fg.distribution(rb_tau[0],'$r(\\tau=1)$ $(\mu m)$','r_VS_pr_tau1',System.title,tag='bm_')

# run durations

print('Done')
