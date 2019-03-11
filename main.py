# external libraries
import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np

# custom classes
from cell_3d import Cell3D
from data import Data
from params import System
from my_io import IO
import figures as fg

#----------------------------------------------------------------------------#
#----------------------------------SIMULATION--------------------------------#
#----------------------------------------------------------------------------#

# Instantiate cell classes
print('Creating cells...')
swimmers = []
for i in range(1):
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
        swimmer.update(System.diffusion_constant, System.rot_diffusion_constant,
                       System.time_step, 2*np.pi)

print('Done')

## PLACE THIS IN A FUNCTION========
#plt.title(System.title)
#plt.hist(swimmer.run_durations, bins='auto', density=True, edgecolor='black')
#x=np.linspace(1,max(swimmer.run_durations),num=50)
#fit=0.1*np.exp(-0.1*x)
#plt.plot(x,fit,'r',lw=2,label='$\lambda e^{-\lambda t},\ \lambda=0.1$')
#plt.yscale('log')
#plt.ylim(0.001,1)
#plt.xlim(min(x),max(x))
#plt.ylabel('Probability density')
#plt.xlabel('Run duration (s)')
##plt.grid(True)
#plt.legend()
#plt.savefig('test.png')
#plt.yscale('linear')
#plt.ylim(0,0.2)
#plt.savefig('test2.png')
#plt.close()

#ang=np.rad2deg(Data.compute_angles(np.array(swimmer.swim_history)))
##ang=np.rad2deg(np.array(swimmer.tumble_angles))
#plt.hist(ang, bins='auto', density=True, edgecolor='black')
#x=np.linspace(-20,20,num=100)
#fit=2*ss.norm.pdf(x,0,np.rad2deg(np.sqrt(2*System.rot_diffusion_constant*System.time_step)))
#plt.plot(x,fit,'r',lw=2,label=r"$P(\theta_{rbm},t)=\frac{2}{\sqrt{4\pi D_r\Delta t}}\exp\left[{\frac{-\theta_{rbm}^2}{4D_r\Delta t}}\right]$")
#plt.xlim(min(ang),max(ang))
##plt.ylim(0,0.012)
#plt.ylabel('Probability density')
#plt.xlabel('Tumble angle between subsequent runs (deg)')
#plt.legend()
#plt.savefig('angle.png',dpi=400)
#plt.close()

#----------------------------------------------------------------------------#
#-------------------------------DATA EXTRACTION------------------------------#
#----------------------------------------------------------------------------#

print('Extracting model data...')

# Create list of cell trajectories (this loop might not be needed, e.g. make
# everything a numpy array and do away with python lists)
brownian_positions = []
positions = []
angles = []
for swimmer in swimmers:
    brownian_positions.append(np.array(swimmer.brownian_history))
    positions.append(np.array(swimmer.combined_history))  # swim_history (pure r&t), combined_history (r&t + TBM)
    angles.append(np.array(swimmer.rbm_angle_history))

brownian_positions = np.array(brownian_positions)
positions = np.array(positions)
angles = np.array(angles)

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
theta = angles[0,:]
print('Done')

# Tracking data
print('Loading experimental data...')
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

datasets = [x,y,z,r,theta,xb,yb,zb,rb]
data_tau, mean, msq, rms = Data.delay_time_loop(datasets, segments,
    System.time_step)

# swimming data
#x_tau = data_tau[0]
#y_tau = data_tau[1]
#z_tau = data_tau[2]
#r_tau = data_tau[3]
#mean_x = mean[0]
#mean_y = mean[1]
#mean_z = mean[2]
#mean_r = mean[3]
msq_x  = msq[0]
msq_y  = msq[1]
msq_z  = msq[2]
msq_r  = msq[3]
msq_theta = msq[4]

# brownian data
xb_tau = data_tau[5]
yb_tau = data_tau[6]
zb_tau = data_tau[7]
rb_tau = data_tau[8]
#mean_xb = mean[5]
#mean_yb = mean[6]
#mean_zb = mean[7]
#mean_rb = mean[8]
msq_xb  = msq[5]
msq_yb  = msq[6]
msq_zb  = msq[7]
msq_rb  = msq[8]
print('Done')

#----------------------------------------------------------------------------#
#-----------------------------------PLOTTING---------------------------------#
#----------------------------------------------------------------------------#

print('Plotting graphs...')

# Trajectory plots (model)
fg.trajectory(positions[0], System.box_size, System.title, tag='model_')
fg.trajectory(brownian_positions[0], System.box_size, System.title, tag='bm_')

# Trajectory plots (experiment)
fg.trajectory(pos_track, System.box_size, System.title, tag='expt_')
fg.scatter([t_track,xt],["t (s)","x ($\mu m$)"],'t_vs_x',"",tag='expt_')
fg.scatter([t_track,yt],["t (s)","y ($\mu m$)"],'t_vs_y',"",tag='expt_')
fg.scatter([t_track,zt],["t (s)","z ($\mu m$)"],'t_vs_z',"",tag='expt_')

# tau vs. mean square plots
fit_xyz = 2*System.diffusion_constant*tau_values
fit_r = 6*System.diffusion_constant*tau_values
fg.scatter([tau_values,msq_xb], 
        ["$\\tau$ (s)","$\langle x^2_{\\tau} \\rangle$ $(\mu m^2)$"],
        'tau_VS_msq_x', System.title,tag='bm_', fit=True, fitdata=[tau_values,fit_xyz])  # brownian x
fg.scatter([tau_values,msq_yb],
        ["$\\tau$ (s)","$\langle y^2_{\\tau} \\rangle$ $(\mu m^2)$"],
        'tau_VS_msq_y', System.title,tag='bm_', fit=True, fitdata=[tau_values,fit_xyz])  # brownian y
fg.scatter([tau_values,msq_zb],
        ["$\\tau$ (s)","$\langle z^2_{\\tau} \\rangle$ $(\mu m^2)$"],
        'tau_VS_msq_z', System.title,tag='bm_', fit=True, fitdata=[tau_values,fit_xyz])  # brownian z
fg.scatter([tau_values,msq_rb],
        ["$\\tau$ (s)","$\langle r^2_{\\tau} \\rangle$ $(\mu m^2)$"],
        'tau_VS_msq_r', System.title, tag='bm_', fit=True, fitdata=[tau_values,fit_r])  # brownian r
fg.scatter([tau_values,msq_theta],
        ["$\\tau$ (s)","$\langle \theta^2_{\\tau} \\rangle$ $(rad^2)$"],
        'tau_VS_msq_theta', System.title, tag='model_', fit=False)  # angular deviation


# histograms
fg.distribution(xb_tau[0],'$x(\\tau=1)$ $(\mu m)$','x_VS_px_tau1',System.title,tag='bm_')
fg.distribution(yb_tau[0],'$y(\\tau=1)$ $(\mu m)$','y_VS_py_tau1',System.title,tag='bm_')
fg.distribution(zb_tau[0],'$z(\\tau=1)$ $(\mu m)$','z_VS_pz_tau1',System.title,tag='bm_')

x=np.linspace(min(rb_tau[0]),max(rb_tau[0]),num=50)
fit=ss.norm.pdf(x,0,np.sqrt(2*System.diffusion_constant*System.time_step))
fg.distribution(rb_tau[0],'$r(\\tau=1)$ $(\mu m)$','r_VS_pr_tau1',System.title,tag='bm_',fit=True,fitdata=[x,fit])

# run durations

print('Done')
