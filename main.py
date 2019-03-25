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
swimmers = []
for i in range(1):
    swimmers.append(Cell3D(name='Escherichia coli', 
        position=np.array([0.0,0.0,0.0]), speed=System.mean_speed, 
        direction=np.array([1.0,0.0,0.0]), tumble_chance=System.tumble_prob, 
        time_step=System.time_step))

# Print simulation parameters at start
print("Simulating {0:1d} cell of {1:s}".format(len(swimmers),swimmers[0].name))
print(System.paramstring)

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

ang=np.rad2deg(Data.compute_angles(np.array(swimmer.swim_history)))
plt.hist(ang, bins='auto', density=True, edgecolor='black')
x=np.linspace(-20,20,num=100)
fit=2*ss.norm.pdf(x,0,np.rad2deg(np.sqrt(2*System.rot_diffusion_constant*System.time_step)))
title_dr=System.title+", $D_r={:6.4f}rad^2$".format(System.rot_diffusion_constant)
title_dr+="$s^{-1}$"
plt.title(title_dr)
plt.plot(x,fit,'r',lw=2,label=r"$P(\theta_{rbm},t)=\frac{2}{\sqrt{4\pi D_r\Delta t}}\exp\left[{\frac{-\theta_{rbm}^2}{4D_r\Delta t}}\right]$")
plt.xlim(0,16)
plt.ylabel('Probability density')
plt.xlabel('Tumble angle between subsequent runs (deg)')
plt.legend()
plt.savefig('angle.png',dpi=400)
plt.close()

#----------------------------------------------------------------------------#
#---------------------------------TRAJECTORIES-------------------------------#
#----------------------------------------------------------------------------#

# Create list of cell trajectories (this loop might not be needed, e.g. make
# everything a numpy array and do away with python lists)
print('Extracting data from model...')
brownian_positions = []
positions = []
rbm_angles= []
angles = []
for swimmer in swimmers:
    brownian_positions.append(np.array(swimmer.brownian_history))  # TBM
    positions.append(np.array(swimmer.combined_history))  # TBM and R&T
    rbm_angles.append(np.array(swimmer.rbm_angle_history))  # RBM
    angles.append(np.array(swimmer.angle_history))  # RBM and tumbles

brownian_positions = np.array(brownian_positions)
positions = np.array(positions)
rbm_angles = np.array(rbm_angles)
angles = np.array(angles)

# MODEL DATA
# brownian
xb = brownian_positions[0,:,0]
yb = brownian_positions[0,:,1]
zb = brownian_positions[0,:,2]
rb = np.linalg.norm(brownian_positions,axis=2)[0]
thetab = rbm_angles[0,:]   # incorrect theta
# swimming & brownian
x = positions[0,:,0]
y = positions[0,:,1]
z = positions[0,:,2]
r = np.linalg.norm(positions,axis=2)[0]
theta = angles[0,:]   # incorrect theta
print('Done')

# Save model swimming data to file
model_filename="model_{:03.0f}s.txt".format(np.max(System.max_time))
print('Saving model trajectory to {}...'.format(model_filename))
IO.save_model(model_filename,[System.timesteps,x,y,z],["%.2f","%.5e","%.5e","%.5e"],
    System.paramstring)
print('Done')

# EXPERIMENT DATA
print('Loading experimental trajectory from file...')
t_track, pos_track, pos_s_track = IO.load_expt('tracks/track34sm.txt')
xt = pos_track[:,0]
yt = pos_track[:,1]
zt = pos_track[:,2]
rt  = np.linalg.norm(pos_track,axis=1)
rt_s = np.linalg.norm(pos_s_track,axis=1)
print('Done')


# TRAJECTORIES
print('Plotting trajectories...')
# model
fg.trajectory(brownian_positions[0], System.box_size, System.title, tag='bm_')  # brownian
fg.trajectory(positions[0], System.box_size, System.title, tag='model_')  # swimming & brownian

# experiment
fg.trajectory(pos_track, System.box_size, System.title, tag='expt_')
fg.scatter([t_track,xt],["t (s)","x ($\mu m$)"],'t_vs_x',"",tag='expt_')
fg.scatter([t_track,yt],["t (s)","y ($\mu m$)"],'t_vs_y',"",tag='expt_')
fg.scatter([t_track,zt],["t (s)","z ($\mu m$)"],'t_vs_z',"",tag='expt_')
print('Done')

#----------------------------------------------------------------------------#
#-----------------------------------ANALYSIS---------------------------------#
#----------------------------------------------------------------------------#

quit()

# Delay-time averaging
print('Delay-time averaging data...')
tau_values = System.timesteps[1:-1]  # segments in range 1 <= tau < tmax
segments = np.linspace(1,len(tau_values),num=len(tau_values), endpoint=True, 
        dtype='int')  # width (in integer steps) of tau segments

datasets = [x,y,z,r,theta,xb,yb,zb,rb,thetab]
data_tau, mean, msq, rms = Data.delay_time_loop(datasets, segments,
    System.time_step)

# swimming & brownian
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

# brownian
xb_tau = data_tau[5]
yb_tau = data_tau[6]
zb_tau = data_tau[7]
rb_tau = data_tau[8]
thetab_tau = data_tau[9]
#mean_xb = mean[5]
#mean_yb = mean[6]
#mean_zb = mean[7]
#mean_rb = mean[8]
msq_xb  = msq[5]
msq_yb  = msq[6]
msq_zb  = msq[7]
msq_rb  = msq[8]
msq_thetab = msq[9]
print('Done')

# TAU VS. MEAN SQUARE SCATTER PLOTS
print('Plotting graphs...')
title_d=System.title+", $D={:6.4f}\mu m^2$".format(System.diffusion_constant)
title_d+="$s^{-1}$"
title_dr=System.title+", $D_r={:6.4f}rad^2$".format(System.rot_diffusion_constant)
title_dr+="$s^{-1}$"

# brownian
fit_xyz = 2*System.diffusion_constant*tau_values
fit_r = 6*System.diffusion_constant*tau_values
fit_th = 4*System.rot_diffusion_constant*tau_values
fg.scatter([tau_values,msq_xb], 
        ["$\\tau$ (s)","$\langle x^2_{\\tau} \\rangle$ $(\mu m^2)$"],
        'tau_VS_msq_x', title_d,tag='bm_', fit=True, fitdata=[tau_values,fit_xyz])  # x
fg.scatter([tau_values,msq_yb],
        ["$\\tau$ (s)","$\langle y^2_{\\tau} \\rangle$ $(\mu m^2)$"],
        'tau_VS_msq_y', title_d,tag='bm_', fit=True, fitdata=[tau_values,fit_xyz])  # y
fg.scatter([tau_values,msq_zb],
        ["$\\tau$ (s)","$\langle z^2_{\\tau} \\rangle$ $(\mu m^2)$"],
        'tau_VS_msq_z', title_d,tag='bm_', fit=True, fitdata=[tau_values,fit_xyz])  # z
fg.scatter([tau_values,msq_rb],
        ["$\\tau$ (s)","$\langle r^2_{\\tau} \\rangle$ $(\mu m^2)$"],
        'tau_VS_msq_r', title_d, tag='bm_', fit=True, fitdata=[tau_values,fit_r])  # r
fg.scatter([tau_values,msq_thetab],
        ["$\\tau$ (s)","$\langle \\Theta^2_{\\tau} \\rangle$ $(rad^2)$"],'tau_VS_msq_Theta',
        title_dr, tag='bm_', fit=True, fitdata=[tau_values,fit_th],
        fitlabel=r"$\langle \Theta^2 \rangle=4D_r\tau$")  # Theta

# model
fg.scatter([tau_values,msq_x],
        ["$\\tau$ (s)","$\langle x^2_{\\tau} \\rangle$ $(\mu m^2)$"],
        'tau_VS_msq_x', title_d,tag='model_', fit=False, fitdata=[tau_values,fit_xyz])  # x
fg.scatter([tau_values,msq_y],
        ["$\\tau$ (s)","$\langle y^2_{\\tau} \\rangle$ $(\mu m^2)$"],
        'tau_VS_msq_y', title_d,tag='model_', fit=False, fitdata=[tau_values,fit_xyz])  # y
fg.scatter([tau_values,msq_z],
        ["$\\tau$ (s)","$\langle z^2_{\\tau} \\rangle$ $(\mu m^2)$"],
        'tau_VS_msq_z', title_d,tag='model_', fit=False, fitdata=[tau_values,fit_xyz])  # z
fg.scatter([tau_values,msq_r],
        ["$\\tau$ (s)","$\langle r^2_{\\tau} \\rangle$ $(\mu m^2)$"],
        'tau_VS_msq_r', title_d, tag='model_', fit=False, fitdata=[tau_values,fit_r])  # r
fg.scatter([tau_values,msq_theta],
        ["$\\tau$ (s)","$\langle \\Theta^2_{\\tau} \\rangle$ $(rad^2)$"],'tau_VS_msq_Theta',
        title_dr, tag='model_', fit=False)  # Theta


# PROBABILITY DISTRIBUTIONS
# brownian
fg.distribution(xb_tau[0],'$x(\\tau=1)$ $(\mu m)$','x_VS_p_tau1',title_d,tag='bm_')  # x
fg.distribution(yb_tau[0],'$y(\\tau=1)$ $(\mu m)$','y_VS_p_tau1',title_d,tag='bm_')  # y
fg.distribution(zb_tau[0],'$z(\\tau=1)$ $(\mu m)$','z_VS_p_tau1',title_d,tag='bm_')  # z

fit_r=np.linspace(min(rb_tau[0]),max(rb_tau[0]),num=50)
fit_p=ss.norm.pdf(fit_r,0,np.sqrt(6*System.diffusion_constant*System.time_step))
fg.distribution(rb_tau[0],'$r(\\tau=1)$ $(\mu m)$','r_VS_p_tau1',title_d,tag='bm_',fit=True,fitdata=[fit_r,fit_p],
    fitlabel=r"$P(r,t)=\frac{1}{\sqrt{12\pi D\Delta t}}\exp\left[{\frac{-r^2}{12D\Delta t}}\right]$")  # r

fit_th=np.linspace(-15,15,num=1000)
fit_p=ss.norm.pdf(fit_th,0,np.rad2deg(np.sqrt(4*System.rot_diffusion_constant*System.time_step)))
fg.distribution(np.rad2deg(thetab_tau[0]),'$\\Theta(\\tau=1)$ $(deg)$','Theta_VS_p_tau1',title_dr, 
    tag='bm_',fit=True, fitdata=[fit_th,fit_p],
    fitlabel=r"$P(\Theta_{rbm},t)=\frac{1}{\sqrt{8\pi D_r\Delta t}}\exp\left[{\frac{-\Theta_{rbm}^2}{8D_r\Delta t}}\right]$")  # theta

print('Done')
