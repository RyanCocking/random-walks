"""
RandomWalks - A code to simulate run-and-tumble swimming and Brownian motion
    
Copyright (C) 2019  R.C.T.B. Cocking

Email: rctc500@york.ac.uk

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

# NOTE: WIP CODE - Lots of extraneous data and figures for dissertation purposes 

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

# NOTE: temp
import matplotlib

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
print("Simulation: {0:s}".format(System.sim_name))
print("Simulating {0:1d} cell of {1:s}".format(len(swimmers),swimmers[0].name))
print(System.param_string)

# Step through time in range 1 <= t <= tmax
print('Computing cell trajectories...')
for time in System.timesteps[1:]:   
    # Update every cell
    for swimmer in swimmers:
        swimmer.update(System.diffusion_constant, System.rot_diffusion_constant,
                       System.time_step, System.cell_run, System.cell_tumble,
                       System.cell_tbm, System.cell_rbm)

print("Number of tumbles = ",len(swimmer.run_durations))
print('Done')

# ====================================
# NOTE - Temporary graph plotting code

#matplotlib.rcParams.update({'font.size': 14})
#matplotlib.rc('xtick', labelsize=15) 
#matplotlib.rc('ytick', labelsize=15)
#matplotlib.rc('axes', labelsize=15)

#if System.cell_run and System.cell_tumble:
    
    #plt.figure(figsize=(6,6))
    ## Run duration distribution
    ## FULL LINEAR PLOT
    #mu = np.mean(swimmer.run_durations)
    #sigma = np.std(swimmer.run_durations)
    #num_runs = len(swimmer.run_durations)
    ##plt.title(System.title+", $\lambda_T={0:5.3f}$, runs={1}".format(System.tumble_prob,num_runs))
    #mybins=np.arange(0,5,step=0.2)  # bin width = 'step' seconds
    #plt.hist(swimmer.run_durations, bins=mybins, density=True, edgecolor='black', facecolor='#1643ff',
    #    label="Data: $\langle t \\rangle={0:6.2f}\pm{1:6.2f}$ s".format(mu,sigma))
    #x=np.linspace(0,max(swimmer.run_durations),num=100)
    #l = System.tumble_prob/System.time_step
    #fit=l*np.exp(-l*x)
    #plt.plot(x,fit,'r',lw=2,label="$P(t)=\\frac{1}{\langle t \\rangle}e^{-t/\langle t \\rangle}=e^{-t}$")
    
    #plt.yscale('linear')
    #plt.ylim(0,1)
    #plt.xlim(0,5)
    #plt.ylabel('Probability density')
    #plt.xlabel('Run duration (s)')
    ##plt.grid(True)
    #plt.tight_layout()
    #plt.legend()
    #plt.savefig("RunDur{0:s}.png".format(System.file_id),dpi=400)
    #plt.close()
    
    ## INSET LOG PLOT - Place manually onto other plot
    #plt.figure(figsize=(7,7))
    #matplotlib.rc('xtick', labelsize=35) 
    #matplotlib.rc('ytick', labelsize=35)
    #matplotlib.rc('axes', labelsize=35)
    #plt.hist(swimmer.run_durations, bins=mybins, density=True, edgecolor='black', lw=1.5, facecolor='#1643ff',
    #label="Data: $\langle t \\rangle={0:6.2f}\pm{1:6.2f}$ s".format(mu,sigma))
    #plt.plot(x,fit,'r',lw=4)
    #plt.yscale('log')
    #plt.ylim(1e-3,1)
    #plt.xlim(0,5)
    ##plt.yticks([1e-3, 1.5e-1, 1e0])
    #plt.xticks([0,2.5,5])
    #plt.tight_layout()
    #plt.savefig("RunDurLog{0:s}.png".format(System.file_id),dpi=200)
    #plt.close()

#matplotlib.rcParams.update({'font.size': 14})
#matplotlib.rc('xtick', labelsize=15) 
#matplotlib.rc('ytick', labelsize=15)
#matplotlib.rc('axes', labelsize=15)

#if System.cell_rbm or System.cell_tumble:
    #plt.figure(figsize=(8,6))
    ## Angle distribution
    #if System.cell_tumble:
        ##Tumble angles 
        #ang=np.rad2deg(np.array(swimmer.tumble_angles))
        #print('Tumble: ',np.max(ang), np.min(ang))
        #mu = np.mean(ang)
        #sigma = np.std(ang)
        #mybins=np.arange(-50,200,step=2)  # bin width = 'step' degrees
        #plt.hist(ang, bins=mybins, density=True, histtype='bar', facecolor='None', edgecolor='#1643ff',
        #    lw=1, label="$\langle \\theta_{{tum}} \\rangle={0:6.2f}\pm{1:6.2f}^\circ$".format(mu,sigma))
        ## Tumble fit
        #x=np.linspace(-50,200,num=500)
        #mu_fit = 68
        #sigma_fit = 36
        #fit=ss.norm.pdf(x,mu_fit,sigma_fit)
        #plt.plot(x,fit,color='#001e99',ls='-',lw=2,label="$\langle \\theta_{{tum}} \\rangle=68\pm36^\circ$")
    #if System.cell_rbm:
        ## RBM angles
        #ang=np.rad2deg(np.array(swimmer.RBM_ANGLES))
        #print('RBM: ',np.max(ang), np.min(ang))
        #mu = np.mean(ang)
        #sigma = np.std(ang)
        #mybins=np.arange(-25,25,step=2)  # bin width = 'step' degrees
        #plt.hist(ang, bins=mybins, density=True, histtype='bar', facecolor='None', edgecolor='r', lw=1, label="$\langle \\theta_r \\rangle={0:6.2f}\pm{1:6.2f}^\circ$".format(mu,sigma))
        ## RBM fit
        #x=np.linspace(-50,50,num=500)
        #sigma_fit = np.rad2deg(np.sqrt(4*System.rot_diffusion_constant*System.time_step))
        #fit=ss.norm.pdf(x,0,sigma_fit)
        #plt.plot(x,fit,color='#a30000',ls='-',lw=2,label="$\langle \\theta_r \\rangle=0\pm\sqrt{{4D_r\Delta t}}$")
    #if System.cell_tumble and System.cell_rbm:
        ## Tumble and RBM
        ##plt.yscale('log')
        #pass
        
    #plt.ylim(0,0.065)
    #plt.xlim(-40,180)
    ##plt.yticks(np.arange(0,0.07,step=0.01))
    #plt.xticks(np.arange(-40, 181, step=20))
    #plt.ylabel('Probability density')
    #plt.xlabel('Angular displacement (deg)')
    #plt.legend()
    #plt.tight_layout()
    #plt.savefig("AngDist{0:s}.png".format(System.file_id),dpi=400)
    #plt.close()

# ====================================
    
#----------------------------------------------------------------------------#
#---------------------------------TRAJECTORIES-------------------------------#
#----------------------------------------------------------------------------#

# Create list of cell trajectories (this loop might not be needed, e.g. make
# everything a numpy array and do away with python lists)
print('Extracting data from model...')
brownian_positions = []
positions = []
rbm_ang_disp = []
ang_disp = []
directions = []
run_durations = []
for swimmer in swimmers:
    brownian_positions.append(np.array(swimmer.brownian_history))  # TBM
    positions.append(np.array(swimmer.combined_history))  # TBM and R&T
    rbm_ang_disp.append(np.array(swimmer.rbm_ang_disp_history))  # RBM
    ang_disp.append(np.array(swimmer.ang_disp_history))  # RBM and tumbles
    directions.append(np.array(swimmer.direction_history))  # unit vectors
    run_durations.append(np.array(swimmer.run_durations))  # run durations

brownian_positions = np.array(brownian_positions)
positions = np.array(positions)
rbm_ang_disp = np.array(rbm_ang_disp)
ang_disp = np.array(ang_disp)
directions = np.array(directions)
run_durations = np.array(run_durations)

# MODEL DATA
# brownian
xb = brownian_positions[0,:,0]
yb = brownian_positions[0,:,1]
zb = brownian_positions[0,:,2]
rb = np.sqrt(np.square(xb) + np.square(yb) + np.square(zb))
thetab = rbm_ang_disp[0,:]
# swimming & brownian
x = positions[0,:,0]
y = positions[0,:,1]
z = positions[0,:,2]
r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
theta = ang_disp[0,:]
rhat = directions[0,:]
print('Done')

# Save run durations to file
filename="RunDurations{0:s}.txt".format(System.file_id)
print('Saving run durations to {}...'.format(filename))
IO.save_model(filename,run_durations,"%.2f",System.param_string)
print('Done')

# Save trajectory to file
filename="ModelTraj{0:s}.txt".format(System.file_id)
print('Saving model trajectory to {}...'.format(filename))
IO.save_model(filename,[System.timesteps,x,y,z,r],["%.2f","%.6e","%.6e","%.6e","%.6e"],
    System.param_string)
print('Done')

print('Plotting model trajectory...')
# model
fg.trajectory(brownian_positions[0], System.box_size, System.title, tag='bm_')  # brownian
fg.trajectory(positions[0], System.box_size, System.title, tag='model_')  # swimming & brownian
print("Done")

if System.run_expt:
    # EXPERIMENT DATA
    print('Loading experimental trajectory from file...')
    tt, pos_track, pos_s_track = IO.load_expt(System.expt_file)
    xt = pos_track[:,0]
    yt = pos_track[:,1]
    zt = pos_track[:,2]
    rhatt = Data.compute_angles(pos_track)[1]
    print('Done')

    # experiment
    print("Printing experimental trajectory...")
    fg.trajectory(pos_track, System.box_size, System.title, tag='expt_')
    fg.scatter([tt,xt],["t (s)","x ($\mu m$)"],'t_vs_x',"",tag='expt_')
    fg.scatter([tt,yt],["t (s)","y ($\mu m$)"],'t_vs_y',"",tag='expt_')
    fg.scatter([tt,zt],["t (s)","z ($\mu m$)"],'t_vs_z',"",tag='expt_')    
    print('Done')

#----------------------------------------------------------------------------#
#-----------------------------------ANALYSIS---------------------------------#
#----------------------------------------------------------------------------#

if System.run_ang_corr:

    # Angular correlation
    print("Computing model angular correlation...")
    tau, angcorr = Data.ang_corr(rhat,System.time_step)
    print("Done")

    # Save ang. corr. data to file
    filename="AngCorr{0:s}.txt".format(System.file_id)
    print('Saving model angular correlation data to {}...'.format(filename))
    IO.save_model(filename,[tau,angcorr],["%.2f","%.6e"],System.param_string)
    print('Done')
    
    if System.run_expt:
        # Angular correlation
        print("Computing experiment angular correlation...")
        tau_t, angcorrt = Data.ang_corr(rhatt,System.time_step)
        print("Done")
        
        # Save to file
        filename="ExptAngCorr.txt"
        print('Saving experimental angular correlation data to {}...'.format(filename))
        IO.save_model(filename,[tau_t,angcorrt],["%.2f","%.6e"],"Experiment")
        print('Done')

    cfit = np.exp(-2.0 * System.rot_diffusion_constant * tau)
    clog = np.where(angcorr<=0, 1e-5, angcorr)
    tc = 1.0/(2.0 * System.rot_diffusion_constant)
    cc = 0.333

    # NOTE - Temporary angular correlation plot
    plt.plot([tc,tc],[1.0,-0.5],color='k',ls='--',lw=0.5,label="$\\tau = 1/2D_r$")
    plt.plot([0,np.max(tau)],[cc,cc],color='k',ls='--',lw=0.5)
    plt.plot(tau,cfit,color='r',label="exp($-2D_r\\tau$)")
    plt.plot(tau,angcorr,'k+',ms=1, label="Model, $D_r={:5.4f} rad^2$/sec".format(System.rot_diffusion_constant))
    plt.xscale('log')
    plt.xlabel("$\\tau$ (s)")
    plt.ylim(-0.2,1.0)
    plt.ylabel("$\langle \hat{r}(\\tau)\cdot \hat{r}(0)  \\rangle$")
    plt.legend()
    plt.savefig("AngCorr{0:s}.png".format(System.file_id),dpi=400)
    plt.close()

if System.run_delay_time:

    # Delay time averaging
    print('Delay time averaging data...')
    # Model
    tau = System.timesteps[1:-1]  # segments in range 1 <= tau < tmax
    segments = np.linspace(1,len(tau), num=len(tau), endpoint=True, 
            dtype='int')  # width (in integer steps) of tau segments
    msq = Data.delay_time_loop([x,y,z,theta], segments, System.time_step) 
    msq_r  = msq[0] + msq[1] + msq[2]
    msq_theta = msq[3]
    
    if System.run_expt:
        # Experiment
        tau_t = tt[1:-1]
        segments = np.linspace(1, len(tau_t), num=len(tau_t), endpoint=True, dtype='int')
        msqt = Data.delay_time_loop([xt,yt,zt], segments, tau_t[1]-tau_t[0])
        msq_rt = msqt[0] + msqt[1] + msqt[2]

    print('Done')

    # Save delay time data to file
    # Model
    filename="ModelMeanSquare_r{0:s}.txt".format(System.file_id)
    print('Saving model mean square displacement data to {}...'.format(filename))
    IO.save_model(filename,[tau,msq_r],["%.2f","%.6e"],System.param_string)
    print('Done')
    filename="ModelMeanSquare_theta{0:s}.txt".format(System.file_id)
    print('Saving model mean square angular displacement data to {}...'.format(filename))
    IO.save_model(filename,[tau,msq_theta],["%.2f","%.6e"],System.param_string)
    print('Done')
    
    if System.run_expt:
        # Experiment
        filename="ExptMeanSquare_r.txt"
        print('Saving experiment mean square displacement data to {}...'.format(filename))
        IO.save_model(filename,[tau_t,msq_rt],["%.2f","%.6e"])
        print('Done')

    # NOTE - Graph plotting
    print('Plotting graphs...')
    title_d=System.title+", $D={:6.4f}\mu m^2$".format(System.diffusion_constant)
    title_d+="$s^{-1}$"
    title_d+=", $D_r={:6.4f}rad^2$".format(System.rot_diffusion_constant)
    title_d+="$s^{-1}$"

    # Model
    fit_xyz = 2*System.diffusion_constant*tau
    fit_r = 6*System.diffusion_constant*tau
    fit_theta = 4*System.rot_diffusion_constant*tau
    fg.scatter([tau,msq_r],["$\\tau$ (s)","$\langle r^2_{\\tau} \\rangle$ $(\mu m^2)$"],
            'tau_VS_msq_r_full'+System.file_id, title_d, tag='model_', fit=False, fitdata=[tau,fit_r],
            fitlabel=r"$6D\tau$")  # r
    fg.scatter([tau,msq_r],
            ["$\\tau$ (s)","$\langle r^2_{\\tau} \\rangle$ $(\mu m^2)$"],
            'tau_VS_msq_r_crop'+System.file_id, title_d, tag='model_', fit=False, fitdata=[tau,fit_r],
            fitlabel=r"$6D\tau$", limx=[0,10])  # r
    fg.scatter([tau,msq_theta],
            ["$\\tau$ (s)","$\langle \\Theta^2_{\\tau} \\rangle$ $(rad^2)$"],'tau_VS_msq_theta_full'+System.file_id,
            title_d, tag='model_', fit=False, fitdata=[tau,fit_theta],
            fitlabel=r"$\langle \Theta^2 \rangle=4D_r\tau$")  # Theta
    fg.scatter([tau,msq_theta],
            ["$\\tau$ (s)","$\langle \\Theta^2_{\\tau} \\rangle$ $(rad^2)$"],'tau_VS_msq_theta_crop'+System.file_id,
            title_d, tag='model_', fit=False, fitdata=[tau,fit_theta],
            fitlabel=r"$\langle \Theta^2 \rangle=4D_r\tau$",limx=[0,10])  # Theta

    if System.run_expt:
        # Experiment
        fg.scatter([tau_t,msq_rt],
                ["$\\tau$ (s)","$\langle r^2_{\\tau} \\rangle$ $(\mu m^2)$"],
                'tau_VS_msq_r_full', "Experiment", tag='expt_', fit=False, fitdata=[tau,fit_r],
                fitlabel="6Dt")  # r
        fg.scatter([tau_t,msq_rt],
                ["$\\tau$ (s)","$\langle r^2_{\\tau} \\rangle$ $(\mu m^2)$"],
                'tau_VS_msq_r_crop', "Experiment", tag='expt_', fit=False, fitdata=[tau,fit_r],
                fitlabel="6Dt", limx=[0,10])  # r

    print("Done")

