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

print('Done')

if System.cell_run and System.cell_tumble:
    mu = np.mean(swimmer.run_durations)
    sigma = np.std(swimmer.run_durations)
    num_runs = len(swimmer.run_durations)
    plt.title(System.title+", $\lambda_T={0:5.3f}$, runs={1}".format(System.tumble_prob,num_runs))
    plt.hist(swimmer.run_durations, bins='auto', density=True, edgecolor='black',
        label="Model: $\mu={0:6.3f}$, $\sigma={1:6.3f}$".format(mu,sigma))
    x=np.linspace(0,max(swimmer.run_durations),num=100)
    l = System.tumble_prob/System.time_step
    fit=l*np.exp(-l*x)
    #plt.plot(x,fit,'r',lw=2,label="$\\frac{\lambda_T}{\Delta t} e^{-\lambda_T t/\Delta t }$"+
             #" ; $\lambda_T={:5.3f}$, $\langle t \\rangle={:5.2f}$".format(System.tumble_prob, System.mean_run_dur))
    plt.plot(x,fit,'r',lw=2,label="Fit: $\mu=\sigma={0:6.3f}$".format(System.mean_run_dur))
    plt.yscale('log')
    plt.ylim(0.001,1.1*l)
    plt.xlim(0,max(x))
    plt.ylabel('Probability density')
    plt.xlabel('Run duration (s)')
    #plt.grid(True)
    plt.legend()
    plt.savefig("RunDurLog{0:s}.png".format(System.file_id),dpi=400)
    plt.yscale('linear')
    #plt.ylim(0,1.1*l)
    plt.savefig("RunDur{0:s}.png".format(System.file_id),dpi=400)
    plt.close()

if System.cell_rbm or System.cell_tumble:
    ang=np.rad2deg(Data.compute_angles(np.array(swimmer.swim_history))[0])
    ang=ang[np.where(ang>0.001)]  # remove negligible angles
    mu = np.mean(ang)
    sigma = np.std(ang)
    plt.hist(ang, bins=50, density=True, edgecolor='black', label="Model: $\mu={0:7.3f}$, $\sigma={1:7.3f}$".format(mu,sigma))
    title_dr=System.title+", $D_r={:6.4f}rad^2$".format(System.rot_diffusion_constant)
    title_dr+="$s^{-1}$"
    plt.title(title_dr)
    if System.cell_rbm and not System.cell_tumble:
        # CURRENTLY WRONG because dotted angles are used instead of recorded angles
        x=np.linspace(-20,20,num=100)
        sigma_fit = np.rad2deg(np.sqrt(4*System.rot_diffusion_constant*System.time_step))
        fit=2*ss.norm.pdf(x,0,sigma_fit)
        plt.plot(x,fit,'r',lw=2,label="Fit: $\mu=0$, $\sigma=4D_r\Delta t={0:7.3f}$".format(sigma_fit))
    if System.cell_tumble and not System.cell_rbm:
        x=np.linspace(0,180,num=100)
        mu_fit = 68
        sigma_fit = 36
        fit=ss.norm.pdf(x,mu_fit,sigma_fit)
        plt.plot(x,fit,'r',lw=2,label="Fit: $\mu={0:7.3f}$, $\sigma={1:7.3f}$".format(mu_fit,sigma_fit))
    if System.cell_tumble and System.cell_rbm:
        plt.yscale('log')
    plt.ylabel('Probability density')
    plt.xlabel('Angular deviation between adjacent timesteps (deg)')
    plt.legend()
    plt.savefig("AngDist{0:s}.png".format(System.file_id),dpi=400)
    plt.close()

#----------------------------------------------------------------------------#
#---------------------------------TRAJECTORIES-------------------------------#
#----------------------------------------------------------------------------#

# Create list of cell trajectories (this loop might not be needed, e.g. make
# everything a numpy array and do away with python lists)
print('Extracting data from model...')
brownian_positions = []
positions = []
rbm_ang_disp= []
ang_disp = []
for swimmer in swimmers:
    brownian_positions.append(np.array(swimmer.brownian_history))  # TBM
    positions.append(np.array(swimmer.combined_history))  # TBM and R&T
    rbm_ang_disp.append(np.array(swimmer.rbm_ang_disp_history))  # RBM
    ang_disp.append(np.array(swimmer.ang_disp_history))  # RBM and tumbles

brownian_positions = np.array(brownian_positions)
positions = np.array(positions)
rbm_angles = np.array(rbm_ang_disp)
angles = np.array(ang_disp)

# MODEL DATA
# brownian
xb = brownian_positions[0,:,0]
yb = brownian_positions[0,:,1]
zb = brownian_positions[0,:,2]
rb = np.sqrt(np.square(xb) + np.square(yb) + np.square(zb))
thetab = rbm_angles[0,:]
# swimming & brownian
x = positions[0,:,0]
y = positions[0,:,1]
z = positions[0,:,2]
r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
theta = angles[0,:]
rhat = Data.compute_angles(positions[0])[1]
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
    tt, pos_track, pos_s_track = IO.load_expt('tracks/track34sm.txt')
    xt = pos_track[:,0]
    yt = pos_track[:,1]
    zt = pos_track[:,2]
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
    print("Computing angular correlation...")
    tau, angcorr = Data.ang_corr(rhat,System.time_step)
    print("Done")

    # Save ang. corr. data to file
    filename="AngCorr{0:s}.txt".format(System.file_id)
    print('Saving angular correlation data to {}...'.format(filename))
    IO.save_model(filename,[tau,angcorr],["%.2f","%.6e"],System.param_string)
    print('Done')

    cfit = np.exp(-2.0 * System.rot_diffusion_constant * tau)
    clog = np.where(angcorr<=0, 1e-5, angcorr)
    tc = 1.0/(2.0 * System.rot_diffusion_constant)
    cc = 0.333

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

    # DELAY TIME VS. MEAN SQUARE SCATTER PLOTS
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

    # PROBABILITY DISTRIBUTIONS
    # brownian
    #fg.distribution(xb_tau[0],'$x(\\tau=1)$ $(\mu m)$','x_VS_p_tau1',title_d,tag='bm_')  # x
    #fg.distribution(yb_tau[0],'$y(\\tau=1)$ $(\mu m)$','y_VS_p_tau1',title_d,tag='bm_')  # y
    #fg.distribution(zb_tau[0],'$z(\\tau=1)$ $(\mu m)$','z_VS_p_tau1',title_d,tag='bm_')  # z

    #fit_r=np.linspace(min(rb_tau[0]),max(rb_tau[0]),num=50)
    #fit_p=ss.norm.pdf(fit_r,0,np.sqrt(6*System.diffusion_constant*System.time_step))
    #fg.distribution(rb_tau[0],'$r(\\tau=1)$ $(\mu m)$','r_VS_p_tau1',title_d,tag='bm_',fit=True,fitdata=[fit_r,fit_p],
        #fitlabel=r"$P(r,t)=\frac{1}{\sqrt{12\pi D\Delta t}}\exp\left[{\frac{-r^2}{12D\Delta t}}\right]$")  # r

    #fit_th=np.linspace(-15,15,num=1000)
    #fit_p=ss.norm.pdf(fit_th,0,np.rad2deg(np.sqrt(4*System.rot_diffusion_constant*System.time_step)))
    #fg.distribution(np.rad2deg(thetab_tau[0]),'$\\Theta(\\tau=1)$ $(deg)$','Theta_VS_p_tau1',title_dr, 
        #tag='bm_',fit=True, fitdata=[fit_th,fit_p],
        #fitlabel=r"$P(\Theta_{rbm},t)=\frac{1}{\sqrt{8\pi D_r\Delta t}}\exp\left[{\frac{-\Theta_{rbm}^2}{8D_r\Delta t}}\right]$")  # theta

    print('Done')
