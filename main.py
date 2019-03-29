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
    plt.title(System.title)
    plt.hist(swimmer.run_durations, bins='auto', density=True, edgecolor='black')
    x=np.linspace(0,max(swimmer.run_durations),num=100)
    l = System.tumble_prob/System.time_step
    fit=l*np.exp(-l*x)
    plt.plot(x,fit,'r',lw=2,label="$\\frac{\lambda_T}{\Delta t} e^{-\lambda_T t/\Delta t }$"+
             " ; $\lambda_T={:4.2f}$, $\langle t \\rangle={:4.2f}$".format(System.tumble_prob, 1.0/l))
    plt.yscale('log')
    plt.ylim(0.001,1.1*l)
    plt.xlim(0,max(x))
    plt.ylabel('Probability density')
    plt.xlabel('Run duration (s)')
    #plt.grid(True)
    plt.legend()
    plt.savefig("RunDurLog{0:s}.png".format(System.file_id))
    plt.yscale('linear')
    #plt.ylim(0,1.1*l)
    plt.savefig("RunDur{0:s}.png".format(System.file_id))
    plt.show()
    plt.close()

#if System.cell_rbm or System.cell_tumble:
    #ang=np.rad2deg(Data.compute_angles(np.array(swimmer.swim_history)))
    #plt.hist(ang, bins='auto', density=True, edgecolor='black')
    #x=np.linspace(-20,20,num=100)
    #fit=2*ss.norm.pdf(x,0,np.rad2deg(np.sqrt(4*System.rot_diffusion_constant*System.time_step)))
    #title_dr=System.title+", $D_r={:6.4f}rad^2$".format(System.rot_diffusion_constant)
    #title_dr+="$s^{-1}$"
    #plt.title(title_dr)
    #plt.plot(x,fit,'r',lw=2,label=r"$P(\theta_{rbm},t)=\frac{2}{\sqrt{8\pi D_r\Delta t}}\exp\left[{\frac{-\theta_{rbm}^2}{8D_r\Delta t}}\right]$")
    #plt.xlim(0,16)
    #plt.ylabel('Probability density')
    #plt.xlabel('Angular deviation between adjacent timesteps (deg)')
    #plt.legend()
    #plt.savefig("AngDist{0:s}.png".format(System.file_id),dpi=400)
    #plt.close()

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
directions = []
for swimmer in swimmers:
    brownian_positions.append(np.array(swimmer.brownian_history))  # TBM
    positions.append(np.array(swimmer.combined_history))  # TBM and R&T
    rbm_angles.append(np.array(swimmer.rbm_angle_history))  # RBM
    angles.append(np.array(swimmer.angle_history))  # RBM and tumbles
    directions.append(np.array(swimmer.direction_history))

brownian_positions = np.array(brownian_positions)
positions = np.array(positions)
rbm_angles = np.array(rbm_angles)
angles = np.array(angles)
directions = np.array(directions)

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
rhat = directions[0,:]
print('Done')

# Save trajectory to file
model_filename="ModelTraj{0:s}.txt".format(System.file_id)
print('Saving model trajectory to {}...'.format(model_filename))
IO.save_model(model_filename,[System.timesteps,x,y,z,r],["%.2f","%.6e","%.6e","%.6e","%.6e"],
    System.param_string)
print('Done')

## EXPERIMENT DATA
#print('Loading experimental trajectory from file...')
#t_track, pos_track, pos_s_track = IO.load_expt('tracks/track34sm.txt')
#xt = pos_track[:,0]
#yt = pos_track[:,1]
#zt = pos_track[:,2]
#rt  = np.linalg.norm(pos_track,axis=1)
#rt_s = np.linalg.norm(pos_s_track,axis=1)
#print('Done')

## TRAJECTORIES
#print('Plotting trajectories...')
## model
#fg.trajectory(brownian_positions[0], System.box_size, System.title, tag='bm_')  # brownian
#fg.trajectory(positions[0], System.box_size, System.title, tag='model_')  # swimming & brownian

## experiment
#fg.trajectory(pos_track, System.box_size, System.title, tag='expt_')
#fg.scatter([t_track,xt],["t (s)","x ($\mu m$)"],'t_vs_x',"",tag='expt_')
#fg.scatter([t_track,yt],["t (s)","y ($\mu m$)"],'t_vs_y',"",tag='expt_')
#fg.scatter([t_track,zt],["t (s)","z ($\mu m$)"],'t_vs_z',"",tag='expt_')
#print('Done')

#----------------------------------------------------------------------------#
#-----------------------------------ANALYSIS---------------------------------#
#----------------------------------------------------------------------------#

if System.run_ang_corr:

    # Angular correlation
    print("Computing angular correlation...")
    tau, angcorr = Data.ang_corr(rhat,System.time_step)
    print("Done")

    # Save ang. corr. data to file
    model_filename="AngCorr{0:s}.txt".format(System.file_id)
    print('Saving angular correlation data to {}...'.format(model_filename))
    IO.save_model(model_filename,[tau,angcorr],["%.2f","%.6e"],System.param_string)
    print('Done')

    cfit = np.exp(-2.0 * System.rot_diffusion_constant * tau)
    chfit = np.exp(-System.rot_diffusion_constant * tau)
    clog = np.where(angcorr<=0, 1e-5, angcorr)
    tc = 1.0/(2.0 * System.rot_diffusion_constant)
    cc = 0.333

    plt.plot([tc,tc],[1.0,-0.5],color='k',ls='--',lw=0.5,label="$\\tau = 1/2D_r$")
    plt.plot([0,np.max(tau)],[cc,cc],color='k',ls='--',lw=0.5)
    plt.plot(tau,cfit,color='r',label="exp($-2D_r\\tau$)")
    #plt.plot(tau,chfit,color='b',label="exp($-D_r\\tau$)")
    plt.plot(tau,angcorr,'k+',ms=1, label="Model, $D_r={:5.4f} rad^2$/sec".format(System.rot_diffusion_constant))
    plt.xscale('log')
    plt.xlabel("$\\tau$ (s)")
    plt.ylim(-0.2,1.0)
    plt.ylabel("$\langle \hat{r}(\\tau)\cdot \hat{r}(0)  \\rangle$")
    plt.legend()
    plt.savefig("AngCorr{0:s}.png".format(System.file_id),dpi=400)
    plt.close()

    plt.plot([tc,tc],[1.0,-0.5],color='k',ls='--',lw=0.5,label="$\\tau_c = 1/2D_r = 3$s")
    plt.plot([0,np.max(tau)],[cc,cc],color='k',ls='--',lw=0.5)
    plt.plot(tau,np.where(cfit<=1e-5,1e-5,cfit),color='r',label="exp($-2D_r\\tau$)")
    #plt.plot(tau,np.where(chfit<=1e-5,1e-5,chfit),color='b',label="exp($-D_r\\tau$)")
    plt.plot(tau,clog,'k+',ms=1,label="Model, $D_r={:5.4f} rad^2$/sec".format(System.rot_diffusion_constant))
    plt.xlabel("$\\tau$ (s)")
    plt.ylabel("$\langle \hat{r}(\\tau)\cdot \hat{r}(0)  \\rangle$")
    plt.xlim(0,20)
    plt.yscale('log')
    plt.legend()
    plt.savefig("AngCorrLog{0:s}.png".format(System.file_id),dpi=400)
    plt.close()

if System.run_delay_time:

    # Delay-time averaging
    print('Delay-time averaging data...')
    tau_values = System.timesteps[1:-1]  # segments in range 1 <= tau < tmax
    segments = np.linspace(1,len(tau_values),num=len(tau_values), endpoint=True, 
            dtype='int')  # width (in integer steps) of tau segments

    datasets = [x,y,z]
    msq = Data.delay_time_loop(datasets, segments, System.time_step)

    # swimming & brownian
    #x_tau = data_tau[0]
    #y_tau = data_tau[1]
    #z_tau = data_tau[2]
    #r_tau = data_tau[3]
    #mean_x = mean[0]
    #mean_y = mean[1]
    #mean_z = mean[2]
    #mean_r = mean[3]
    #msq_x  = msq[0]
    #msq_y  = msq[1]
    #msq_z  = msq[2]
    msq_r  = msq[0] +msq[1] +msq[2]
    #msq_theta = msq[4]

    ## brownian
    #xb_tau = data_tau[0]
    #yb_tau = data_tau[1]
    #zb_tau = data_tau[2]
    ##thetab_tau = data_tau[9]
    #mean_xb = mean[0]
    #mean_yb = mean[1]
    #mean_zb = mean[2]
    #msq_xb  = msq[0]
    #msq_yb  = msq[1]
    #msq_zb  = msq[2]
    #msq_rb  = msq_xb + msq_yb + msq_zb
    ##msq_thetab = msq[9]
    print('Done')

    # Save delay time data to file
    model_filename="MeanSquare{0:s}.txt".format(System.file_id)
    print('Saving mean squared displacement data to {}...'.format(model_filename))
    IO.save_model(model_filename,[tau_values,msq_r],["%.2f","%.6e"],System.param_string)
    print('Done')
    
    quit()

    # DELAY TIME VS. MEAN SQUARE SCATTER PLOTS
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
            'tau_VS_msq_r', title_d, tag='3bm_', fit=True, fitdata=[tau_values,fit_r],
            fitlabel="6Dt", limx=[0,200], limy=[0,100])  # r
    fg.scatter([tau_values,msq_rb],
            ["$\\tau$ (s)","$\langle r^2_{\\tau} \\rangle$ $(\mu m^2)$"],
            'tau_VS_msq_r', title_d, tag='2bm_', fit=True, fitdata=[tau_values,fit_r],
            fitlabel="6Dt", limx=[0,10], limy=[0,15])  # r
    #fg.scatter([tau_values,msq_thetab],
            #["$\\tau$ (s)","$\langle \\Theta^2_{\\tau} \\rangle$ $(rad^2)$"],'tau_VS_msq_theta',
            #title_dr, tag='bm_', fit=True, fitdata=[tau_values,fit_th],
            #fitlabel=r"$\langle \Theta^2 \rangle=4D_r\tau$", logx=False, limy=[0,15], limx=[0,20])  # Theta
    #fg.scatter([tau_values,msq_thetab],
            #["$\\tau$ (s)","$\langle \\Theta^2_{\\tau} \\rangle$ $(rad^2)$"],'tau_VS_msq_theta',
            #title_dr, tag='2bm_', fit=True, fitdata=[tau_values,fit_th],
            #fitlabel=r"$\langle \Theta^2 \rangle=4D_r\tau$", logx=True, limy=[0,40])  # Theta

    # model
    #fg.scatter([tau_values,msq_x],
            #["$\\tau$ (s)","$\langle x^2_{\\tau} \\rangle$ $(\mu m^2)$"],
            #'tau_VS_msq_x', title_d,tag='model_', fit=False, fitdata=[tau_values,fit_xyz])  # x
    #fg.scatter([tau_values,msq_y],
            #["$\\tau$ (s)","$\langle y^2_{\\tau} \\rangle$ $(\mu m^2)$"],
            #'tau_VS_msq_y', title_d,tag='model_', fit=False, fitdata=[tau_values,fit_xyz])  # y
    #fg.scatter([tau_values,msq_z],
            #["$\\tau$ (s)","$\langle z^2_{\\tau} \\rangle$ $(\mu m^2)$"],
            #'tau_VS_msq_z', title_d,tag='model_', fit=False, fitdata=[tau_values,fit_xyz])  # z
    #fg.scatter([tau_values,msq_r],
            #["$\\tau$ (s)","$\langle r^2_{\\tau} \\rangle$ $(\mu m^2)$"],
            #'tau_VS_msq_r', title_d, tag='model_', fit=False, fitdata=[tau_values,fit_r])  # r
    #fg.scatter([tau_values,msq_theta],
            #["$\\tau$ (s)","$\langle \\Theta^2_{\\tau} \\rangle$ $(rad^2)$"],'tau_VS_msq_theta',
            #title_dr, tag='model_', fit=False)  # Theta


    # PROBABILITY DISTRIBUTIONS
    # brownian
    fg.distribution(xb_tau[0],'$x(\\tau=1)$ $(\mu m)$','x_VS_p_tau1',title_d,tag='bm_')  # x
    fg.distribution(yb_tau[0],'$y(\\tau=1)$ $(\mu m)$','y_VS_p_tau1',title_d,tag='bm_')  # y
    fg.distribution(zb_tau[0],'$z(\\tau=1)$ $(\mu m)$','z_VS_p_tau1',title_d,tag='bm_')  # z

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
