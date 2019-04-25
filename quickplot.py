# For plotting things outside of main file

import re
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import figures as fg
from params import System

def msq_options(xlbl, ylbl, xmin=1e-2, xmax=1e3, ymin=1e-1, ymax=1e7, xs="log", ys="log"):
    
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    
    plt.xscale(xs)
    plt.yscale(ys)
    
    plt.legend()
    
    return 0

def gen_fit(const, time):
    return np.array(const*time)

def decide_color(name, colors, extra_color):
    if re.findall("Run",name) and not re.findall("Tum",name) and not re.findall("TBM",name):
        return colors[0]
    elif re.findall("RunTum",name) and not re.findall("TBM",name):
        return colors[1]
    elif re.findall("RunTumTBM",name) or re.findall("All",name):
        return colors[2]
    elif re.findall("RunTBM",name):
        return colors[3]
    else:
        return extra_color

def separate_files(name, rbm_list, norbm_list):
    if re.findall("RBM",name) or re.findall("All",name):
        if re.findall("Run",name) and not re.findall("Tum",name) and not re.findall("TBM",name):
            rbm_list[0] = name
        elif re.findall("RunTum",name) and not re.findall("TBM",name):
            rbm_list[1] = name
        elif re.findall("RunTumTBM",name) or re.findall("All",name):
            rbm_list[2] = name
        elif re.findall("RunTBM",name):
            rbm_list[3] = name
    else:
        if re.findall("Run",name) and not re.findall("Tum",name) and not re.findall("TBM",name):
            norbm_list[0] = name
        elif re.findall("RunTum",name) and not re.findall("TBM",name):
            norbm_list[1] = name
        elif re.findall("RunTumTBM",name):
            norbm_list[2] = name
        elif re.findall("RunTBM",name):
            norbm_list[3] = name

plot_msd=False
plot_msad=False

# Shorthand parameters
Dkt = System.diffusion_constant 
Dswim = System.swim_diffusion_constant
Dr = System.rot_diffusion_constant
lt = System.tumble_prob
v = System.mean_speed
tc = 1.0/(2.0*Dr)  # correlation time

# Fonts
matplotlib.rcParams.update({'font.size': 13})
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16)
matplotlib.rc('axes', labelsize=16)

folder = "results/lt={:s}/".format(str(lt))

# Colours
expt_color='#ff6a00'  # orange
colors=['#ac2fc6','#e0d100','#ea0000','#23c600']  # [m,y,r,g]
bm_color='#306ac9'  # blue

# Units
units_mu="$\mu m^2 s^{-1}$"
units_rad="$rad^2 s^{-1}$"

if plot_msd:

    labels = glob.glob(folder+"*")
    for i,label in enumerate(labels,0):
        labels[i] = label[len(folder)+8:]
        if labels[i]=="TBMRBM":
            labels[i]="Brownian"

    t_expt  = np.loadtxt("results/ExptMeanSquare_r.txt")[:,0]
    r2_expt = np.loadtxt("results/ExptMeanSquare_r.txt")[:,1]

    msd_files  = glob.glob(folder+"*/ModelMeanSquare_r*")
    #msad_files = glob.glob(folder+"*/ModelMeanSquare_theta*")

    title = "$D=$"+"{:5.4f} ".format(Dkt)+units_mu+", $D_{swim}=$"+"{:5.4f} ".format(Dswim)+units_mu+", $D_r=$"+"{:5.4f} ".format(Dr)+units_rad+", $\lambda_T=$"+"{:5.4f}".format(lt)
    title = ""

    # darker
    #colors=['#84009b','#c1a700','#ea0000','#00a508']  # [m,y,r,g]
    #bm_color='#004b8e'  # blue
    

    run_fit_label="$\langle r^2 \\rangle = \langle v \\rangle^2 \\tau^2$"
    bm_fit_label="$\langle r^2 \\rangle = 6D\\tau$"
    swim_fit_label="$\langle r^2 \\rangle = 6D_{swim}\\tau$"

    plt.figure(figsize=(7,9))
    plt.title(title)

    # Plot MSD mean-square displacements THAT CONTAIN RBM
    j=0
    for i,filename in enumerate(msd_files,0):
        mylabel = labels[i]
        mycolor = decide_color(mylabel, colors, bm_color)
        
        data = np.loadtxt(filename)
        t = data[:,0]
        msq_r = data[:,1]
        
        if re.findall("RBM",mylabel) or re.findall("All",mylabel) or re.findall("Brownian",mylabel):
            plt.plot(t, msq_r, label=mylabel,lw=3,ls='-',color=mycolor)
    
    r2_run_fit = gen_fit(v**2,np.square(t))
    r2_bm_fit = gen_fit(6*Dkt, t)
    r2_swim_fit = gen_fit(6*Dswim, t)

    plt.plot(t_expt, r2_expt, label="Experiment_24s", lw=3, ls='--', color=expt_color)

    plt.plot(t, r2_run_fit, label=run_fit_label, lw=2, ls='--',color='k')
    plt.plot(t, r2_bm_fit, label=bm_fit_label, lw=2, ls=':',color='k')
    plt.plot(t, r2_swim_fit, label=swim_fit_label, lw=2, ls='-.',color='k')
    plt.plot([tc,tc], [1e-2,1e7], label="$\\tau_c=(2D_r)^{-1}=$"+"{:4.2f} s".format(tc), lw=2, ls=':', color='r')

    msq_options("Delay time (s)","Mean-square displacement $(\mu m^2)$")
    #plt.show()
    plt.legend()
    plt.tight_layout()
    plt.savefig('MeanSquarePlot_r_RBM.png',dpi=400)
    plt.close()

    # ====================

    plt.figure(figsize=(7,9))
    plt.title(title)

    # Plot MSD mean-square displacements THAT DO NOT CONTAIN RBM
    j=0
    for i,filename in enumerate(msd_files,0):
        mylabel = labels[i]
        mycolor = decide_color(mylabel, colors, bm_color)
        data = np.loadtxt(filename)
        t = data[:,0]
        msq_r = data[:,1]

        if not re.findall("RBM",mylabel) and not re.findall("All",mylabel):
            plt.plot(t, msq_r, label=mylabel,lw=3,ls='-',color=mycolor)

    
    
    plt.plot(t_expt, r2_expt, label="Experiment_24s", lw=3, ls='--', color=expt_color)

    plt.plot(t, r2_run_fit, label=run_fit_label, lw=2, ls='--',color='k')
    plt.plot(t, r2_bm_fit, label=bm_fit_label, lw=2, ls=':',color='k')
    plt.plot(t, r2_swim_fit, label=swim_fit_label, lw=2, ls='-.',color='k')
    plt.plot([tc,tc], [1e-2,1e7], label="$\\tau_c=(2D_r)^{-1}=$"+"{:4.2f} s".format(tc), lw=2, ls=':', color='r')

    msq_options("Delay time (s)","Mean-square displacement $(\mu m^2)$")
    #plt.show()
    plt.legend()
    plt.tight_layout()
    plt.savefig('MeanSquarePlot_r.png')
    plt.close()

    rbm_list=np.array(['s','y','x','f'],dtype='object')
    norbm_list=np.copy(rbm_list)

    for path in msd_files:
        separate_files(path,rbm_list,norbm_list)

    labels = ["Run","RunTum","RunTumTBM","RunTBM"]

    plt.figure(figsize=(7,7))
    plt.title(title)

    for i in range(0,4):
        d1 = np.loadtxt(rbm_list[i])
        d2 = np.loadtxt(norbm_list[i])
        mycolor = decide_color(norbm_list[i], colors, bm_color)
        t  = d1[:,0]
        r1 = d1[:,1]
        r2 = d2[:,1]
        ratio = np.divide(r1,r2) # rbm:no rbm
        plt.plot(t,ratio,label=labels[i],color=mycolor,lw=3,ls='-')
        
    plt.plot([tc,tc], [1e-4,1e0], label="$\\tau_c=(2D_r)^{-1}=$"+"{:4.2f} s".format(tc), lw=2, ls=':', color='r')
        
    msq_options("Delay time (s)", "$Q_{MSD}$", xmin=1e-2, xmax=1e3, ymin=1e-3, ymax=1e0, xs="log", ys="log")
    plt.legend(bbox_to_anchor=(0.5,0.95))
    plt.tight_layout()
    plt.savefig("MeanSquarePlot_Ratio_r.png",dpi=400)
    
if plot_msad:
    
    folder+="lt={:s}_".format(str(lt))

    # Load data
    MSAD_tbm_rbm = np.loadtxt(folder+"TBMRBM/ModelMeanSquare_theta_test-TBM-RBM_1000s.txt")
    MSAD_run_rbm = np.loadtxt(folder+"RunRBM/ModelMeanSquare_theta_Run-RBM_1000s.txt")
    MSAD_run_tbm_rbm = np.loadtxt(folder+"RunTBMRBM/ModelMeanSquare_theta_Run-TBM-RBM_1000s.txt")
    MSAD_run_tum = np.loadtxt(folder+"RunTum/ModelMeanSquare_theta_Run-Tumble_1000s.txt")
    MSAD_run_tum_tbm = np.loadtxt(folder+"RunTumTBM/ModelMeanSquare_theta_test-Run-Tumble-TBM_1000s.txt")
    MSAD_run_tum_tbm_rbm = np.loadtxt(folder+"All/ModelMeanSquare_theta_test-Run-Tumble-TBM-RBM_1000s.txt")
    
    tau = MSAD_tbm_rbm[:,0]
    
    plt.figure(figsize=(6,8))
    
    # Plot graph
    plt.plot(tau, MSAD_tbm_rbm[:,1],label="Brownian",ls='-', lw=4)
    plt.plot(tau, MSAD_run_tbm_rbm[:,1],label="RunTBMRBM",ls='-', lw=2)
    
    plt.plot(tau, MSAD_run_tum[:,1],label="RunTum",ls='-', lw=2)
    plt.plot(tau, MSAD_run_rbm[:,1],label="RunRBM",ls='-', lw=2, color=colors[0])
    plt.plot(tau, MSAD_run_tum_tbm[:,1],label="RunTumTBM",ls='-', lw=2, color=colors[1])
    plt.plot(tau, MSAD_run_tum_tbm_rbm[:,1],label="RunTumTBMRBM",ls='-', lw=2,color=colors[2])
    
    plt.xlabel("Delay time (s)")
    plt.ylabel("Mean-square angular displacement $(rad^2)$")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-2,1e3)
    plt.ylim(1e-2, 1e6)
    
    plt.plot(tau, 4*Dr*tau, label="$\langle \\theta^2 \\rangle = 4D_r\\tau$", color='k', ls='--', lw=2)
    plt.legend()
    plt.savefig("MeanSquareThetaPlot.png",dpi=400)
    plt.close()
    
    plt.figure(figsize=(7,7))
    ratio = MSAD_run_tum_tbm_rbm[:,1] / MSAD_run_tum_tbm[:,1]
    plt.plot(tau, ratio, color='k', lw=3, ls='-')
    plt.plot([1e-2,1e3],[1,1],color='k',lw=1,ls='--')
    
    plt.xlabel("$\\tau$ (s)")
    plt.ylabel("$Q_{MSAD}$")
    plt.xscale('log')
    
    plt.xlim(1e-2,1e3)
    
    plt.savefig("ThetaRatio.png",dpi=400)
    plt.close()

# Trajectories (Brownian and realistic cell)
xyz_brown=np.loadtxt("results/lt=0.02/lt=0.02_TBMRBM/ModelTraj_TBM-RBM_1000s.txt")
t=xyz_brown[:,0]
x=xyz_brown[:,1]
y=xyz_brown[:,2]
z=xyz_brown[:,3]

box=35

# Brownian
# 3D
fig=plt.figure(figsize=(8,8))
ax3d = fig.add_subplot(111, projection='3d')
ax3d.set_xlabel('\n x ($\mu$m)', linespacing=3.2)
ax3d.set_ylabel('\n y ($\mu$m)', linespacing=3.2)
ax3d.set_zlabel('\n z ($\mu$m)', linespacing=3.2)
ax3d.plot(x,y,z,'bo',ms=0.5)
plt.tight_layout()
plt.savefig('XYZ_3D_Brownian.png',dpi=400)
plt.close()
# xy
plt.figure(figsize=(8,8))
plt.plot(x,y,'bo',ms=0.5)
plt.plot(x[0],y[0],'kx',ms=15, label="Origin")
plt.plot(x[0],y[0],marker='o',markeredgecolor='k',markerfacecolor='None',ms=15)
plt.xlabel('x ($\mu$m)')
plt.ylabel('y ($\mu$m)')
plt.tight_layout
plt.savefig("XY_Brownian.png",dpi=400)
plt.close()
# yz
plt.figure(figsize=(8,8))
plt.plot(y,z,'bo',ms=0.5)
plt.plot(y[0],z[0],'kx',ms=15, label="Origin")
plt.plot(y[0],z[0],marker='o',markeredgecolor='k',markerfacecolor='None',ms=15)
plt.xlabel('y ($\mu$m)')
plt.ylabel('z ($\mu$m)')
plt.tight_layout
plt.savefig("YZ_Brownian.png",dpi=400)
plt.close()

# xz
plt.figure(figsize=(8,8))
plt.plot(x,z,'bo',ms=0.5)
plt.plot(x[0],z[0],'kx',ms=15, label="Origin")
plt.plot(x[0],z[0],marker='o',markeredgecolor='k',markerfacecolor='None',ms=15)
plt.xlabel('x ($\mu$m)')
plt.ylabel('z ($\mu$m)')
plt.tight_layout
plt.savefig("XZ_Brownian.png",dpi=400)
plt.close()

# Cell
xyz_cell=np.loadtxt("results/lt=0.02/lt=0.02_All/ModelTraj_test-Run-Tumble-TBM-RBM_1000s.txt")
t=xyz_cell[:,0]
x=xyz_cell[:,1]
y=xyz_cell[:,2]
z=xyz_cell[:,3]

run_durations = np.loadtxt("RunDurations_Run-Tumble-TBM-RBM_200s.txt")
tumble_times=np.cumsum(run_durations)
tumble_index=np.divide(tumble_times,System.time_step).astype('int32')-1
num_tumbles=tumble_times.size

# 3D
fig=plt.figure(figsize=(9,9))
ax3d = fig.add_subplot(111, projection='3d')
ax3d.set_xlabel('\n x ($\mu$m)', linespacing=3.2)
ax3d.set_ylabel('\n y ($\mu$m)', linespacing=3.2)
ax3d.set_zlabel('\n z ($\mu$m)', linespacing=3.2)
ax3d.plot(x[:10000],y[:10000],z[:10000],'bo',ms=1) # first 200 steps (clarity)
plt.tight_layout()
plt.savefig('XYZ_3D_Cell.png',dpi=400)
plt.close()

# xy
plt.figure(figsize=(8,8))
#plt.plot([0,0],[-box,box],ls=':',color='k',lw=1)
#plt.plot([-box,box],[0,0],ls=':',color='k',lw=1)
#plt.xlim(-box,box)
#plt.ylim(-box,box)
plt.plot(x[:10000],y[:10000],'bo',ms=1, label="Runs")
plt.plot(x[tumble_index], y[tumble_index],marker='o',ms=3,markeredgecolor='red',markerfacecolor='red',ls='None',lw=0, label="Tumbles")  # highlight tumbles
plt.plot(x[0],y[0],'kx',ms=15)
plt.plot(x[0],y[0],marker='o',markeredgecolor='k',markerfacecolor='None',ms=15)
plt.xlabel('x ($\mu$m)')
plt.ylabel('y ($\mu$m)')
plt.legend()
plt.tight_layout()
plt.savefig("XY_Cell.png",dpi=400)
plt.close()
# yz
plt.figure(figsize=(8,8))
plt.plot(y[:10000],z[:10000],'bo',ms=1, label="Runs")
plt.plot(y[tumble_index], z[tumble_index],marker='o',ms=3,markeredgecolor='red',markerfacecolor='red',ls='None',lw=0, label="Tumbles")
plt.plot(y[0],z[0],'kx',ms=15)
plt.plot(y[0],z[0],marker='o',markeredgecolor='k',markerfacecolor='None',ms=15)
plt.xlabel('y ($\mu$m)')
plt.ylabel('z ($\mu$m)')
plt.legend()
plt.tight_layout()
plt.savefig("YZ_Cell.png",dpi=400)
plt.close()

# xz
plt.figure(figsize=(8,8))
plt.plot(x[:10000],z[:10000],'bo',ms=1, label="Runs")
plt.plot(x[tumble_index], z[tumble_index],marker='o',ms=3,markeredgecolor='red',markerfacecolor='red',ls='None',lw=0, label="Tumbles")
plt.plot(x[0],z[0],'kx',ms=15)
plt.plot(x[0],z[0],marker='o',markeredgecolor='k',markerfacecolor='None',ms=15)
plt.xlabel('x ($\mu$m)')
plt.ylabel('z ($\mu$m)')
plt.legend()
plt.tight_layout()
plt.savefig("XZ_Cell.png",dpi=400)
plt.close()

# Angular correlation (Brownian)
ac_brown=np.loadtxt("results/lt=0.02/lt=0.02_TBMRBM/AngCorr_TBM-RBM_1000s.txt")

plt.figure(figsize=(8,6))
t=ac_brown[:,0]
acf=ac_brown[:,1]
fit=np.exp(-np.divide(t,tc))
plt.plot(t,acf,color='b',ms=1,marker='o',lw=0,ls='None',label="Brownian data")
plt.plot(t,fit,color='k',ls='--',lw=2, label="$C(\\tau)=\exp(-2D_r\\tau)$")
plt.plot([tc,tc],[-0.2,1.0],color='r',ls=':',lw=2,label="$\\tau_c=(2D_r)^{-1}$="+"{:4.2f} s".format(tc))
plt.ylim(-0.25,1.0)
plt.xlim(1e-2,1e3)
plt.xscale('log')
plt.xlabel('Delay time (s)')
plt.ylabel('Angular correlation function')
plt.legend()
plt.tight_layout()
plt.savefig("ACF.png",dpi=400)
plt.close()

# Early-time MSD and MSAD (Brownian)
msd_brown=np.loadtxt("results/lt=0.02/lt=0.02_TBMRBM/ModelMeanSquare_r_test-TBM-RBM_1000s.txt")
msad_brown=np.loadtxt("results/lt=0.02/lt=0.02_TBMRBM/ModelMeanSquare_theta_test-TBM-RBM_1000s.txt")

plt.figure(figsize=(7,7))
t=msd_brown[:,0]
msd=msd_brown[:,1]
fit=6*Dkt*t
plt.plot(t,msd,color='b',ms=2,marker='o',lw=0,ls='None',label="Brownian data")
plt.plot(t,fit,color='k',ls='--',lw=2, label="$\langle r^2 \\rangle=6D\\tau$")
plt.plot([tc,tc],[1e-3,1e2],color='r',ls=':',lw=2,label="$\\tau_c=(2D_r)^{-1}$="+"{:4.2f} s".format(tc))
plt.ylim(0,16)
plt.xlim(0,10)
#plt.xscale('log')
plt.xlabel('Delay time (s)')
plt.ylabel("Mean-square displacement $(\mu m^2)$")
plt.legend(loc=4)
plt.tight_layout()
plt.savefig("MSD_EarlyTime.png",dpi=400)
plt.close()

plt.figure(figsize=(7,7))
t=msad_brown[:,0]
msad=msad_brown[:,1]
fit=4*Dr*t
plt.plot(t,msad,color='b',ms=2,marker='o',lw=0,ls='None',label="Brownian data")
plt.plot(t,fit,color='k',ls='--',lw=2, label="$\langle \\theta^2 \\rangle=4D_r\\tau$")
plt.plot([tc,tc],[1e-3,1e2],color='r',ls=':',lw=2,label="$\\tau_c=(2D_r)^{-1}$="+"{:4.2f} s".format(tc))
plt.ylim(0,8.0)
plt.xlim(0,10)
#plt.xscale('log')
plt.xlabel('Delay time (s)')
plt.ylabel("Mean-square angular displacement $(rad^2)$")
plt.legend(loc=4)
plt.tight_layout()
plt.savefig("MSAD_EarlyTime.png",dpi=400)
plt.close()
