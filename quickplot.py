# For plotting things outside of main file

import re
import glob
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

# Shorthand parameters
Dkt = System.diffusion_constant 
Dswim = System.swim_diffusion_constant
Dr = System.rot_diffusion_constant
lt = System.tumble_prob
v = System.mean_speed

folder = "results/lt={:s}/".format(str(lt))
labels = glob.glob(folder+"*")
for i,label in enumerate(labels,0):
    labels[i] = label[len(folder)+8:]
    if labels[i]=="TBMRBM":
        labels[i]="Brownian"

t_expt  = np.loadtxt("results/ExptMeanSquare_r.txt")[:,0]
r2_expt = np.loadtxt("results/ExptMeanSquare_r.txt")[:,1]

msd_files  = glob.glob(folder+"*/ModelMeanSquare_r*")
#msad_files = glob.glob(folder+"*/ModelMeanSquare_theta*")

units_mu="$\mu m^2 s^{-1}$"
units_rad="$rad^2 s^{-1}$"

title = "$D_{TBM}=$"+"{:5.4f} ".format(Dkt)+units_mu+", $D_{swim}=$"+"{:5.4f} ".format(Dswim)+units_mu+", $D_r=$"+"{:5.4f} ".format(Dr)+units_rad+", $\lambda_T=$"+"{:5.4f}".format(lt)
title = ""

# darker
#colors=['#84009b','#c1a700','#ea0000','#00a508']  # [m,y,r,g]
#bm_color='#004b8e'  # blue
expt_color='#ff6a00'  # orange

# lighter
colors=['#de3fff','#ffee00','#ea0000','#2eff00']  # [m,y,r,g]
bm_color='#3f88ff'  # blue

run_fit_label="$\langle r_{run}^2 \\rangle = \langle v \\rangle^2 \\tau^2$"
bm_fit_label="$\langle r_{brown}^2 \\rangle = 6D_{TBM}\\tau$"
swim_fit_label="$\langle r_{swim}^2 \\rangle = 6D_{swim}\\tau$"

plt.figure(figsize=(6,8))
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

msq_options("$\\tau$ (s)","$\langle r^2 \\rangle (\mu m^2)$")
#plt.show()
plt.savefig('MeanSquarePlot_r_RBM.png',dpi=400)
plt.close()

# ====================

plt.figure(figsize=(6,8))
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

 
 
plt.plot(t_expt, r2_expt, label="Experiment_34s", lw=3, ls='--', color=expt_color)

plt.plot(t, r2_run_fit, label=run_fit_label, lw=2, ls='--',color='k')
plt.plot(t, r2_bm_fit, label=bm_fit_label, lw=2, ls=':',color='k')
plt.plot(t, r2_swim_fit, label=swim_fit_label, lw=2, ls='-.',color='k')

msq_options("$\\tau$ (s)","$\langle r^2 \\rangle (\mu m^2)$")
#plt.show()
plt.savefig('MeanSquarePlot_r.png')
plt.close()

rbm_list=np.array(['s','y','x','f'],dtype='object')
norbm_list=np.copy(rbm_list)

for path in msd_files:
    separate_files(path,rbm_list,norbm_list)

labels = ["Run","RunTum","RunTumTBM","RunTBM"]

plt.figure(figsize=(6,6))
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
    
msq_options("$\\tau$ (s)", "$Q_{MSD}$", xmin=1e-2, xmax=1e3, ymin=1e-3, ymax=1e0, xs="log", ys="log")
plt.legend()
plt.savefig("MeanSquarePlot_Ratio_r.png",dpi=400)
quit()

# swim data
data_r = np.loadtxt("ModelMeanSquare_r_Run-Tumble-TBM-RBM_1000s.txt")
data_th = np.loadtxt("ModelMeanSquare_theta_Run-Tumble-TBM-RBM_1000s.txt")
data_ac = np.loadtxt("AngCorr_Run-Tumble-TBM-RBM_1000s.txt")
traj = np.loadtxt("ModelTraj_Run-Tumble-TBM-RBM_1000s.txt")

tau = data_r[:,0]
msq_r = data_r[:,1]
msq_theta = data_th[:,1]
angcorr = data_ac[:,1]
x = traj[:,1]
y = traj[:,2]
z = traj[:,3]

# brownian data
data_r_bm = np.loadtxt("ModelMeanSquare_r_TBM-RBM_1000s.txt")
data_th_bm = np.loadtxt("ModelMeanSquare_theta_TBM-RBM_1000s.txt")
#data_ac_bm = 

msq_r_bm = data_r_bm[:,1]
msq_theta_bm = data_th_bm[:,1]
#angcorr_bm = data_ac_bm[:,1]

# experiment data
data_expt = np.loadtxt("ExptMeanSquare_r.txt")
track_expt=np.loadtxt("tracks/track34sm.txt")
xt=track_expt[:,1] - track_expt[:,1][0]
yt=track_expt[:,2] - track_expt[:,2][0]
zt=track_expt[:,3] - track_expt[:,3][0]
taue=data_expt[:,0]
rsqt=data_expt[:,1]

fit_r = 6 * System.diffusion_constant * tau
fit_theta = 4 * System.rot_diffusion_constant * tau
fit_swim = 6 * System.swim_diffusion_constant * tau

# title
title_d=System.title+", $D={:6.4f}\mu m^2$".format(System.diffusion_constant)
title_d+="$s^{-1}$"
title_d+=", $D_r={:6.4f}rad^2$".format(System.rot_diffusion_constant)
title_d+="$s^{-1}$"

# MSD and MSAD

# MSD - Full range
plt.figure()
plt.title(System.title)
plt.plot(tau,msq_r,markeredgecolor='k',markerfacecolor='none',ls='none',ms=4,marker='^',label="Swimming")
plt.plot(tau,fit_swim,color='g',lw=2,ls='--',label="$6[D_{TBM}+D_{swim}]\\tau$")
plt.plot(tau,msq_r_bm,markeredgecolor='b',markerfacecolor='none',ls='none',ms=4,marker='o',label="Brownian")
plt.plot(tau,fit_r,color='r',lw=2,ls='--',label="$6D_{TBM}\\tau$")
plt.xlim(tau[0],tau[-1])
plt.ylim(1e-2,1e7)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("$\\tau$ (s)")
plt.ylabel("MSD ($\mu m^2$)")
plt.legend()
plt.savefig("MeansquareDisp.png",dpi=400)
plt.close()

# MSD - Experiment
plt.figure()
plt.title(System.title)
plt.plot(tau,msq_r,markeredgecolor='k',markerfacecolor='none',ls='none',ms=4,marker='^',label="Swimming")
plt.plot(tau,fit_swim,color='g',lw=2,ls='--',label="$6[D_{TBM}+D_{swim}]\\tau$")
plt.plot(tau,msq_r_bm,markeredgecolor='b',markerfacecolor='none',ls='none',ms=4,marker='o',label="Brownian")
plt.plot(tau,fit_r,color='r',lw=2,ls='--',label="$6D_{TBM}\\tau$")
plt.plot(taue,rsqt,markeredgecolor='magenta',markerfacecolor='none',ls='none',ms=4,marker='D',label="Experiment")
plt.xlim(tau[0],tau[-1])
plt.ylim(1e-2,1e7)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("$\\tau$ (s)")
plt.ylabel("MSD ($\mu m^2$)")
plt.legend()
plt.savefig("MeansquareExptDisp.png",dpi=400)
plt.close()

# MSAD - Full range
plt.figure()
plt.title(System.title)
plt.plot(tau,msq_theta,markeredgecolor='k',markerfacecolor='none',ls='none',ms=4,marker='^',label="Swimming")
plt.plot(tau,msq_theta_bm,markeredgecolor='b',markerfacecolor='none',ls='none',ms=4,marker='o',label="Brownian")
plt.plot(tau,fit_theta,color='r',lw=2,ls='--',label="$4D_{r}\\tau$")
plt.xlim(tau[0],tau[-1])
plt.ylim(1e-2,1e7)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("$\\tau$ (s)")
plt.ylabel("MSAD ($rad^2$)")
plt.legend()
plt.savefig("MeansquareAngular.png",dpi=400)
plt.close()

angcorr=angcorr[:-1]

# angular correlation
cfit = np.exp(-2.0 * System.rot_diffusion_constant * tau)
plt.title(title_d)
plt.plot(tau,cfit,color='r',lw=1,ls='--',label="exp($-2D_r\\tau$)")
plt.plot(tau,angcorr,'k+',ms=1, label="Model")
plt.plot([tau[0],tau[-1]],[0,0],'k--',lw=0.5)
plt.xlim(tau[0],tau[-1])
plt.xscale('log')
plt.xlabel("$\\tau$ (s)")
plt.ylim(-0.2,1.0)
plt.ylabel("$\langle \hat{r}(\\tau)\cdot \hat{r}(0)  \\rangle$")
plt.legend()
plt.savefig("QUICKPLOT_AngCorr{0:s}.png".format(System.file_id),dpi=400)
plt.close()

fig = plt.figure()
ax3d = fig.add_subplot(111, projection='3d')
ax3d.set_xlabel('x ($\mu$m)')
ax3d.set_ylabel('y ($\mu$m)')
ax3d.set_zlabel('z ($\mu$m)')
ax3d.plot(x,y,z,lw=0.5,ms=1,color='k',marker='+',ls='-', label="Model")
plt.tight_layout()
plt.title(title_d)
plt.savefig('QUICKPLOT_long3D.png',dpi=400)
plt.close()

# INCLUDING EXPERIMENTAL DATA

# 3d trajectory
x=x[:len(xt)]
y=y[:len(yt)]
z=z[:len(zt)]
box_size=1000

fig = plt.figure()
ax3d = fig.add_subplot(111, projection='3d')
ax3d.set_xlabel('x ($\mu$m)')
ax3d.set_ylabel('y ($\mu$m)')
ax3d.set_zlabel('z ($\mu$m)')
ax3d.plot(x,y,z,lw=1,ms=1.2,color='k',marker='+',ls='-', label="Model")
ax3d.plot(xt,yt,zt,lw=1,ms=1.2,color='r',marker='+',ls='-', label="Experiment")
plt.tight_layout()
plt.title(title_d)
plt.savefig('QUICKPLOT_3D.png',dpi=400)
plt.close()

# x,y projection
plt.plot(x,y,lw=1,ms=1.2,color='k',marker='+',ls='-', label="Model")
plt.plot(xt,yt,lw=1,ms=1.2,color='r',marker='+',ls='-', label="Experiment")
plt.plot([-box_size,box_size],[0,0],'k--',lw=0.5)
plt.plot([0,0],[-box_size,box_size],'k--',lw=0.5)
plt.xlim(min(min(x*1.1),min(xt*1.1)),max(max(x*1.1),max(xt*1.1)))
plt.ylim(min(min(y*1.1),min(yt*1.1)),max(max(y*1.1),max(yt*1.1)))
plt.xlabel('x ($\mu$m)')
plt.ylabel('y ($\mu$m)')
plt.title(title_d)
plt.legend()
plt.savefig('QUICKPLOT_xy.png',dpi=400)
plt.close()

# y,z projection
plt.plot(y,z,lw=1,ms=1.2,color='k',marker='+',ls='-', label="Model")
plt.plot(yt,zt,lw=1,ms=1.2,color='r',marker='+',ls='-', label="Experiment")
plt.plot([-box_size,box_size],[0,0],'k--',lw=0.5)
plt.plot([0,0],[-box_size,box_size],'k--',lw=0.5)
plt.xlim(min(min(y*1.1),min(yt*1.1)),max(max(y*1.1),max(yt*1.1)))
plt.ylim(min(min(z*1.1),min(zt*1.1)),max(max(z*1.1),max(zt*1.1)))
plt.xlabel('y ($\mu$m)')
plt.ylabel('z ($\mu$m)')      
plt.title(title_d)
plt.legend()
plt.savefig('QUICKPLOT_yz.png',dpi=400)
plt.close()

# x,z projection
plt.plot(x,z,lw=1,ms=1.2,color='k',marker='+',ls='-', label="Model")
plt.plot(xt,zt,lw=1,ms=1.2,color='r',marker='+',ls='-', label="Experiment")
plt.plot([-box_size,box_size],[0,0],'k--',lw=0.5)
plt.plot([0,0],[-box_size,box_size],'k--',lw=0.5)
plt.xlim(min(min(x*1.1),min(xt*1.1)),max(max(x*1.1),max(xt*1.1)))
plt.ylim(min(min(z*1.1),min(zt*1.1)),max(max(z*1.1),max(zt*1.1)))
plt.xlabel('x ($\mu$m)')
plt.ylabel('z ($\mu$m)')
plt.title(title_d)
plt.legend()
plt.savefig('QUICKPLOT_xz.png',dpi=400)
plt.close()

# MSD and MSAD
fg.scatter([tau,msq_r],
    ["$\\tau$ (s)","$\langle r^2_{\\tau} \\rangle$ $(\mu m^2)$"],
    'tau_VS_msq_r_COMP'+System.file_id, title_d, tag='QUICKPLOT_', fit=True, fitdata=[taue,rsqt],
    fitlabel="Experiment", limx=[0,25],logy=True)  # r
