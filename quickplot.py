# For plotting things outside of main file

import matplotlib.pyplot as plt
import numpy as np
import figures as fg
from params import System

# swim data
data_r = np.loadtxt("ModelMeanSquare_r_test-Run-Tumble-TBM-RBM_1000s.txt")
data_th = np.loadtxt("ModelMeanSquare_theta_test-Run-Tumble-TBM-RBM_1000s.txt")
data_ac = np.loadtxt("AngCorr_test-Run-Tumble-TBM-RBM_1000s.txt")
traj = np.loadtxt("ModelTraj_test-Run-Tumble-TBM-RBM_1000s.txt")

tau = data_r[:,0]
msq_r = data_r[:,1]
msq_theta = data_th[:,1]
angcorr = data_ac[:,1]
x = traj[:,1]
y = traj[:,2]
z = traj[:,3]

# brownian data
data_r_bm = np.loadtxt("ModelMeanSquare_r_test-TBM-RBM_1000s.txt")
data_th_bm = np.loadtxt("ModelMeanSquare_theta_test-TBM-RBM_1000s.txt")
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
