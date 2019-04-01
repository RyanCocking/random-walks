# For plotting things outside of main file

import matplotlib.pyplot as plt
import numpy as np
import figures as fg
from params import System

data_r = np.loadtxt("ModelMeanSquare_r_test-Run-Tumble-TBM-RBM_1000s.txt")
data_th = np.loadtxt("ModelMeanSquare_theta_test-Run-Tumble-TBM-RBM_1000s.txt")
data_ac = np.loadtxt("AngCorr_test-Run-Tumble-TBM-RBM_1000s.txt")
tau = data_r[:,0]
msq_r = data_r[:,1]
msq_theta = data_th[:,1]
angcorr = data_ac[:,1]

fit_r = 6 * System.diffusion_constant * tau
fit_theta = 4 * System.rot_diffusion_constant * tau

title_d=System.title+", $D={:6.4f}\mu m^2$".format(System.diffusion_constant)
title_d+="$s^{-1}$"
title_d+=", $D_r={:6.4f}rad^2$".format(System.rot_diffusion_constant)
title_d+="$s^{-1}$"

fg.scatter([tau,msq_r],
    ["$\\tau$ (s)","$\langle r^2_{\\tau} \\rangle$ $(\mu m^2)$"],
    'tau_VS_msq_r_crop'+System.file_id, "", tag='QUICKPLOT_', fit=True, fitdata=[tau,fit_r],
    fitlabel="6Dt", logx=True)  # r
fg.scatter([tau,msq_theta],
    ["$\\tau$ (s)","$\langle \\Theta^2_{\\tau} \\rangle$ $(rad^2)$"],'tau_VS_msq_theta_full'+System.file_id,
    title_d, tag='QUICKPLOT_', fit=True, fitdata=[tau,fit_theta],
    fitlabel=r"$\langle \Theta^2 \rangle=4D_r\tau$", logx=True)  # Theta

cfit = np.exp(-2.0 * System.rot_diffusion_constant * tau)
plt.plot(tau,cfit,color='r',label="exp($-2D_r\\tau$)")
plt.plot(tau,angcorr[:-1],'k+',ms=1, label="Model, $D_r={:5.4f} rad^2$/sec".format(System.rot_diffusion_constant))
plt.xscale('log')
plt.xlabel("$\\tau$ (s)")
plt.ylim(-0.2,1.0)
plt.ylabel("$\langle \hat{r}(\\tau)\cdot \hat{r}(0)  \\rangle$")
plt.legend()
plt.savefig("QUICKPLOT_AngCorr{0:s}.png".format(System.file_id),dpi=400)
plt.close()
