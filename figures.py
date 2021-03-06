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

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as ss

folder = 'plots/'

class FigureTemplate(mpl.figure.Figure):
    """Basic template for a 2D matplotlib figure"""

    def __init__(self, *args, figsize=(16,16), figtitle='template',
            **kwargs):
        super().__init__(*args, **kwargs)


    def clarity(box_size):
        """Add extra features to a plot for clarity, such as dotted lines
        at x,y = 0 and the bounding box edges."""

        # Limit edges of plot
        plt.xlim(-0.5*box_size,0.5*box_size)
        plt.ylim(-0.5*box_size,0.5*box_size)
        # Mark origin with dotted lines
        plt.plot([-0.5*box_size,0.5*box_size],[0,0],color='k',ls='--',lw=0.5)
        plt.plot([0,0],[-0.5*box_size,0.5*box_size],color='k',ls='--',lw=0.5)


def scatter(data, axis_labels, dataname, title, tag="", final=True, 
            regress=False, fit=False, fitdata=[[0],[0]], fitlabel="Fit", 
            logx=False, logy=False, limx=[], limy=[]):
    """2D scatter plot. Input lists taken in as [x,y]."""

    fig = plt.figure(FigureClass=FigureTemplate, figtitle=title)
    ax = fig.add_subplot(111)
    figname = 'scatter_'+dataname

    plt.title(title)

    if len(data) != 2:
        print('ERROR - Only 2D data may be shown on a scatter plot')
        quit()

    x = data[0]
    y = data[1]

    plt.plot(x, y, 'ko', ms=0.5, label="Model")
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])

    # optional linear regression
    if regress:
        reg_data = ss.linregress(x,y)
        slope = reg_data[0]
        yint = reg_data[1]
        plt.text(0.01,0.95,'Y = {:.4}X + {:.4}'.format(slope,yint), 
                transform = ax.transAxes, color='red', label=fitlabel)
        figname += '_linreg'

    # optional line fitting
    if fit:
        xfit = fitdata[0]
        yfit = fitdata[1]
        plt.plot(xfit,yfit,'r--',lw=1,label=fitlabel)
        figname += '_fitted'

    if final:
        if logx:
            plt.xscale('log')
        if logy:
            plt.yscale('log')
        
        if limx:
            plt.xlim(limx[0],limx[1])
        if limy:
            plt.ylim(limy[0],limy[1])
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(folder+tag+figname+'.png',dpi=400)
        plt.close()


def distribution(xdata, xlabel, dataname, title, tag="_", fit=False, fitdata=[[0],[0]], fitlabel="Fit"):
    """
    Probability distribution of a 1D dataset
    """

    fig = plt.figure(FigureClass=FigureTemplate, figtitle=title)
    figname = 'distribution_'+dataname

    n, bins, patches = plt.hist(xdata, bins='auto',edgecolor='black',density=True)

    plt.xlabel(xlabel)
    plt.ylabel('Probability density')
    plt.title(title)

    if fit:
        xfit = fitdata[0]
        yfit = fitdata[1]
        plt.plot(xfit,yfit,'r-',lw=1,label=fitlabel)
        figname += '_fitted'

    plt.tight_layout()
    plt.legend()
    plt.savefig(folder+tag+figname+'.png',dpi=400)
    plt.close()

def trajectory(pos, box_size, title, tag=""):
    """Plot trajectories of some positional data for a single cell in
    2-3 dimensions. 'pos' contains the trajectory of a cell, which may
    be in two or three dimensions."""

    fig = plt.figure(FigureClass=FigureTemplate, figtitle=title)
    figname = 'trajectory_'
    ndim = pos.shape[1]

    if ndim == 2:
        dataname = 'xy'

        x = pos[:,0]
        y = pos[:,1]
        plt.plot(x,y,'-o',lw=1,ms=1.2)

        plt.xlabel('x ($\mu$m)')
        plt.ylabel('y ($\mu$m)')
        FigureTemplate.clarity(box_size)
        plt.title(title)
        plt.savefig(folder+tag+figname+dataname+'.png',dpi=400)
        plt.close()

    elif ndim == 3:

        x = pos[:,0]
        y = pos[:,1]
        z = pos[:,2]

        # 3d plot
        ax3d = fig.add_subplot(111, projection='3d')
        ax3d.set_xlabel('x ($\mu$m)')
        ax3d.set_ylabel('y ($\mu$m)')
        ax3d.set_zlabel('z ($\mu$m)')
        ax3d.plot(x,y,z,'-',lw=1,ms=1.2)
        plt.tight_layout()
        plt.title(title)
        plt.savefig(folder+tag+figname+'3D.png',dpi=400)
        plt.close()

        # x,y projection
        plt.plot(x,y,'-',lw=1,ms=1.2)
        plt.xlabel('x ($\mu$m)')
        plt.ylabel('y ($\mu$m)')
        FigureTemplate.clarity(box_size)       
        plt.title(title)
        plt.savefig(folder+tag+figname+'xy.png',dpi=400)
        plt.close()

        # y,z projection
        plt.plot(y,z,'-',lw=1,ms=1.2)
        plt.xlabel('y ($\mu$m)')
        plt.ylabel('z ($\mu$m)')
        FigureTemplate.clarity(box_size)       
        plt.title(title)
        plt.savefig(folder+tag+figname+'yz.png',dpi=400)
        plt.close()

        # x,z projection
        plt.plot(x,z,'-',lw=1,ms=1.2)
        plt.xlabel('x ($\mu$m)')
        plt.ylabel('z ($\mu$m)')
        FigureTemplate.clarity(box_size)       
        plt.title(title)
        plt.savefig(folder+tag+figname+'xz.png',dpi=400)
        plt.close()

    else:
        print("ERROR - Unacceptable number ({}) of dimensions in positional "
                "data; should be 2 or 3.".format(ndim))
        quit()
