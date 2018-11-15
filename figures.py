import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as anim

class FigureTemplate(matplotlib.figure.Figure):
    """Basic template for a 2D matplotlib figure"""

    def __init__(self, *args, figsize=(10,10), figtitle='template',
            **kwargs):
        super().__init__(*args, **kwargs)


def scatter(data, axis_labels, dataname, title):
    """2D scatter plot. Input lists taken in as [x,y]."""
    fig = plt.figure(FigureClass=FigureTemplate, figtitle=title)
    figname = 'scatter_'+dataname

    plt.title(title)

    if len(data) != 2:
        print('ERROR - Only 2D data may be shown on a scatter plot')
        quit()

    x = data[0]
    y = data[1]

    plt.plot(x, y, 'ko', ms=1)
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])

    plt.tight_layout()
    plt.savefig(figname+'.png')
    plt.close()

def distribution(xdata, dataname, title):
    """Probability distribution of a 1D dataset"""
    fig = plt.figure(FigureClass=FigureTemplate, figtitle=title)
    figname = 'distribution_'+dataname

    n, bins, patches = plt.hist(xdata, bins='auto')

    plt.savefig(figname+'.png')
    plt.close()

def trajectory(pos, box_size, title):
    """Plot trajectories of some positional data for a single cell in 
    2-3 dimensions. 'pos' contains the trajectory of a cell, which may
    be in two or three dimensions."""

    fig = plt.figure(FigureClass=FigureTemplate, figtitle=title)
    figname = 'trajectory_'
    plt.title(title)
    ndim = pos.shape[1]

    if ndim == 2:
        dataname = 'xy'

        x = pos[:,0]
        y = pos[:,1]
        plt.plot(x,y,'-o',lw=0.5,ms=1.2)

        # Extra bits for clarity
        plt.xlabel('x ($\mu$m)')
        plt.ylabel('y ($\mu$m)')
        plt.xlim(-0.5*box_size,0.5*box_size)
        plt.ylim(-0.5*box_size,0.5*box_size)
        plt.plot([-0.5*box_size,0.5*box_size],[0,0],color='k',ls='--',lw=0.5)
        plt.plot([0,0],[-0.5*box_size,0.5*box_size],color='k',ls='--',lw=0.5)

        plt.savefig(figname+dataname+'.png')
        plt.close()

    elif ndim == 3:
        dataname = 'xyz'

        x = pos[:,0]
        y = pos[:,1]
        z = pos[:,2]

        # NEEDS 
        # combine 3d plot with 2d projections into a single figure
        # 3d plot
        ax3d = fig.add_subplot(221, projection='3d')
        ax3d.set_xlabel('x ($\mu$m)')
        ax3d.set_ylabel('y ($\mu$m)')
        ax3d.set_zlabel('z ($\mu$m)')
        ax3d.plot(x,y,z,'-o',lw=0.5,ms=1.2) 

        # x,y projection
        plt.subplot(222)
        plt.plot(x,y)
        plt.xlabel('x ($\mu$m)')
        plt.ylabel('y ($\mu$m)')

        # y,z projection
        plt.subplot(223)
        plt.plot(y,z)
        plt.xlabel('y ($\mu$m)')
        plt.ylabel('z ($\mu$m)')

        # x,z projection
        plt.subplot(224)
        plt.plot(x,z)
        plt.xlabel('x ($\mu$m)')
        plt.ylabel('z ($\mu$m)')

        plt.savefig(figname+dataname+'.png')
        plt.close()

        # 3D rotating animation
        #ax = fig.add_subplot(111, projection='3d')
        #ax.set_title(title)

        #ax.set_xlabel('x ($\mu$m)')
        #ax.set_ylabel('y ($\mu$m)')
        #ax.set_zlabel('z ($\mu$m)')

        # 3D point & line plot
        #ax.plot(x,y,z,'-o',lw=0.5,ms=1.2)

        # rotate the axes and update
        #for angle in range(0, 720):
        #    ax.view_init(30, angle)
        #    plt.draw()
        #    plt.pause(.001)

    else:
        print("ERROR - Unacceptable number ({}) of dimensions in positional "
                "data; should be 2 or 3.".format(ndim))
        quit()
