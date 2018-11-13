import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

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

def trajectory(traj_all_cells, box_size, title):
    """Plot trajectories of some positional data for a single cell in 
    1-3 dimensions. \'traj_all_cells\' is 1D list of length equal to the 
    number of cells. Each element contains the trajectory of a cell, 
    which may be in 1, 2 or 3 dimensions."""

    fig = plt.figure(FigureClass=FigureTemplate, figtitle=title)
    figname = 'trajectory_'
    plt.title(title)
    ndim = traj_all_cells[0].shape[1]

    if ndim == 1:
        dataname = 'x'
    elif ndim == 2:
        dataname = 'xy'
        legend_labels = []
        for cell_number,traj in enumerate(traj_all_cells,1):
            x = traj[:,0]
            y = traj[:,1]
            plt.plot(x,y,'-o',lw=0.5,ms=1.2)
            legend_labels.append('Cell {}'.format(cell_number))

        # Extra bits for clarity
        plt.xlabel('x ($\mu$m)')
        plt.ylabel('y ($\mu$m)')
        plt.xlim(-0.5*box_size,0.5*box_size)
        plt.ylim(-0.5*box_size,0.5*box_size)
        plt.plot([-0.5*box_size,0.5*box_size],[0,0],color='k',ls='--',lw=0.5)
        plt.plot([0,0],[-0.5*box_size,0.5*box_size],color='k',ls='--',lw=0.5)

        plt.legend(legend_labels)

    elif ndim == 3:
        dataname = 'xyz'
    else:
        print("ERROR - Unacceptable number ({}) of dimensions in positional "
                "data; should be 1, 2 or 3.".format(ndim))
        quit()

    figname += dataname
    plt.savefig(figname+'.png',dpi=400)
    plt.close()
