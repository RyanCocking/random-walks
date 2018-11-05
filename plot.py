import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

class FigureTemplate(matplotlib.figure.Figure):
    """Basic template for a 2D matplotlib figure"""

    def __init__(self, *args, figtitle='template', **kwargs):
        super().__init__(*args, **kwargs)


def distribution(xdata, dataname):
    """Probability distribution of a 1D dataset"""
    fig = plt.figure(FigureClass=FigureTemplate)
    figname = 'dist_'+dataname

    n, bins, patches = plt.hist(xdata, bins='auto')

    plt.savefig(figname+'.png')
    plt.close()


distribution(np.array([1,2,2,3,3,3,4,4,4]),'example')
