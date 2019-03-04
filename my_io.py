# Input/output file
import numpy as np

class IO:

    def load_track(filename):
        """Load cell tracking data from a text file. Return time (s), 
        positions (mu_m) and smoothed positions (mu_m)."""

        t, x, y, z, xs, ys, zs = np.loadtxt(filename,unpack=True,
                usecols=[0,1,2,3,4,5,6])

        # output track data in same array format as model
        pos = []
        pos_s = []
        for i in range(0,len(x)):
            pos.append(np.array([x[i],y[i],z[i]]))
            pos_s.append(np.array([xs[i],ys[i],zs[i]]))

        pos = np.array(pos)
        pos_s = np.array(pos_s)

        return t, pos, pos_s

    def save_track(filename):
        pass

