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
import sys

class Data:

    def mean_square(data, axis=0):
        """Square every element of a 1D dataset and calculate the mean"""
        return np.mean(np.square(data),axis=axis)

    def root_mean_square(data, axis=0):
        """Square every element of a 1D dataset and calculate the square
        root of the mean"""
        return np.sqrt(np.mean(np.square(data),axis=axis))

    def delay_time(data, segment_size, step_size):
        """For a given segment of time (delay time), compute the mean, mean
        square and root mean square of a 1D dataset. The segment is moved 
        through a dataset and is used to gain statistics equivalent to
        averaging over many cells, when only one has been simulated."""

        # Number of segment moves.
        # N = [tmax/dt + 1] - tau/dt    # NOTE: (includes t=0 data point!)
        N = len(data) - segment_size
        
        segment_data = np.zeros(N)

        # num iterations = [tmax/dt + 1] - tau/dt   # NOTE: this is the sample size of the mean
        for i in range(0,N):  
            segment_data[i] = data[i+segment_size] - data[i]
        
        delay_time = segment_size*step_size
        meansq = np.mean(np.square(segment_data),axis=0)
        
        return meansq, delay_time

    def delay_time_loop(datasets,step_segments,step_size):
        """Apply the delay time averaging method to a number
        of 1D datasets to calculate the mean square average.
        
        datasets is a python list of 1D numpy arrays.
        
        step_segments is a 1D array of delay time segments.
        """

        meansq_list  = []

        for data in datasets:
            meansq = np.zeros(len(step_segments))
    
            # num iterations = [tmax/dt - 1]
            for i,segment in enumerate(step_segments,0):  
                meansq[i], tau = Data.delay_time(data, segment, step_size)

            meansq_list.append(meansq)
            
        return meansq_list

    def compute_angles(coords):
        """
        For a cell trajectory of N coordinates (x,y,z), compute the
        dot product between the two displacement vectors produced by
        groups of three points, to produce an N-2 array of angles.
        
        Also calculate a corresponding N-1 array of direction unit 
        vectors, rhat (currently only works for a running cell, since
        a translational Brownian trajectory is independent of any rotational
        Brownian motion).
        
        coords is a 3*N array; an array with each element 
        containing an x,y,z coordinate, i.e. coords[0][0] = x(0).
        """

        N = coords.shape[0]
        angles = np.zeros(N-2, dtype=np.float)
        rhat = np.zeros(N-1, dtype=np.object)

        for i in range(0, N-2):
            r1 = np.array(coords[i+1] - coords[i])
            r2 = np.array(coords[i+2] - coords[i+1])
            mag_r1 = np.abs(np.linalg.norm(r1))
            mag_r2 = np.abs(np.linalg.norm(r2))
            
            # Guard against zero vectors
            if mag_r1 > 0 and mag_r2 > 0:
                costheta = np.dot(r1,r2) / (mag_r1 * mag_r2)
                rhat[i] = r1 / mag_r1
            else:
                costheta = 0
                rhat[i] = rhat[i-1]
            
            # If statement to guard against any NaNs from arccos
            if np.abs(costheta) < 1:
                angles[i] = np.arccos(costheta)
            else:
                angles[i] = 0.0

        if mag_r2 > 0:
            rhat[N-2] = r2 / mag_r2
        else:
            rhat[N-2] = rhat[N-1]

        return angles, rhat

    def ang_corr(rhat, step_size):
        """
        Given an array of N direction unit vectors, compute the angular
        correlation function for N-1 delay times (excludes tau=0).
        
        Over an entire simulation the function should have the form:
        
        .. math::
            <\hat{r}(\tau)\cdot\hat{r}(0)>=e^{-2D_r\tau}
        """

        N = np.shape(rhat)[0]

        delay_time = np.zeros(N-1)
        corr = np.copy(delay_time)

        for i,segment in enumerate(range(1,N), 0):
            corrsum = 0
            samples = N - segment      # 1 <= samples < N
            tau = step_size * segment  # seconds
            for j in range(0,samples):
                corrsum += np.dot(rhat[j], rhat[j+segment])

            delay_time[i] = tau
            corr[i] = corrsum / samples

        return delay_time, corr
