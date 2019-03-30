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

        N = len(data) - segment_size  # Number of segment moves
        segment_data = np.zeros(N)

        for i in range(0,N):
            segment_data[i] = data[i+segment_size] - data[i]
        
        delay_time = segment_size*step_size
        #mean = np.mean(segment_data,axis=0)
        meansq = np.mean(np.square(segment_data),axis=0)
        #rms = Data.root_mean_square(segment_data)

        return meansq, delay_time

    def delay_time_loop(datasets,step_segments,step_size):
        """Apply the delay time averaging method to an arbitrary number
        of 1D datasets.
        
        datasets is an arbitrarily long python list of 1D numpy arrays.
        step_segments is a 1D array of segments, in step units (typically
        steps in time).
        
        Three python lists are returned: the means, mean squares and root mean
        squares, in the same order that their origin datasets were passed
        into this function."""
    
        #segmented_data_list = []
        #mean_list = []
        meansq_list  = []
        #rms_list  = []

        for data in datasets:
            #segmented_data = []
            #mean = np.zeros(len(step_segments))
            meansq = np.zeros(len(step_segments))
            #rms = np.copy(mean)
    
            for i,segment in enumerate(step_segments,0):
                #sd, mean[i], meansq[i], rms[i], tau = Data.delay_time(data,
                        #segment, step_size)
                meansq[i], tau = Data.delay_time(data, segment, step_size)
                #segmented_data.append(sd)

            #segmented_data_list.append(np.array(segmented_data))
            #mean_list.append(mean)
            meansq_list.append(meansq)
            #rms_list.append(rms)
    
        return meansq_list

    def compute_angles(coords):
        """
        For a cell trajectory of N coordinates (x,y,z), compute the
        dot product between the two displacement vectors produced by
        groups of three points, to produce an N-2 array of angles.
        
        coords is a 3*N array; an array with each element 
        containing an x,y,z coordinate, i.e. coords[0][0] = x(0).
        """

        N = coords.shape[0]-2
        angles = np.zeros(N, dtype=np.float)

        for i in range(0, N):
            r1 = np.array(coords[i+1] - coords[i])
            r2 = np.array(coords[i+2] - coords[i+1])
            mag_r1 = np.abs(np.linalg.norm(r1))
            mag_r2 = np.abs(np.linalg.norm(r2))
            costheta = np.dot(r1,r2)/(mag_r1*mag_r2)            
            # If statement to guard against any NaNs from arccos
            if np.abs(costheta) < 1:
                angles[i] = np.arccos(costheta)
            else:
                angles[i] = 0.0

        return angles

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
