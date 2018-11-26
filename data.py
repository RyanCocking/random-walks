import numpy as np

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
        mean = np.mean(segment_data,axis=0)
        meansq = Data.mean_square(segment_data)
        rms = Data.root_mean_square(segment_data)

        return segment_data, mean, meansq, rms, delay_time

    def delay_time_loop(datasets,step_segments,step_size):
        """Apply the delay time averaging method to an arbitrary number
        of 1D datasets.
        
        datasets is an arbitrarily long python list of 1D numpy arrays.
        step_segments is a 1D array of segments, in step units (typically
        steps in time).
        
        Three python lists are returned: the means, mean squares and root mean
        squares, in the same order that their origin datasets were passed
        into this function."""
    
        segmented_data_list = []
        mean_list = []
        meansq_list  = []
        rms_list  = []

        for data in datasets:
            segmented_data = []
            mean = np.zeros(len(step_segments))
            meansq = np.copy(mean)
            rms = np.copy(mean)
    
            for i,segment in enumerate(step_segments,0):
                sd, mean[i], meansq[i], rms[i], tau = Data.delay_time(data,
                        segment, step_size)
                segmented_data.append(sd)

            segmented_data_list.append(np.array(segmented_data))
            mean_list.append(mean)
            meansq_list.append(meansq)
            rms_list.append(rms)
    
        return segmented_data_list, mean_list, meansq_list, rms_list
