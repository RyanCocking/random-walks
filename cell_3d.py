import numpy as np

def Rx(angle):
    sin = np.sin(angle)
    cos = np.cos(angle)

    return np.array([[1,0,0],
                     [0,cos,-sin],
                     [0,sin,cos]])

def Ry(angle):
    sin = np.sin(angle)
    cos = np.cos(angle)

    return np.array([[cos,0,sin],
                     [0,1,0],
                     [-sin,0,cos]])

def Rz(angle):
    sin = np.sin(angle)
    cos = np.cos(angle)

    return np.array([[cos,-sin,0],
                     [sin,cos,0],
                     [0,0,1]])

def rotate(rhat,theta,phi):
    """
    Transform a unit vector, rhat, to align with the z-axis. Perform rotations
    by theta radians about the y-axis and phi radians about the z-axis (in that 
    order). Inverse-transform the vector back to its original position. 
    Spherical polar coordinates are used here (rhat, theta, phi), with 
    anticlockwise rotations.
    """
    
    # Initial polar (alpha) and azimuthal (beta) angles of unit vector rhat
    alpha=np.arccos(rhat[2]/1.0)
    alpha*=np.sign(rhat[0])
    beta=np.arctan(rhat[1]/rhat[0])
        
    # Align to z-axis
    rhat=np.matmul(Rz(-beta),rhat)
    rhat=np.matmul(Ry(-alpha),rhat)
    
    # Rotate relative to transformed frame of reference
    rhat=np.matmul(Ry(theta),rhat)
    rhat=np.matmul(Rz(phi),rhat)
    
    # Return to original frame
    rhat=np.matmul(Ry(alpha),rhat)
    rhat=np.matmul(Rz(beta),rhat)
    
    return rhat

class Cell3D:

    def __init__(self, name, position, speed, direction, tumble_chance,
                 time_step):
        self.name = name
        self.swim_position = position
        self.brownian_position = np.copy(position)
        self.speed = speed
        self.direction = direction
        self.velocity = self.speed * self.direction
        self.tumble_chance = tumble_chance  # probability per time step
        self.run_duration = 0

        self.swim_history = []
        self.swim_history.append(np.copy(self.swim_position))
        self.brownian_history = []
        self.brownian_history.append(np.copy(self.brownian_position))
        self.combined_history = []
        self.combined_history.append(np.copy(self.swim_position)+np.copy(self.brownian_position))
        self.run_durations = []
        self.tumble_angles = []


    def run(self, time_step):
        """Move forwards in current direction."""
        self.velocity = self.speed * self.direction
        self.swim_position += (self.velocity * time_step)


    def tumble(self, tumble_mean, tumble_stddev):
        """
        Perform a tumble on the unit direction vector of the cell. Tumble
        angle (theta) is drawn from a normal distribution defined by the input
        arguments. Following a rotation by theta, the vector is revolved
        through an angle phi, drawn from a uniform random distribution within
        the range 0 <= phi < 2pi radians.
        
        Experimental data for E.coli suggests tumbles are biased in the forward
        direction, with a mean tumble angle of ~68 degrees [HC Berg, Random Walks
        in Biology, p86; 1993].
        """

        tumble_angle = np.random.normal(tumble_mean,tumble_stddev)
        rev_angle = np.random.uniform(0,2.0*np.pi)
        self.direction = rotate(self.direction,tumble_angle,rev_angle)


    def compute_step_size(D, dt):
        """Randomly determine the distance that the cell will step
        by drawing from a Gaussian distribution centred at zero, with
        a standard deviation that is proportional to the square root
        of time."""

        mean = 0
        std_dev = np.sqrt(2*D*dt)

        return np.random.normal(mean,std_dev)


    def trans_brownian_motion(self, diffusion_constant, time_step):
        """Thermal fluctuations in the x, y and z axes. Uses the Berg
        approach of having a 50% chance to move +/- a step in each
        axis. Add the Brownian steps to their own position history."""

        # independent steps in x, y and z
        dx = Cell3D.compute_step_size(diffusion_constant, time_step)
        dy = Cell3D.compute_step_size(diffusion_constant, time_step)
        dz = Cell3D.compute_step_size(diffusion_constant, time_step)

        self.brownian_position += np.array([dx,dy,dz])


    def rot_brownian_motion(self):
        """Under construction."""
        pass


    def update(self, diffusion_constant, time_step, max_tumble_angle):
        """Execute a run and attempt to tumble every timestep. Append data
        to arrays."""

        self.run(time_step)
        self.run_duration += time_step
        self.trans_brownian_motion(diffusion_constant, time_step)

        old_direction = self.direction

        if np.random.random() < self.tumble_chance:
            self.tumble(np.deg2rad(68),1.0)
            angle = np.arccos(np.dot(old_direction,self.direction))
            self.tumble_angles.append(angle)
            self.run_durations.append(self.run_duration)
            self.run_duration = 0

        self.brownian_history.append(np.copy(self.brownian_position))
        self.swim_history.append(np.copy(self.swim_position))
        self.combined_history.append(np.copy(self.brownian_position)+np.copy(self.swim_position))
