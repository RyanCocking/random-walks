import numpy as np

def rotation_matrix_x(angle):
    sin = np.sin(angle)
    cos = np.cos(angle)

    return np.array([[1,0,0],
                     [0,cos,-sin],
                     [0,sin,cos]])

def rotation_matrix_y(angle):
    sin = np.sin(angle)
    cos = np.cos(angle)

    return np.array([[cos,0,sin],
                     [0,1,0],
                     [-sin,0,cos]])

def rotation_matrix_z(angle):
    sin = np.sin(angle)
    cos = np.cos(angle)

    return np.array([[cos,-sin,0],
                     [sin,cos,0],
                     [0,0,1]])

class Cell3D:

    def __init__(self, name, position, speed, direction, tumble_chance_per_sec,
                 time_step):
        self.name = name
        self.swim_position = position
        self.brownian_position = np.copy(position)
        self.speed = speed
        self.direction = direction
        self.velocity = self.speed * self.direction
        self.tumble_chance = tumble_chance_per_sec*time_step
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


    def tumble(self, max_angle):
        """Randomly and independently rotate about the x, y and z axes 
        between 0 and some maximum angle."""

        #construct rotation matrices
        R_x = rotation_matrix_x(np.random.random()*max_angle)
        R_y = rotation_matrix_y(np.random.random()*max_angle)
        R_z = rotation_matrix_z(np.random.random()*max_angle)

        #perform rotation
        self.direction = np.matmul(self.direction, R_x)
        self.direction = np.matmul(self.direction, R_y)
        self.direction = np.matmul(self.direction, R_z)


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
            self.tumble(max_tumble_angle)
            angle = np.arccos(np.dot(old_direction,self.direction))
            self.tumble_angles.append(angle)
            self.run_durations.append(self.run_duration)
            self.run_duration = 0

        self.brownian_history.append(np.copy(self.brownian_position))
        self.swim_history.append(np.copy(self.swim_position))
        self.combined_history.append(np.copy(self.brownian_position)+np.copy(self.swim_position))
