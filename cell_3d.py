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
    
    def __init__(self, name, position, speed, direction, tumble_chance):
        self.name = name
        self.swim_position = position
        self.brownian_position = np.copy(position)
        self.speed = speed
        self.direction = direction
        self.velocity = self.speed * self.direction
        self.tumble_chance = tumble_chance

        self.swim_history = []
        self.swim_history.append(np.copy(self.swim_position))
        self.brownian_history = []
        self.brownian_history.append(np.copy(self.brownian_position))

    
    def run(self, time_step):
        """Move forwards in current direction."""
        self.velocity = self.speed * self.direction
        self.swim_position += (self.velocity * time_step)

   
    def tumble(self, max_angle):
        """Rotate randomly about the x, y and z axesby some input angle."""
        
        R_x = rotation_matrix_x(np.random.random()*max_angle)
        R_y = rotation_matrix_y(np.random.random()*max_angle)
        R_z = rotation_matrix_z(np.random.random()*max_angle)
        
        self.direction = np.matmul(self.direction, R_x)
        self.direction = np.matmul(self.direction, R_y)
        self.direction = np.matmul(self.direction, R_z)


    def trans_brownian_motion(self, bm_step):
        """Thermal fluctuations in the x, y and z axes. Uses the Berg
        approach of having a 50% chance to move +/- a step in each
        axis. Add the Brownian steps to their own position history."""

        rng_x = np.random.random()
        rng_y = np.random.random()
        rng_z = np.random.random()

        displacement = np.array([bm_step, bm_step, bm_step])

        # swap signs if random numbers exceed 0.5
        if rng_x > 0.5:
            displacement[0] = -displacement[0]

        if rng_y > 0.5:
            displacement[1] = -displacement[1]

        if rng_z > 0.5:
            displacement[2] = -displacement[2]
        
        self.brownian_position += displacement


    def rot_brownian_motion(self):
        """Under construction."""
        pass

    def update(self, bm_step, time_step, max_angle):
        """Execute a run and attempt to tumble every timestep. Append data
        to arrays."""

        self.trans_brownian_motion(bm_step)
        #self.run(time_step)

        #Run-only
        #if np.random.random() > self.tumble_chance:
        #    self.tumble(max_angle)

        self.brownian_history.append(np.copy(self.brownian_position))
        self.swim_history.append(np.copy(self.swim_position))
