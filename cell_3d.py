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
        self.position = position
        self.brownian_position = position
        self.speed = speed
        self.direction = direction
        self.velocity = self.speed * self.direction
        self.tumble_chance = tumble_chance

        self.position_history = []
        self.position_history.append(np.copy(self.position))
        self.brownian_history = []
        self.brownian_history.append(np.copy(self.position))

    
    def run(self, step_size):
        """Move forwards in current direction"""
        self.velocity = self.speed * self.direction
        self.position = self.position + (self.velocity * step_size)

   
    def tumble(self, max_angle):
        """Rotate randomly about the x, y and z axes"""
        
        R_x = rotation_matrix_x(np.random.random()*max_angle)
        R_y = rotation_matrix_y(np.random.random()*max_angle)
        R_z = rotation_matrix_z(np.random.random()*max_angle)
        
        self.direction = np.matmul(self.direction, R_x)
        self.direction = np.matmul(self.direction, R_y)
        self.direction = np.matmul(self.direction, R_z)


    def trans_brownian_motion(self):
        """Thermal fluctuations in the x, y and z axes. Uses the Berg
        approach of having a 50% chance to move by +/- a step in each
        axis. Add the Brownian steps to their own position history."""

        #self.brownian_position = self.brownian_position + 
        pass

    def rot_brownian_motion(self):
        pass

    def update(self, step_size, max_angle):
        """Execute a run and attempt to tumble every timestep. Append data
        to arrays."""
         
        #self.run(step_size)

        #Run-only
        #if np.random.random() > self.tumble_chance:
        #    self.tumble(max_angle)

        self.position_history.append(np.copy(self.position))
        self.brownian_history.append(np.copy(self.brownian_position))
