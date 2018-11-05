import numpy as np

def rotation_matrix_2D(angle):
    sin = np.sin(angle)
    cos = np.cos(angle)
    
    return np.array([[cos,-sin],
                     [sin,cos]])

class Cell2D:
    
    def __init__(self, name, position, speed, direction, tumble_chance):
        self.name = name
        self.position = position
        self.speed = speed
        self.direction = direction
        self.velocity = self.speed * self.direction
        self.tumble_chance = tumble_chance

        self.position_history = []

    
    def run(self, step_size):
        """Move forwards in current direction"""
        self.velocity = self.speed * self.direction
        self.position = self.position + (self.velocity * step_size)


    def tumble(self, max_angle):
        """Rotate randomly about the x and y axes"""

        R = rotation_matrix_2D(np.random.random()*max_angle)
        self.direction = np.matmul(self.direction, R)
                

    def update(self, step_size, max_angle):
        """Execute a run and attempt to tumble every timestep. Append data
        to arrays."""
         
        self.run(step_size)

        if np.random.random() > self.tumble_chance:
            self.tumble(max_angle)

        self.position_history.append(np.copy(self.position))
