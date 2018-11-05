import numpy as np

class Cell1D:
    
    def __init__(self, name, position, speed, direction, run_chance,
                 tumble_chance):
        self.name = name
        self.position = position
        self.speed = speed
        self.direction = direction
        self.velocity = self.speed * self.direction
        self.run_chance = run_chance
        self.tumble_chance = tumble_chance
        self.run_mode = True

        self.position_history = []
        self.mode_history = np.array([],dtype=bool)
    
    def edge_check(self, box_size, step_size):
        """Check if a cell has wandered to the edge of the box. If it has,
        reverse its direction to simulate an elastic collision with the box
        walls."""

        edge = 0.5*box_size - (self.speed * step_size)
    
        if abs(self.position[0]) >= edge:
            self.direction[0] = -self.direction[0]
                                                   
        if abs(self.position[1]) >= edge:
            self.direction[1] = -self.direction[1]

        if abs(self.position[2]) >= edge:
            self.direction[2] = -self.direction[2]

    
    def run(self, step_size):
        """Move forwards in current direction"""
        self.velocity = self.speed * self.direction
        self.position = self.position + (self.velocity * step_size)


    def tumble(self, max_angle, x_matrix, y_matrix, z_matrix):
        """Rotate randomly about the x, y and z axes"""
        
        R_x = x_matrix(np.random.random()*max_angle)
        R_y = y_matrix(np.random.random()*max_angle)
        R_z = z_matrix(np.random.random()*max_angle)
        
        self.direction = np.matmul(self.direction, R_x)
        self.direction = np.matmul(self.direction, R_y)
        self.direction = np.matmul(self.direction, R_z)
        
    
    def switch_mode(self):
        """Attempt to switch from run to tumble mode, and vice-versa"""
        
        random_float = np.random.random()
        if self.run_mode==True:
            if (random_float > self.run_chance) and (self.tumble_chance > 0):
                self.run_mode = False
                

        elif self.run_mode==False:
            if (random_float > self.tumble_chance) and (self.run_chance > 0):
                self.run_mode = True
               


    def update(self, step_size, max_angle, x_matrix, y_matrix, z_matrix):
        """Attempt to switch movement mode then execute said mode. Append data
        to arrays."""
        
        switch_success = self.switch_mode()

        if self.run_mode == True:
            #self.edge_check(System.box_size, System.step_size)
            self.run(step_size)
        elif self.run_mode == False:
            self.tumble(max_angle, x_matrix, y_matrix, z_matrix)

        self.position_history.append(np.copy(self.position))
        self.mode_history = np.append(self.mode_history,self.run_mode)
    
