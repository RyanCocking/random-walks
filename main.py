import matplotlib.pyplot as plt
import numpy as np

class System:
    
    max_time = 100
    step_size = 1
    total_steps = int(max_time / step_size)
    timesteps = np.linspace(0,max_time,num=total_steps)
    
    np.random.seed(100)
    pi = np.pi

    
def rotation_matrix_x(angle):
    pass

def rotation_matrix_y(angle):
    pass
    
def rotation_matrix_z(angle):
    pass

    
class Cell:
    
    def __init__(self, name, position, speed, direction, run_chance,
                 tumble_chance, run_mode):
        self.name = name
        self.position = position
        self.speed = speed
        self.direction = direction
        self.velocity = self.speed * self.direction
        self.run_chance = run_chance
        self.tumble_chance = tumble_chance
        self.run_mode = run_mode
    
    
    def run(self):
        """Move forwards in current direction"""
        self.velocity = self.speed * self.direction
        self.position += self.velocity
    
    def tumble(self, max_angle):
        """Rotate randomly about the x, y and z axes"""
        pass
    
    def switch_mode(self):
        """Attempt to switch from run to tumble mode, or vice-versa"""
        
        random_float = np.random.random()
        if self.run_mode==True:
            if random_float > self.run_chance:
                self.run_mode = False
                
        else:
            if random_float > self.tumble_chance:
                self.run_mode = True
    
    def update(self):
        """Attempt to switch movement mode, then execute said mode"""
        
        self.switch_mode()
        
        if self.run_mode == True:
            self.run()
        else:
            self.tumble(2*System.pi)
    
    
    
swimmer = Cell(name='Escherichia coli', position=np.array([0,0,0]), speed=20,
               direction=np.array([1,0,0]), run_chance=0.3, tumble_chance=0.1,
               run_mode=True)

positions = []

for time in System.timesteps:
    
    swimmer.update()    
    positions.append(np.copy(swimmer.position))
 

positions = np.array(positions)
plt.plot(positions[:,0],positions[:,1])
plt.xlabel('x ($\mu$m)')
plt.ylabel('y ($\mu$m)')
plt.show()
