import matplotlib.pyplot as plt
import numpy as np

class System:
    
    max_time = 100
    step_size = 1
    total_steps = int(max_time / step_size)
    timesteps = np.linspace(0,max_time,num=total_steps)  # seconds
    
    np.random.seed(100)
    pi = np.pi
    
    box_size = 400  # micrometres

    
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
        self.position = self.position + (self.velocity * System.step_size)
    
    def tumble(self, max_angle):
        """Rotate randomly about the x, y and z axes"""
        
        R_x = rotation_matrix_x(np.random.random()*max_angle)
        R_y = rotation_matrix_y(np.random.random()*max_angle)
        R_z = rotation_matrix_z(np.random.random()*max_angle)
        
        self.direction = np.matmul(self.direction, R_x)
        self.direction = np.matmul(self.direction, R_y)
        self.direction = np.matmul(self.direction, R_z)
        
    
    def switch_mode(self):
        """Attempt to switch from run to tumble mode, and vice-versa"""
        
        random_float = np.random.random()
        if self.run_mode==True:
            if random_float > self.run_chance:
                self.run_mode = False
                
        elif self.run_mode==False:
            if random_float > self.tumble_chance:
                self.run_mode = True
    
    def update(self):
        """Attempt to switch movement mode, then execute said mode"""
        
        self.switch_mode()
        
        if self.run_mode == True:
            self.run()
        elif self.run_mode == False:
            self.tumble(2*System.pi)
    
    
    
swimmer = Cell(name='Escherichia coli', position=np.array([0,0,0]), speed=20,
               direction=np.array([1,0,0]), run_chance=0.3, tumble_chance=0.1,
               run_mode=True)

positions = []

for time in System.timesteps:
    
    swimmer.update()    
    positions.append(np.copy(swimmer.position))
 

positions = np.array(positions)

# plot x,y
plt.plot(positions[:,0],positions[:,1])
plt.xlabel('x ($\mu$m)')
plt.ylabel('y ($\mu$m)')
plt.xlim(-0.5*System.box_size,0.5*System.box_size)
plt.ylim(-0.5*System.box_size,0.5*System.box_size)
plt.plot([-0.5*System.box_size,0.5*System.box_size],[0,0],color='k',ls='--',
         lw=0.5)
plt.plot([0,0],[-0.5*System.box_size,0.5*System.box_size],color='k',ls='--',
         lw=0.5)
plt.savefig('xy_run_tumble.png')
