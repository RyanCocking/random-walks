import matplotlib.pyplot as plt
import numpy as np

class System:
    
    max_time = 100000
    step_size = 1
    total_steps = int(max_time / step_size)
    timesteps = np.linspace(0,max_time,num=total_steps)  # seconds
    
    seed = 100
    np.random.seed(seed)
    pi = np.pi
    
    box_size = 1000  # micrometres

    
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
    
    def edge_check(self):
        """Check if a cell has wandered to the edge of the box. If it has,
        reverse its direction to simulate an elastic collision with the box
        walls."""

        edge = 0.5*System.box_size - (self.speed * System.step_size)
    
        if abs(self.position[0]) >= edge:
            self.direction[0] = -self.direction[0]
                                                   
        if abs(self.position[1]) >= edge:
            self.direction[1] = -self.direction[1]

        if abs(self.position[2]) >= edge:
            self.direction[2] = -self.direction[2]

    
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
            if (random_float > self.run_chance) and (self.tumble_chance > 0):
                self.run_mode = False
                
        elif self.run_mode==False:
            if (random_float > self.tumble_chance) and (self.run_chance > 0):
                self.run_mode = True
    
    def update(self):
        """Attempt to switch movement mode, then execute said mode"""
        
        self.switch_mode()
        
        if self.run_mode == True:
            #self.edge_check()
            self.run()
        elif self.run_mode == False:
            self.tumble(2*System.pi)
    

    
swimmer = Cell(name='Escherichia coli', position=np.array([0,0,0]), speed=20,
               direction=np.array([1,1,0]), run_chance=0.3, tumble_chance=0.05,
               run_mode=True)

positions = []

for time in System.timesteps:
    
    swimmer.update()    
    positions.append(np.copy(swimmer.position))
 

positions = np.array(positions)
x_pos = positions[:,0]
y_pos = positions[:,1]
z_pos = positions[:,2]
displacement = np.array(np.linalg.norm(positions,ord=None,axis=1))

# plot x,y
plt.plot(x_pos,y_pos)
plt.xlabel('x ($\mu$m)')
plt.ylabel('y ($\mu$m)')
plt.xlim(-System.box_size,System.box_size)
plt.ylim(-System.box_size,System.box_size)
plt.plot([-0.5*System.box_size,0.5*System.box_size],[0,0],color='k',ls='--',
         lw=0.5)
plt.plot([0,0],[-0.5*System.box_size,0.5*System.box_size],color='k',ls='--',
         lw=0.5)
plt.savefig('xy_run_tumble.png')
plt.close()

# plot t,<r^2>
plt.plot(System.timesteps,displacement)
plt.ylabel('|r| ($\mu$m)')
plt.xlabel('t (s)')
plt.xlim(0,System.max_time)
plt.savefig('norm_r_vs_t.png')
plt.close()
# plot x,y,z distributions
plt.hist(np.unique(x_pos), bins='auto', density=False)
#plt.plot([0,0],[0,1],lw=0.5,ls='--',color='k')
plt.savefig('x_hist.png')
plt.close()
plt.hist(np.unique(y_pos), bins='auto', density=False)
#plt.plot([0,0],[0,1],lw=0.5,ls='--',color='k')
plt.savefig('y_hist.png')
plt.close()
plt.hist(np.unique(z_pos), bins='auto', density=False)
#plt.plot([0,0],[0,1],lw=0.5,ls='--',color='k')
plt.savefig('z_hist.png')
plt.close()
