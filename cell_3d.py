"""
RandomWalks - A code to simulate run-and-tumble swimming and Brownian motion
    
Copyright (C) 2019  R.C.T.B. Cocking

Email: rctc500@york.ac.uk

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np

# Cartesian rotation matrices
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
    
    Transformations are performed on the unit vector, u (a copy of rhat).
    """
    
    # Initial polar (alpha) and azimuthal (beta) angles of unit vector rhat
    alpha=np.arccos(rhat[2]/1.0)
    alpha*=np.sign(rhat[0])
    beta=np.arctan(rhat[1]/rhat[0])
        
    # Align to z-axis
    u=np.copy(rhat)
    u=np.matmul(Rz(-beta),u)
    u=np.matmul(Ry(-alpha),u)
    
    # Rotate relative to transformed frame of reference
    u=np.matmul(Ry(theta),u)
    u=np.matmul(Rz(phi),u)
    
    # Return to original frame
    u=np.matmul(Ry(alpha),u)
    u=np.matmul(Rz(beta),u)
    
    return u

class Cell3D:

    def __init__(self, name, position, speed, direction, tumble_chance,
                 time_step):
        """
        Initial cell values
        """
        
        self.name = name
        self.swim_position = position
        self.brownian_position = np.copy(position)
        self.speed = speed
        self.direction = direction  # direction unit vector, rhat
        self.velocity = self.speed * self.direction
        self.tumble_chance = tumble_chance  # probability per time step
        self.run_duration = 0
        self.ang_disp = 0.0  # angular deviation (displacement)
        self.rbm_ang_disp = 0.0

        self.swim_history = []
        self.swim_history.append(np.copy(self.swim_position))  # run & tumble
        self.brownian_history = []
        self.brownian_history.append(np.copy(self.brownian_position))  # translational brownian motion
        self.combined_history = []
        self.combined_history.append(np.copy(self.swim_position)+np.copy(self.brownian_position))
        self.rbm_ang_disp_history = [0.0]  # angular deviation due to rotational brownian motion
        self.ang_disp_history = [0.0]  # ang. dev. due to both RBM and tumbles
        self.run_durations = []
        self.tumble_angles = []
        self.direction_history = []
        self.direction_history.append(np.copy(self.direction))


    def run(self, time_step):
        """
        Move forwards in current direction and increment the duration
        of the run.
        """
        
        self.velocity = self.speed * self.direction
        self.swim_position += (self.velocity * time_step)
        self.run_duration += time_step
        

    def tumble(self, tumble_mean, tumble_stddev):
        """
        Perform a tumble on the direction unit vector of the cell. Tumble
        angle (theta_t) is drawn from a normal distribution defined by the input
        arguments. 
        
        Following a rotation by theta_t, the unit vector is revolved into 3D
        space through an angle phi, drawn from a uniform random distribution
        such that 0 <= phi < 2pi radians.

        Experimental data for E.coli suggests tumbles are biased in the forward
        direction, with a mean tumble angle of ~68 degrees [HC Berg, Random Walks
        in Biology, p86; 1993].
        """

        theta_t = np.random.normal(loc=tumble_mean, scale=tumble_stddev, size=None)
        phi = np.random.uniform(0, 2.0*np.pi)
        self.direction = rotate(self.direction, theta_t, phi)
        
        return theta_t

    def draw_brownian_step(dim, D, dt):
        """
        Return a value drawn from a normal distribution centred at zero,
        with variance proportional to dim*2*D*dt. dim is an integer that
        corresponds to the number of axes contributing to the step (1, 2 or 3).
        D is the diffusion coefficient of translational or rotational Brownian
        motion.
        """

        return np.random.normal(loc=0, scale=np.sqrt(dim*2.0*D*dt), size=None)

    def trans_brownian_motion(self, diffusion_constant, time_step):
        """
        Thermal fluctuations in the x, y and z axes. Cell has a 50% 
        chance to move +/- a step in each axis.
        """

        # independent steps in x, y and z
        dx = Cell3D.draw_brownian_step(1, diffusion_constant, time_step)
        dy = Cell3D.draw_brownian_step(1, diffusion_constant, time_step)
        dz = Cell3D.draw_brownian_step(1, diffusion_constant, time_step)

        self.brownian_position += np.array([dx,dy,dz])


    def rot_brownian_motion(self, rot_diffusion_constant, time_step):
        """
        Deviate by some angle, theta_rbm, according to rotational Brownian
        motion, relative to the direction unit vector of the cell. 
        
        theta_rbm is drawn from a normal distribution centred at zero with a
        variance proportional to 4*D_r*dt, where D_r is the rotational diffusion
        constant and dt is the simulation time step. 
        
        Following a rotation by theta_rbm, the unit vector is revolved into 3D
        space through an angle phi, drawn from a uniform random distribution
        such that 0 <= phi < 2pi radians.
        """

        theta_rbm = Cell3D.draw_brownian_step(2, rot_diffusion_constant, time_step)
        phi = np.random.uniform(0, 2.0*np.pi)
        self.direction = rotate(self.direction, theta_rbm, phi)

        return theta_rbm

    def update(self, diffusion_constant, rot_diffusion_constant, time_step,
               enable_run=True, enable_tumble=True, enable_tbm=True,
               enable_rbm=True):
        """
        Called once per time step:
             1) Run forwards in the current direction
             2) Move via translational Brownian motion
             3) Rotate via rotational Brownian motion
             4) If dice roll is successful, rotate via tumbling
             5) Record cell state
        """
        
        if enable_run:
            self.run(time_step)
            
        if enable_tbm:
            self.trans_brownian_motion(diffusion_constant, time_step)        
            
        if enable_rbm:
            rbm_angle = self.rot_brownian_motion(rot_diffusion_constant, time_step)
        else:
            rbm_angle = 0.0
            
        self.rbm_ang_disp += rbm_angle
        angle = rbm_angle

        # Perform tumble if dice roll successful
        if (np.random.random() < self.tumble_chance) and (enable_tumble):
            angle = self.tumble(np.deg2rad(68),np.deg2rad(36))  # distribution from Berg & Brown 1972
            self.tumble_angles.append(angle)
            self.run_durations.append(self.run_duration)
            self.run_duration = 0

        self.ang_disp += angle
        
        # Append data to lists
        self.brownian_history.append(np.copy(self.brownian_position))  # TBM
        self.swim_history.append(np.copy(self.swim_position))  # Runs
        self.combined_history.append(np.copy(self.brownian_position)+np.copy(self.swim_position))  # TBM and runs
        self.rbm_ang_disp_history.append(self.rbm_ang_disp)  # RBM
        self.ang_disp_history.append(self.ang_disp)  # RBM and tumbles
        self.direction_history.append(np.copy(self.direction))  # rhat
