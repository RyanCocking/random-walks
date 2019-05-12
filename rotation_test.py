# NOT PART OF OFFICIAL CODE

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def inv_dot(a,b):
    maga=np.linalg.norm(a)
    magb=np.linalg.norm(b)
    return (1.0/(maga*magb))*np.arccos(np.dot(a,b))

#Setup 3d plot

ax3d = plt.figure().add_subplot(111, projection='3d')
ax3d.set_xlabel('x ($\mu$m)')
ax3d.set_ylabel('y ($\mu$m)')
ax3d.set_zlabel('z ($\mu$m)')

ax3d.plot([0,.1],[0,0],[0,0],'-',lw=1,color='k')
ax3d.plot([0,0],[0,.1],[0,0],'-',lw=1,color='k')
ax3d.plot([0,0],[0,0],[0,.1],'-',lw=1,color='k')

plt.tight_layout()

vec=np.array([1,1,1])
mag=np.linalg.norm(vec)
rhat=vec/mag
rhat_o=rhat

x=np.array([1.0,0.0,0.0])
y=np.array([0.0,1.0,0.0])
z=np.array([0.0,0.0,1.0])

# Original unit vector
ax3d.plot([0,rhat[0]],[0,rhat[1]],[0,rhat[2]],'-',lw=3,color='r')
ax3d.plot([rhat[0],rhat[0]],[rhat[1],rhat[1]],[rhat[2],0],'--',lw=2,color='r')
ax3d.plot([rhat[0],0],[rhat[1],0],[0,0],'--',lw=2,color='r')

print("Rhat1 = ",rhat)

alpha=np.arccos(rhat[2]/1.0)
alpha*=np.sign(rhat[0])
beta=np.arctan(rhat[1]/rhat[0])

print("Alpha1 = ",np.rad2deg(alpha))
print("Beta1 = ",np.rad2deg(beta))
print(" ")

# First transform: -beta about z
rhat=np.matmul(Rz(-beta),rhat)
ax3d.plot([0,rhat[0]],[0,rhat[1]],[0,rhat[2]],'-',lw=3,color='b')
ax3d.plot([rhat[0],rhat[0]],[rhat[1],rhat[1]],[rhat[2],0],'--',lw=2,color='b')
ax3d.plot([rhat[0],0],[rhat[1],0],[0,0],'--',lw=2,color='b')

# Second transform: -alpha about y
rhat=np.matmul(Ry(-alpha),rhat)
ax3d.plot([0,rhat[0]],[0,rhat[1]],[0,rhat[2]],'-',lw=3,color='g')
ax3d.plot([rhat[0],rhat[0]],[rhat[1],rhat[1]],[rhat[2],0],'--',lw=2,color='g')
ax3d.plot([rhat[0],0],[rhat[1],0],[0,0],'--',lw=2,color='g')

print("Rhat3 = ",rhat)
print(" ")

# TUMBLING
tumble=np.deg2rad(10)
rev=np.random.uniform(0,2*3.14159)
print(np.rad2deg(rev))
# Aligned vector, tumbled and revolved
rhat=np.matmul(Ry(tumble),rhat)
rhat=np.matmul(Rz(rev),rhat)
ax3d.plot([0,rhat[0]],[0,rhat[1]],[0,rhat[2]],'-.',lw=3,color='g')
ax3d.plot([rhat[0],rhat[0]],[rhat[1],rhat[1]],[rhat[2],0],'--',lw=2,color='g')
ax3d.plot([rhat[0],0],[rhat[1],0],[0,0],'--',lw=2,color='g')

# First back-transform: +alpha about y
rhat=np.matmul(Ry(alpha),rhat)
ax3d.plot([0,rhat[0]],[0,rhat[1]],[0,rhat[2]],'-.',lw=3,color='b')
ax3d.plot([rhat[0],rhat[0]],[rhat[1],rhat[1]],[rhat[2],0],'--',lw=2,color='b')
ax3d.plot([rhat[0],0],[rhat[1],0],[0,0],'--',lw=2,color='b')

# Second back-transform: +beta about z
rhat=np.matmul(Rz(beta),rhat)
ax3d.plot([0,rhat[0]],[0,rhat[1]],[0,rhat[2]],'-.',lw=3,color='r')
ax3d.plot([rhat[0],rhat[0]],[rhat[1],rhat[1]],[rhat[2],0],'--',lw=2,color='r')
ax3d.plot([rhat[0],0],[rhat[1],0],[0,0],'--',lw=2,color='r')

print("Tumble angle = ",np.rad2deg(tumble))
print("Dot angle = ",np.rad2deg(np.arccos(np.dot(rhat,rhat_o))))

ax3d.view_init(elev=50,azim=20)
plt.savefig('Rotation.png')
plt.show()
