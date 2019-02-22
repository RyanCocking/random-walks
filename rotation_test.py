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

def todeg(angle):
    return angle*180.0/3.14159

#Setup 3d plot

ax3d = plt.figure().add_subplot(111, projection='3d')
ax3d.set_xlabel('x ($\mu$m)')
ax3d.set_ylabel('y ($\mu$m)')
ax3d.set_zlabel('z ($\mu$m)')

ax3d.plot([0,.1],[0,0],[0,0],'-',lw=1,color='k')
ax3d.plot([0,0],[0,.1],[0,0],'-',lw=1,color='k')
ax3d.plot([0,0],[0,0],[0,.1],'-',lw=1,color='k')

plt.tight_layout()
plt.title('Final transformation')

vec=np.array([1.0,1.0,1.0])
mag=np.linalg.norm(vec)
rhat=vec/mag

x=np.array([1.0,0.0,0.0])
y=np.array([0.0,1.0,0.0])
z=np.array([0.0,0.0,1.0])

ax3d.plot([0,rhat[0]],[0,rhat[1]],[0,rhat[2]],'-',lw=3,color='r')
ax3d.plot([rhat[0],rhat[0]],[rhat[1],rhat[1]],[rhat[2],0],'--',lw=2,color='r')
ax3d.plot([rhat[0],0],[rhat[1],0],[0,0],'--',lw=2,color='r')

print("Pre-transform")
print("Rhat = ",rhat)

alpha=np.arccos(rhat[2]/1.0)
beta=np.arctan(rhat[1]/rhat[0])

print("Alpha = ",todeg(alpha))
print("Beta = ",todeg(beta))
print(" ")

rhat=np.matmul(rhat,Rz(-beta))

ax3d.plot([0,rhat[0]],[0,rhat[1]],[0,rhat[2]],'-',lw=3,color='b')
ax3d.plot([rhat[0],rhat[0]],[rhat[1],rhat[1]],[rhat[2],0],'--',lw=2,color='b')
ax3d.plot([rhat[0],0],[rhat[1],0],[0,0],'--',lw=2,color='b')

rhat=np.matmul(rhat,Rx(-alpha))

ax3d.plot([0,rhat[0]],[0,rhat[1]],[0,rhat[2]],'-',lw=3,color='g')
ax3d.plot([rhat[0],rhat[0]],[rhat[1],rhat[1]],[rhat[2],0],'--',lw=2,color='g')
ax3d.plot([rhat[0],0],[rhat[1],0],[0,0],'--',lw=2,color='g')

plt.show()
