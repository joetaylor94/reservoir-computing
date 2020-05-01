'''
Joe Taylor (j.taylor@bath.ac.uk)
Department of Physics
University of Bath, UK
May 1st, 2020


Model of the 1963 Lorenz system for use with reservoir computing.

The model is a three-dimensional system of ordinary differential equations 
now known as the Lorenz equations:
    
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

System parameters p = [sigma, rho, beta] are chosen to be p =[10, 8/3, 28]. 
The system exhibits chaotic behavior for these values.

Values for all three state variables are output by the script.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define parameters
SIGMA = 10
RHO = 28
BETA = 8/3

# Model duration
T = 50
dt = 0.01

# Generate array of time points, from zero to T
t = np.arange(0,T,dt)

##############################################################################
# Model Equations of Motion
##############################################################################

# Define equations of motion for full system state X = [x,y,z]
# Function reads in system state and returns its derivative
def dXdt(X,t):
    x, y, z = X
    dxdt = SIGMA * (y - x)
    dydt = x * (RHO - z) - y
    dzdt = x * y - BETA * z
    return dxdt, dydt, dzdt


##############################################################################
# Initializing the Lorenz model
##############################################################################

# Initialize state variable values for 
# t=0: X(0) = [x(0),y(0),z(0)]
init = [0.1000,0.1,0.1]

##############################################################################
# Running model: forward-integrating the equations of motion
##############################################################################
        
# Integrate model equations
# Arguments: state derivative, initial neuron state x(0), time point array
X = odeint(dXdt, init, t)

# Define variables to simplify analysis
x = X[:,0]
y = X[:,1]
z = X[:,2]

# Adding Gaussian error to outputs trace (mV)
sigma_obs = 0.0
x_obs = x + np.random.normal(0, sigma_obs, len(x))
y_obs = y + np.random.normal(0, sigma_obs, len(y))
z_obs = z + np.random.normal(0, sigma_obs, len(z))

##############################################################################
# Plotting and saving model output
##############################################################################


# Plotting state variable time series
plt.figure()
plt.subplot(3,1,1)
plt.ylabel("x")
plt.plot(t,x_obs,'k',linewidth=0.8)
plt.subplot(3,1,2)
plt.ylabel("y")
plt.plot(t,y_obs,'k',linewidth=0.8)  
plt.subplot(3,1,3)
plt.ylabel("z")
plt.plot(t,z_obs,'k',linewidth=0.8)  
plt.show()

# Plotting 3D system trajectory
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x,y,z, 'b', linewidth=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

# Save time series data 
f = open('output/x_data.csv', 'w')
for i in range(int(len(x_obs))):
    f.write('%f \n' % x_obs[i])
f.close()

f = open('output/y_data.csv', 'w')
for i in range(int(len(y_obs))):
    f.write('%f \n' % y_obs[i])
f.close()

f = open('output/z_data.csv', 'w')
for i in range(int(len(z_obs))):
    f.write('%f\n'% z_obs[i])
f.close()

f = open('output/xyz_data.csv', 'w')
for i in range(int(len(z_obs))):
    f.write('%f\t%f\t%f\n' % (x_obs[i],y_obs[i],z_obs[i]))
f.close()


