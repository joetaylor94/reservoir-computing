'''
Joe Taylor (j.taylor@bath.ac.uk)
Department of Physics
University of Bath, UK
May 1st, 2020


Conductance model of an RVLM neuron for use with reservoir computing
using a modified Hodgkin-Huxley framework of ion channel gating.

Model parameters are chosen so as to replicate the behaviour of
the thalamocortical relay neuron presented in Huguenard J, McCormick DA, 
Shepherd GM (1997) 'Electrophysiology of the Neuron'.

The neuron model consists of three ionic currents: a passive leak current,
a transient sodium current (NaT), and a potassium current (K). The sodium
current is controlled by an activation gating variable (m) and an 
inactivation gating variable (h). The potassium channel is non-inactivating
and is controlld by a single activation gating variable (n).

The full model state x comprises four state variables - the membrane voltage
and the three gating varibales m, h, and n, and is thus described as:

   x = [V,m,h,n]

The only state variable that it is possible to measure experimentally is the 
membrane voltage. This is the state variable output by the python script.
'''

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define constants
TEMP_C = 35
FARADAY = 96480
PI = 3.14159265359

# Model duration (ms)
T = 7400
dt = 0.025

# Generate array of time points, from zero to T
t = np.arange(0,T,dt)

##############################################################################
# Model Equations of Motion
##############################################################################

# Define functions for gating kinetics of ion channels
# Effect of temperature is accounted for by the Q10 coeff
def mm_inf(VV): return 0.5*(1 + sp.tanh((VV - amV1)/ amV2))
def mm_tau(VV): return (tm0 + epsm*(1 - sp.tanh((VV - amV1)/ amV3)*sp.tanh((VV - amV1)/ amV3))) / 3.0**((TEMP_C-23.5)/10)
def hh_inf(VV): return 0.5*(1 + sp.tanh((VV - ahV1)/ ahV2))
def hh_tau(VV): return (th0 + epsh*(1 - sp.tanh((VV - ahV1)/ ahV3)*sp.tanh((VV - ahV1)/ ahV3))) / 3.0**((TEMP_C-23.5)/10)
def nn_inf(VV): return 0.5*(1 + sp.tanh((VV - anV1)/ anV2))
def nn_tau(VV): return (tn0 + epsn*(1 - sp.tanh((VV - anV1)/ anV3)*sp.tanh((VV - anV1)/ anV3))) / 3.0**((TEMP_C-23.5)/10)

# Define functions for ionic currents (in uA/cm^2)
# Currents correspond to passive leak, delayed-rectifier potassium,
# and transient sodium currents
def I_Leak(VV): return gLeak * (VV - EL)
def I_K(VV,nn): return gK * nn**4 * (VV - EK)
def I_NaT(VV,mm,hh): return gNaT * mm**3 * hh * (VV - ENa)

# Define equations of motion for full neuron state x = [V,m,h,n]
# Use idx to read in correct current stimulation data point
# Function reads in system state and returns its derivative
def dXdt(X,t):
    VV, mm, hh, nn, idx = X
    soma_area = soma_len*soma_diam*PI
    idx = int(t/dt)
    dVVdt = (-(I_NaT(VV,mm,hh) + I_K(VV,nn) + I_Leak(VV)) + (i_inj(t) + stim[idx])/soma_area) / Cm
    dmmdt = (mm_inf(VV) - mm)/mm_tau(VV)
    dhhdt = (hh_inf(VV) - hh)/hh_tau(VV)
    dnndt = (nn_inf(VV) - nn)/nn_tau(VV)
    return dVVdt, dmmdt, dhhdt, dnndt, idx

##############################################################################
# Model Parameters
##############################################################################
    
# Soma dimensions (cm)
soma_len = 0.01       
soma_diam = 0.029/PI 

# Define model parameters
# conductances: gX; reversal potentials: EX;
# thresholds: aXV1; membrane capacitance: Cm;
# time constants: tx0, epsx
Cm      = 1       
gNaT    = 69       
ENa     = 41
gK      = 6.9
EK      = -100
EL      = -65
gLeak   = 0.465
amV1    = -39.92
amV2    = 10
amV3    = 23.39
tm0     = 0.143
epsm    = 1.099
ahV1    = -65.37
ahV2    = -17.65
ahV3    = 27.22
th0     = 0.701
epsh    = 12.90
anV1    = -34.58
anV2    = 22.17
anV3    = 23.58
tn0     = 1.291
epsn    = 4.314

##############################################################################
# Preparing current stimulation to be injected into the neuron
##############################################################################

# Function for injected a current step (uA/cm^2)
# Args: amplitude, init time, final time
def i_inj(t):
    return amp*(t>t_i) - amp*(t>t_f)

# Function for loading current injection protocol (uA/cm^2)
# Args: file path, amplitude scale (default = 0.02), sample every 'n'th point
def load_stim(name, scale, n):
    stim = []
    with open(name, "r") as ins:
        count = 0
        for line in ins:
            count+=1
            if count % n == 0:
                stim.append(scale*(float(line.rstrip('\n'))))
        ins.close()
    return stim
            
# Initialise stim or load external stimulation files
# If not loading in external stim, uncomment line below
#stim = np.zeros(int(2*T/dt))
stim = load_stim('stim_files/Pstandard_100khz_0.dat', 0.02, 20)
stim += load_stim('stim_files/Pstandard_100khz_1.dat', 0.02, 20)
stim += load_stim('stim_files/Pstandard_100khz_2.dat', 0.02, 20)
stim += load_stim('stim_files/Pstandard_100khz_3.dat', 0.02, 20)
stim += load_stim('stim_files/Pstandard_100khz_4.dat', 0.02, 20)


# Current step (uA/cm^2)
# Define amplitude, init time and end time
amp = 0 #0.003
t_i = 100
t_f = 300

##############################################################################
# Initializing the neuron model
##############################################################################

# Initialize state variable values for t=0: x(0) = [V(0),m(0),h(0),n(0)]
# Default vals correspond to neuron at steady-state resting potential
# Final value in the init array is idx (starts at 0)
init = [-65,0.00742,0.47258,0.06356,0]

##############################################################################
# Running model: forward-integrating the equations of motion
##############################################################################
        
# Integrate model equations
# Arguments: state derivative, initial neuron state x(0), time point array
X = odeint(dXdt, init, t)

# Define variables to simplify analysis
VV = X[:,0]
mm = X[:,1]
hh = X[:,2]
nn = X[:,3]

# Adding Gaussian error to voltage trace (mV)
sigma_obs = 0.1
obs_error = np.random.normal(0, sigma_obs, len(VV))
VV_obs = VV + obs_error

##############################################################################
# Plotting and saving model output
##############################################################################

# Define total current
stimulation = stim[0:len(VV)] + i_inj(t)

# Plotting membrane voltage and stimulation time series
plt.subplot(2,1,1)
plt.plot(t,VV_obs,'k',linewidth=0.8)
plt.ylabel("Membrane Potential (mV)")
plt.subplot(2,1,2)
plt.ylabel("Current (uA)")
plt.plot(t,stimulation,'b',linewidth=0.8)  
plt.show()

# Save voltage data (without gaussian noise)
f = open('output/voltage_clean.csv', 'w')
for i in range(int(len(VV))):
    f.write('%f \n' % VV[i])
f.close()

# Save voltage data (with gaussian noise)
f = open('output/voltage.csv', 'w')
for i in range(int(len(VV))):
    f.write('%f \n' % VV_obs[i])
f.close()

# Save current stimulation data
f = open('output/stimulation.csv', 'w')
for i in range(int(len(VV))):
    f.write('%f\n' % stimulation[i])
f.close()

