'''
Joe Taylor (j.taylor@bath.ac.uk)
Department of Physics
University of Bath, UK
May 1st, 2020

Python script of an echo state network for reservoir computing.
We extend the reservoir technique first proposed by Jaeger and Haas [1] to
accommodate external forcing. This approach was recently used by Pathak et 
al. to predict the spatiotemporal dynamics of the Kuramoto-Sivashinsky 
equation [2].

This script reads in past measurements of the evolution of a D-dimensional
dynamical system. Often only L state variables out of a total D are 
measurable. The number of forcing functions is denoted by F.

Example ~ Hodgkin-Huxley Neuron Model:
    
    Full neuron state given by x = [V,m,h,n]
    
    System is therefore four-dimensional. Typically only the membrane
    voltage V of the neuron is obtainable.
    
    The neuron is driven by external current stimulation. This is the 
    forcing function.
    
    Therefore: D = 4, L = 1, and F = 1.

    One can present the past evolution of membrane voltage to the echo
    state network during training, and subsequently predict this quantity
    for times beyond the training dataset.

[1] "Harnessing nonlinearity: Predicting chaotic systems and saving energy  
    in wireless communication", Science 304, 78 (2004)
[2] "Reservoir observers: Model-free inference of unmeasured variables in 
    chaotic systems", Chaos 27, 041102 (2017)
'''

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import pandas as pd

# Model parameters. 
model_pars = {'D': 4, 'L': 1, 'F': 1, 'dt': 0.025}

# Approximate size of reservoir 
approx_size = 2000

# Reservoir hyperparameters
res_hyperpars = {'radius':1.25,
                 'degree': 6,
                 'sigma': 0.4,
                 'train_length': 100000,
                 'N': int(np.floor(approx_size/(model_pars['L'] + model_pars['F'])) * (model_pars['L']+model_pars['F'])),
                 'num_inputs': model_pars['L'] + model_pars['F'],
                 'predict_length': 50000,
                 'beta': 0.0001
                 }

##############################################################################
# Functions for reservoir generation
##############################################################################

# Initialise a reservoir with node connections 
# given by adjacency matrix A
def gen_reservoir(size,radius,degree):
    """
    -Initialize reservoir with random 'degree' of connectivity
    -Sparsity: fraction of elements in adjacency matrix A that are non-zero 
    -Rescale A such that largest eigenvalue is equal to the 'spectral radius'
    |  :arg size: number of nodes in the reservoir
    |  :arg radius: spectral radius of adjacency matrix A
    |  :arg degree: average degree of connectivity of reservoir nodes
    |  :return: adjacency matrix A 
    """
    sparsity = degree/float(size);
    A = sparse.rand(size,size,density=sparsity,random_state=None).todense()
    eigenvals = np.linalg.eigvals(A)
    rescale = np.max(np.abs(eigenvals))
    A = (A/rescale) * radius
    return A

##############################################################################
# Functions for reservoir training
##############################################################################

# Calculate reservoir state r(t) for all 0 < t < T
def reservoir_layer(A, Win, data, forc, res_hyperpars):
    """
    -Res state at any time given by vector r(t)=[r_1(t),r_2(t),...r_N(t)]
    -N x T matrix encodes full res state at each time point in 0 < t < T
    -Res state evolves according to r(t+1) = tanh(A[r(t)]+Win[u(t)])
    |  :arg A: adjecency matrix of reservoir
    |  :arg Win: N x L input weight matrix ('input layer')
    |  :arg data: past measurements of dynamical system
    |  :arg forc: past measurements of forcing function
    |  :return: states matrix (N x T)
    """
    input = np.append(data,forc, axis=0)
    states = np.zeros((res_hyperpars['N'],res_hyperpars['train_length']))
    for i in range(res_hyperpars['train_length']-1):
        states[:,i+1] = np.tanh(np.dot(A,states[:,i]) + np.dot(Win,input[:,i]))
    return states

# Master function for training the network
def train_reservoir(res_hyperpars, data, forc):
    """
    -Generates reservoir network and trains it on input data
    -Calculates number (q) of connections to res for each input 
    |  :arg res_hyperpars: reservoir hyperparameters
    |  :arg data: past measurements of dynamical system
    |  :arg forc: past measurements of forcing function
    |  :return: final res state x, Wout, adj matrix A, Win
    """
    A = gen_reservoir(res_hyperpars['N'], res_hyperpars['radius'], res_hyperpars['degree'])
    q = int(res_hyperpars['N']/res_hyperpars['num_inputs'])
    Win = np.zeros((res_hyperpars['N'],res_hyperpars['num_inputs']))
    for i in range(res_hyperpars['num_inputs']):
#        np.random.seed(seed=seed_val+i)
        Win[i*q: (i+1)*q,i] = res_hyperpars['sigma'] * (-1 + 2 * np.random.rand(1,q)[0])       
    states = reservoir_layer(A, Win, data, forc, res_hyperpars)
    Wout = train(res_hyperpars, states, data)
    x = states[:,-1]
    return x, Wout, A, Win

# Find optimum weights in the output layer Wout
# using Tikhonov regularized regression
def train(res_hyperpars,states,data):
    """
    -Calculate the weights of the output layer by minimizing the
    -distance between the reservoir output and training data using
    -Tikhonov regularized regression.
    |  :arg res_hyperpars: reservoir hyperparameters
    |  :arg states: full reservoir state for all times
    |  :arg data: past measurements of dynamical system
    |  :return: transpose of optimized output layer Wout
    """
    beta = res_hyperpars['beta']
    idenmat = beta * sparse.identity(res_hyperpars['N'])
    states2 = states.copy()
    for j in range(2,np.shape(states2)[0]-2):
        if (np.mod(j,2)==0):
            states2[j,:] = (states[j-1,:]*states[j-2,:]).copy()
    U = np.dot(states2,states2.transpose()) + idenmat
    Uinv = np.linalg.inv(U)
    Wout = np.dot(Uinv,np.dot(states2,data.transpose()))
    return Wout.transpose()

##############################################################################
# Functions for reservoir prediction
##############################################################################
    
# Predict future evolution of the system for t > T
def predict(A, Win, res_hyperpars, forc, x, Wout):
    """
    -Uses final reservoir state to predict subsequent measurement
    -which is then presented to the reservoir as the next input
    |  :arg A: adjacency matrix A
    |  :arg Win: input layer 
    |  :arg res_hyperpars: reservoir hyperparameters
    |  :arg forc: past measurements of forcing function
    |  :arg x: final reservoir state at end of training 
    |  :arg Wout: output layer
    |  :return: transpose of optimized output layer Wout
    """
    forc = np.squeeze(forc)
    output = np.zeros((res_hyperpars['num_inputs'],res_hyperpars['predict_length']))
    for i in range(res_hyperpars['predict_length']):
        x2 = x.copy()
        for j in range(2,np.shape(x2)[0]-2):
            if (np.mod(j,2)==0):
                x2[j] = (x[j-1]*x[j-2]).copy()
        out = np.squeeze(np.asarray(np.dot(Wout,x2)))
        output[:,i] = out
        out = np.array([out,forc[i]])
        x1 = np.tanh(np.dot(A,x) + np.squeeze(np.dot(Win,out)))
        x = np.squeeze(np.asarray(x1))
    return output, x

##############################################################################
# Set up problem and calculate! Brrrrr!
##############################################################################
    
# File paths for data and forcing
dname = 'voltage.csv'
fname = 'stimulation.csv'

# Read in data files
dataf = pd.read_csv(dname,header=None)
forcf = pd.read_csv(fname,header=None)
data = np.transpose(np.array(dataf))
forc = np.transpose(np.array(forcf))

# Subtract mean from data, then normalise
# data and forcing by their standard devs
data -= np.mean(data)
data /= np.std(data)
forc /= np.std(forc)

# Skip n data points
skip_n = 0

# Seed value
seed_val = 1

# Train the reservoir on your data: 0 < t < T
print('Training...')
x,Wout,A,Win = train_reservoir(res_hyperpars, data[:,skip_n:skip_n+res_hyperpars['train_length']], forc[:,skip_n:skip_n+res_hyperpars['train_length']])

# Prediction of system evolution: t > T
print('Optimization complete! Predicting...')
output, _ = predict(A, Win,res_hyperpars,forc[:,res_hyperpars['train_length']:],x,Wout)

np.save('output.npy', output)
#output = np.load('output.npy')

# Define axes for plotting
tr = res_hyperpars['train_length']
pr = res_hyperpars['predict_length']
x_axis = list(range(0, tr + pr + skip_n))
x_pred = list(range(tr - 1, tr + pr - 1))

# Plot data and predictions
plt.figure()
plt.subplot(2,1,1)
plt.plot(x_axis[skip_n : tr + pr], data[0, skip_n : tr + pr], 'k',linewidth=1)
plt.plot(x_pred, output[0], 'r', linewidth=1)
plt.subplot(2,1,2)
plt.plot(x_axis[skip_n : tr + pr], forc[0, skip_n : tr + pr], 'b',linewidth=1)
plt.show()


