

import matplotlib.pyplot as plt
import numpy as np

from context import SeReMpy
from SeReMpy.Geostats import *



def disturbModel(model, nsim, l=10.0):

    I=model.shape[0]
    J=model.shape[1]    

    variance_max = model.std()*1.0
    variance_max = variance_max*variance_max
    variance_min = model.std()*0.1
    variance_min = variance_min*variance_min

    X,Y =  np.mgrid[0:I,0:J]

    krig = 1
    sgsim = np.zeros((I, J, 3*nsim))
    xcoords = np.array([X.ravel(), Y.ravel()]).transpose()

    aux = 0
    for i in range(nsim):
        
        variance = np.random.uniform(0,1,1) * (variance_max - variance_min) + variance_min        
        dcoords = np.array([np.round(I/2), np.round(J/2)]).reshape((1,2))
        dz = model[int(np.round(I/2)), int(np.round(J/2))].reshape((1,1)) + np.sqrt(variance)*np.array(np.random.randn()).reshape((1,1))        
        
        sim = SeqGaussianSimulation(xcoords, dcoords, dz, 0.0, variance, l, 'exp', krig)
        sim = sim.reshape(I,J)
        sim = model + sim
        sgsim[:,:,aux] = np.reshape(sim, (I, J))
        aux += 1

        sim = SeqGaussianSimulation(xcoords, dcoords, dz, 0.0, variance, l, 'sph', krig)
        sim = sim.reshape(I,J)
        sim = model + sim
        sgsim[:,:,aux] = np.reshape(sim, (I, J))
        aux += 1

        sim = SeqGaussianSimulation(xcoords, dcoords, dz, 0.0, variance, l, 'gau', krig)
        sim = sim.reshape(I,J)
        sim = model + sim
        sgsim[:,:,aux] = np.reshape(sim, (I, J))
        aux += 1

    return sgsim



if __name__ == '__main__':

    model,Y =  np.mgrid[0:10,0:10]

    sim = disturbModel(model, 3)
    
    plt.imshow(model)
    plt.show()

    plt.imshow(sim[:,:,0])
    plt.show()

    plt.imshow(sim[:,:,1])
    plt.show()

    plt.imshow(sim[:,:,2])
    plt.show()