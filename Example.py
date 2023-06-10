import numpy as np
from DynamicsModel import *
from ParticleFilter import ParticleFilter
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
plt.style.use('ggplot')
plt.rc('xtick', labelsize=7) #fontsize of the x tick labels
plt.rc('ytick', labelsize=7) #fontsize of the y tick labels


# Initial conditions
x0 = np.array([[1, 2]]).T
Cov0 = 0.5 * np.diag(np.ones(2,))
numberOfSamples = 500
particleFilter = ParticleFilter(x0, Cov0, numberOfSamples, stateDynamics, measurementDynamics, Q, R)   

fig1 = plt.figure()
plt.plot(particleFilter.particles[0,:], particleFilter.particles[1,:], '.')
plt.xlabel('$x^1_k$')
plt.ylabel('$x^2_k$')
plt.title('Initial particles')
plt.savefig('Figures/InitialParticles.PNG')#,bbox_inches ="tight")

fig2 = plt.figure()
plt.plot(particleFilter.likelihoods)
plt.xlabel('index')
plt.ylabel('likelihood')
plt.title('Initial particles\' weights')
plt.savefig('Figures/Likelihoods.PNG',bbox_inches ="tight")

fig3 = plt.figure()
horizon = 40
x_rec = np.full((particleFilter.rx, horizon), np.nan)
x_true=np.full((particleFilter.rx, horizon + 1), np.nan)
x_true[:,0] = x0.squeeze() + sqrtm(Cov0) @ np.random.randn(2,)

particleFilter.initialize(x0, Cov0)
for k in range(horizon):
    u = np.full((1,), 0.5) #Step function. This can be replaced by a feedback controller
    x_true[:, k+1] = stateDynamics(x_true[:, k], u)+ sqrtm(particleFilter.Q).real @ np.random.randn(particleFilter.rx,)
    y = measurementDynamics(x_true[:, k+1], u)+ sqrtm(particleFilter.R).real @ np.random.randn(particleFilter.ry,)
    particleFilter.Apply_PF(u, y)
    x_rec[:, k] = particleFilter.sampleAverage()

plt.plot(x_rec.T, linewidth = 1)
plt.plot(x_true[:,0:-2].T, linewidth = 1)
plt.xlabel('time-step $k$')
plt.ylabel('magnitude')
plt.legend(('$x^2_{estimate}$', '$x^2_{estimate}$', '$x^1_{true}$', '$x^2_{true}$'))
plt.savefig('Figures/QuickRollout.PNG',bbox_inches ="tight")

