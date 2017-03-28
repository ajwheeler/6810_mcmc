import mcmc
import numpy as np
import corner
import matplotlib.pyplot as plt

def lnprob(theta):
    x,y = theta[0], theta[1]
    return -(x**2 + y**2)

nw = 100
s = mcmc.sampler(lnprob, nw, 2, stepsize=.2)

theta0 = [np.array([0,0]) + np.random.randn(2) for _ in range(nw)]
s.run(theta0, 100, 10000)
fig = corner.corner(s.get_flat_chain(), range = [(-1,1), (-1,1)])
plt.show()