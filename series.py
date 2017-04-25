# Use mcmc to estimate coefficients of series expansion of log(x+1)

import mcmc
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """the function to estimate the expansion coefficients for"""
    return np.log(x+1)

def true_coefficients(n):
    """the correct Taylor expansion of log(x+1)"""
    return [0] + [(-1)**(i+1) * 1./i for i in range(1,n)]

def series(coefficients, x):
    """Plug x into the series with given coefficients"""
    return sum([c * x**i for i,c in enumerate(coefficients)])

def labels(dim):
    """The first dim labels"""
    return [r"$a_" + str(i) + r"$" for i in xrange(dim)]

def generate_data(f, xmin, xmax, npoints=10, sigma=0.5):
    """draw npoints points from f with gaussian noise"""
    xs = np.linspace(xmin, xmax, npoints)

    errs = np.random.normal(scale=sigma, size=npoints)

    f = np.vectorize(f)
    ys = f(xs) + errs

    return xs, ys

def lnprob(c, xs, ys, sigma):
    """log-likelihood for coefficients c given data, assuming gaussian errors"""
    guess = series(c, xs)
    return -sum((guess-ys)**2)/(2* sigma**2)

if __name__ == '__main__':
    #the number of coefficients to estimate, the dimension of the paramter space
    dim = 6

    #the noise to apply to the data
    sigma = 0.1

    #generate the data
    xs, ys = generate_data(f,0,1, sigma=sigma)

    #the number of walkers
    nw = 1000

    #start positions for chains
    c0 = np.zeros((nw,dim)) + 1e-4*np.random.normal(size=(nw,dim))

    #create, run sampler
    s = mcmc.sampler(lambda c: lnprob(c, xs, ys, sigma), nw, dim, stepsize=.1)
    s.run(c0, 500, 500)

    print(s.autocorrelation())

    #draw cornerplot
    s.corner(labels=labels(dim), true_vals=true_coefficients(dim))
    plt.show()
