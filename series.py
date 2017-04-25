import mcmc
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.log(x+1)

def true_coefficients(n):
    return [0] + [(-1)**(i+1) * 1./i for i in range(1,n)]

def series(coefficients, x):
    return sum([c * x**i for i,c in enumerate(coefficients)])

def labels(dim):
    return [r"$a_" + str(i) + r"$" for i in xrange(dim)]

def generate_data(f, xmin, xmax, npoints=10, sigma=0.5):
    """draw npoints points from f with gaussian noise"""
    xs = np.linspace(xmin, xmax, npoints)

    errs = np.random.normal(scale=sigma, size=npoints)

    f = np.vectorize(f)
    ys = f(xs) + errs

    return xs, ys

def lnprob(c, xs, ys, sigma):
    guess = series(c, xs)
    return -sum((guess-ys)**2)/(2* sigma**2)

if __name__ == '__main__':
    dim = 6
    sigma = 0.1

    xs, ys = generate_data(f,0,1, sigma=sigma)

    nw = 1000
    c0 = np.zeros((nw,dim)) + 1e-4*np.random.normal(size=(nw,dim))

    s = mcmc.sampler(lambda c: lnprob(c, xs, ys, sigma), nw, dim, stepsize=.1)

    s.run(c0, 500, 500)
    print(s.autocorrelation())

    s.corner(labels=labels(dim), true_vals=true_coefficients(dim))
    plt.show()
