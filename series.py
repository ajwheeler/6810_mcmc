import mcmc
import numpy as np

def generate_data(f, xmin, xmax, npoints=100):
    xs = np.linspace(xmin, xmax, npoints)
    print xs

generate_data(1,0,1)
