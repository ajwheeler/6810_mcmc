# 6810_mcmc
https://github.com/ajwheeler/6810_mcmc

Programmer: Adam Wheeler

Simple MCMC sampler for Physics 6810 at OSU.  This module draws samples from a
n-dimensional posterior distribution given a likelihood function.

## Done
 - functional sampler
 - methods to automatically generate trace and corner plots
 - test on a simple bivariate Gaussian
 - use to estimate coefficients of series expansion from noisy data (`series.py`)
 - method to calculate autocorrelation time

## to-do list:
 - multi-threading
 - ensemble sampling?

this API is heavily influenced by that of [emcee](http://dan.iel.fm/emcee/current/api/).

## Dependancies
 - `numpy`
 - `matplotlib` is required for both the plotting methods
 - `corner.py` is required to generate corner plots
