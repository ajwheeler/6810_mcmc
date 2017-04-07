# 6810_mcmc
https://github.com/ajwheeler/6810_mcmc

Programmer: Adam Wheeler

Simple MCMC sampler for Physics 6810 at OSU.  This module draws samples from a
n-dimensional posterior distribution given a likelihood function.

## Done
 - functional sampler
 - methods to automatically generate trace and corner plots
 - test on a simple bivariate Gaussian 

## to-do list:
 - multi-threading
 - autocorrelation time?
 - write what to expect from the acceptance ratio for a converged chain
 - ensemble sampling
 - test on a more interesting problem

this API is heavily influenced by that of [emcee](http://dan.iel.fm/emcee/current/api/).
