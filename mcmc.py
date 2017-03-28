# Programmer: Adam Wheeler
# for physics 6810 at OSU
#
# to-do list:
#  - multithreding
#  - autocorrelation time
#
# this API is heavilly influenced by emcee
# http://dan.iel.fm/emcee/current/api/

import numpy as np

class sampler:

    #TODO make multithreaded?
    def __init__(self, lnprob, nwalkers, dim, stepsize=1):
        self._lnprob = lnprob
        self._nwalkers = nwalkers
        self._dim = dim
        self._stepsize = stepsize

    def run(self, theta0, nburnin, nsample):
        #TODO make not horrible
        for pos in self.sample(theta0, nburnin):
            theta0 = pos

        self._chain = np.empty((nsample, self._nwalkers, self._dim))
        for i, pos in enumerate(self.sample(theta0, nsample)):
            self._chain[i] = pos

    def get_chain(self):
        return self._chain

    def get_flat_chain(self):
        ns, nw, d = self._chain.shape
        return self._chain.reshape((nw*ns, d))

    def sample(self, theta0, nsample):
        positions = np.array(theta0)
        yield positions

        for _ in xrange(nsample-1):
            for w in xrange(self._nwalkers):
                #pick a random step, draw from a gaussian with width self._stepsize
                step = self._stepsize * np.random.randn(self._dim)

                #calculate the probabilities
                #TODO optimize to not calculate each one twice
                f1 = np.exp(self._lnprob(positions[w]))
                f2 = np.exp(self._lnprob(positions[w] + step))

                #alpha = f2/f1 = P2/P1
                alpha = f2/f1

                #take the proposed step if it is more probable than current pos
                #else take it proportionally to the ration of probabilities
                if alpha > 1 or np.random.rand() < alpha:
                    positions[w] += step
            yield positions
