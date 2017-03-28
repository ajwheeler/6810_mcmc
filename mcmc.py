import numpy as np

class sampler:
    """Simple MCMC sampler"""
    def __init__(self, lnprob, nwalkers, dim, stepsize=1):
        self._lnprob = lnprob
        self._nwalkers = nwalkers
        self._dim = dim
        self._stepsize = stepsize

    def run(self, theta0, nburnin, nsample):
        """Run and save a chain of length nsample"""
        #TODO make not horrible
        for pos in self.sample(theta0, nburnin):
            theta0 = pos

        self._chain = np.empty((nsample, self._nwalkers, self._dim))
        for i, pos in enumerate(self.sample(theta0, nsample)):
            self._chain[i] = pos

    def get_chain(self):
        """returns 3-axis chain.  Axes are: sample number, walker, parameter"""
        return self._chain

    def get_flat_chain(self):
        """returns chain that has been flattened to (samplers * walkers), parameters"""
        ns, nw, d = self._chain.shape
        return self._chain.reshape((nw*ns, d))

    def sample(self, theta0, nsample):
        """generator for nsample iterations of walker positions, starting at theta0"""
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

    def cornerplot(self, ranges=None):
        """return plt fig of cornerplot"""
        import corner #import here so class can be used without this dependancy
        return corner.corner(self.get_flat_chain(), range=ranges)
