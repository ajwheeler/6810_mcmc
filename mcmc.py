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
        #burnin
        for pos in self.sample(theta0, nburnin):
            theta0 = pos

        accepted = 0

        self._chain = np.empty((nsample, self._nwalkers, self._dim))
        for i, pos in enumerate(self.sample(theta0, nsample)):
            self._chain[i] = pos

            accepted += sum((self._chain[i] != self._chain[i-1]).any(axis=1))

        self._acceptance = float(accepted)/float(nsample * self._nwalkers)

    def chain(self):
        """returns 3-axis chain.  Axes are: sample number, walker, parameter"""
        return self._chain

    def flat_chain(self):
        """returns chain that has been flattened to (samplers * walkers), parameters"""
        ns, nw, d = self._chain.shape
        return self._chain.reshape((nw*ns, d))

    def acceptance_ratio(self):
        """return fraction of times the proposed step for a walker is excepted"""
        return self._acceptance

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

    def autocorrelation(self):
        """autocorrelation for each parameter in current chain"""
        flatchain = self.flat_chain()
        _, length = flatchain.shape

        s = np.empty(self._dim)
        ss = np.empty(self._dim)
        aa = np.empty(self._dim)
        for i in xrange(length-1):
            s += flatchain[i]
            ss += flatchain[i]**2
            aa += flatchain[i]*flatchain[i+1]
        s /= length
        ss /= length
        aa /= length

        return (aa - s**2)/(ss - s**2)



    def corner(self, ranges=None, true_vals=None, labels=None):
        """return plt fig of cornerplot of current chain"""
        #import here so class can be used without this dependancy
        import corner
        return corner.corner(self.flat_chain(), range=ranges, truths=true_vals,
                             labels=labels, show_titles=True if labels else False)

    def trace(self, walker, ranges=None, true_vals=None):
        """return plt fig containing trace plot of walker from current chain"""
        #import here so class can be used without this dependancy
        import matplotlib.pyplot as plt

        #get traceplot range
        xs = range(self.chain().shape[0])

        #initialize the figure, choose it's size
        fig = plt.figure(figsize=(20,3 * self._dim))

        for i in xrange(self._dim):
            plt.subplot(self._dim,1,i)
            ys = self._chain[:, walker, i]
            plt.plot(xs,self.chain()[:, walker, i], marker=',')

            if true_vals:
                plt.plot([xs[0], xs[-1]], [true_vals[i], true_vals[i]])

        return fig
