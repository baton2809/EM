from hmmlearn.hmm import GaussianHMM
import numpy as np
from hmmlearn.base import _BaseHMM
from scipy.stats import poisson
from sklearn import cluster
import math
import string

class PoissonHMM(_BaseHMM):

    def _get_rates(self):
        """Emission rate for each state."""
        return self._rates

    def _set_rates(self, rates):
        rates = np.asarray(rates)
        self._rates = rates.copy()

    rates_ = property(_get_rates, _set_rates)

    def _compute_log_likelihood(self, obs):
        n_samples = len(obs)
        log_prob = np.empty((n_samples, self.n_components))
        for c, rate in enumerate(self._rates):
            log_prob[:, c] = poisson.logpmf(obs, rate)
        return log_prob

    def _generate_sample_from_state(self, state, random_state=None):
        return poisson.rvs(self._rates[state])

    def _init(self, obs, params='str'):
        super(PoissonHMM, self)._init(obs, params=params)  # A и pi (для всей модели) -  2 x 2, 1 x 2

        concat_obs = np.concatenate(obs)
        if 'r' in params:
            if self.n_components == 4:          # if 4 then .T, elif 2 then nothing
                concat_obs = concat_obs.T
            kmeans = cluster.KMeans(n_clusters=self.n_components)
            self._rates = (kmeans.fit(np.atleast_2d(concat_obs).T)  # was np.atleast_2d(concat_obs).T
                               .cluster_centers_.T[0])
            self._rates.sort()

    def _initialize_sufficient_statistics(self):
        stats = super(PoissonHMM, self)._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros(self.n_components)
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(PoissonHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)

        if 'r' in params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs)

    def _do_mstep(self, stats, params):
        super(PoissonHMM, self)._do_mstep(stats, params)

        if 'r' in params:
            self._rates = stats['obs'] / stats["post"]

class MultiPoissonHMM(PoissonHMM):

    n_sample = 2

    def _get_D(self):
        return self._D

    def _set_D(self, D):
        D = np.asarray(D)
        self._rates = D.copy()

    D_ = property(_get_D, _set_D)

    def _init(self, obs, params='str'):
        # super(MultiPoissonHMM, self)._init(obs, params='str')
        super(PoissonHMM, self)._init(obs, params=params)
        # судя по формулам должны вычисляться правильно.

        # для каждого образца отдельно lambda
        # сортируем, меньший - "-" , больший - "+"
        # обьединяем.
        # получаем 4x1 rates
        concat_obs = np.concatenate(obs)
        if 'r' in params:
            kmeans = cluster.KMeans(n_clusters=int(self.n_components / 2))
            r1 = (kmeans.fit(np.atleast_2d(concat_obs.T[0]).T)
                  .cluster_centers_.T[0])
            r2 = (kmeans.fit(np.atleast_2d(concat_obs.T[1]).T)
                  .cluster_centers_.T[0])
            r1.sort(), r2.sort()
            r = []
            r.extend(r1), r.extend(r2)
            self._rates = r
        # а тогда то ли находит при пересчете lambda.
        # убедись на сентетическом примере
        # перепроверить

        self._D = np.array([[0, 1, 0, 1], [2, 3, 3, 2]])

    def _compute_log_likelihood(self, obs):

        pois = np.ones((len(obs), self.n_components))
        for i in range(self.n_components):
            for d in range(self.n_sample):
                for c in range(self.n_components):
                    if self._D[d, i] == c:
                        pois[:, i] *= poisson.logpmf(obs[:, d], self.rates_[c])

        log_prob = pois
        return log_prob

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(PoissonHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)

        if 'r' in params:
            stats['post'] += posteriors.sum(axis=0) * 2
            stats['obs'] += np.dot(posteriors.T, obs.sum(axis=1))

    def _do_mstep(self, stats, params):
        super(PoissonHMM, self)._do_mstep(stats, params)

        if 'r' in params:
            self._rates = stats['obs'] / stats["post"]

if __name__ == "__main__":
    x = np.loadtxt("covvec20", dtype=int)

    # hmm = PoissonHMM(n_components=2)
    # hmm.fit(obs=[x])
    # np.savetxt('y', hmm.predict(x))
    # print(len(x))
    # t = np.asarray(hmm.predict(x))
    # print(len(t[t > 0.]))

    # 240651
    # 109987

    # print(hmm.rates_)
    # print(hmm.transmat_)
    # print(hmm.startprob_)

    hmmMult = MultiPoissonHMM(n_components=4)
    x_2dim = np.array([x, x]).T
    hmmMult.fit([x_2dim])
    t = hmmMult.predict(x_2dim)
    print(len(t[t > 1.])) # т.к. 0 (- -) 1(+ +) - похожи; 2(- +) 3(+ -) - не похожи
    # рез-т говорит, что непохожих нет, что правильно так как образцы полностью одинкаовые.

    # np.savetxt('y', t)
    # print(hmmMult.rates_)
    # print(hmmMult.transmat_)
    # print(hmmMult.startprob_)
