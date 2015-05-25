from collections import Counter
from hmmlearn.hmm import GaussianHMM
import numpy as np
from hmmlearn.base import _BaseHMM
from scipy.misc import logsumexp
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


class MultiPoissonHMM(_BaseHMM):
    def __init__(self, *args, **kwargs):
        self.use_null = kwargs.pop("use_null", False)
        super(MultiPoissonHMM, self).__init__(*args, **kwargs)

        self.n_components1d = int(np.sqrt(self.n_components))

    def _get_rates(self):
        """Emission rate for each state."""
        return self._rates

    def _set_rates(self, rates):
        rates = np.asarray(rates)
        self._rates = rates.copy()

    rates_ = property(_get_rates, _set_rates)

    def _get_D(self):
        return self._D

    def _set_D(self, D):
        D = np.asarray(D)
        self._rates = D.copy()

    D_ = property(_get_D, _set_D)

    def _init(self, obs, params='str'):
        super(MultiPoissonHMM, self)._init(obs, params=params)

        concat_obs = np.concatenate(obs)

        self.n_samples = len(concat_obs)
        self.n_rates = self.n_components1d * self.n_samples

        # self._D = np.array([[0, 1, 0, 1], [2, 3, 3, 2]])
        # self._D = np.array([[0, 2, 1, 0, 1, 0, 2, 2, 1],
        #                     [3, 5, 4, 4, 3, 5, 3, 4, 5]])
        self._D = self._compute_D(self.n_samples)
        print(self._D)

        if 'r' in params:
            rates = []
            for d in range(self.n_samples):
                v = concat_obs[d].mean()
                c = 10
                # kmeans = cluster.KMeans(n_clusters=self.n_components1d)
                # r = (kmeans.fit(np.atleast_2d(concat_obs[d]).T)
                #       .cluster_centers_.T[0])
                # r.sort()
                # if self.use_null:
                #     r[1:] = r[:-1]
                #     r[0] = 0
                # r[-1] = r[-2] * 10
                r = np.array([0, v/c, v*c])

                rates.append(r)

            self._rates = np.concatenate(rates)
            print(self._rates)

    def _compute_D(self, n_samples):
        D = np.zeros((n_samples, self.n_components), dtype=int)
        p = 0
        for d in range(n_samples // 2):
            D[d] = p, p+1, p+2, p, p, p+1, p+2, p+2, p+1
            p += 3
        for d in range(n_samples // 2, n_samples):
            D[d] = p, p+1, p+2, p+2, p+1, p, p, p+1, p+2
            p += 3

        return D

    def _compute_log_likelihood(self, obs):
        log_prob = np.zeros((obs.shape[1], self.n_components))
        for i in range(self.n_components):
            for d in range(self.n_samples):
                for c in range(self.n_rates):
                    if self._D[d, i] == c:
                        work_buffer = poisson.logpmf(obs[d], self.rates_[c])
                        log_prob[:, i] += np.where(np.isnan(work_buffer),
                                                   np.log(obs[d] == 0),
                                                   work_buffer)

        return log_prob

    def _initialize_sufficient_statistics(self):
        stats = super(MultiPoissonHMM, self)._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_rates)
        stats['obs'] = np.zeros(self.n_rates)
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(MultiPoissonHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)

        if 'r' in params:
            posteriors = posteriors.T.copy()
            for c in range(self.n_rates):
                for d in range(self.n_samples):
                    for i in range(self.n_components):
                        if self._D[d, i] == c:
                            stats['post'][c] += posteriors[i].sum(axis=0)
                            stats["obs"][c] += posteriors[i].dot(obs[d])

            self.posteriors = posteriors.copy()


    def _do_mstep(self, stats, params):
        super(MultiPoissonHMM, self)._do_mstep(stats, params)

        if 'r' in params:
            self._rates = stats['obs'] / stats["post"]
            print(self._rates)

    def estimate_fdr(self, log_p0, mask):
        res = log_p0.T * mask
        res[res == 0] = -np.inf     # exp(f) get 0.0
        acc = logsumexp(res)
        count = mask.sum()
        mFDR = np.exp(acc - np.log(count))
        return mFDR

if __name__ == "__main__":
    # TJK, TJJ <=> TJP, TJR
    x = np.loadtxt("../data/coverages/biologiсal_sample_chr1_0", dtype=int)
    y = np.loadtxt("../data/coverages/ENCFF000TJJ__chr1", dtype=int)
    # hmmMult = MultiPoissonHMM(n_components=4)
    hmmMult = MultiPoissonHMM(n_components=9, use_null=True)
    x_2dim = np.array([x, y])
    hmmMult.fit([x_2dim])
    t = hmmMult.predict(x_2dim, "map")
    print(len(x) == len(y))
    print(Counter(t))
    # для n_components=9:
    print("{0:.2f}%".format((1 - len(t[t > 2.]) / len(t)) * 100))  #  0 (- -) 1(+ +) - похожи; 2(- +) 3(+ -) - не похожи
    # mFDR
    # H0 - разницы нет. H1 - разница есть
    # H0 отвергается, если p(++) + p(--) + p(nn) <= 0.5
    log_p0 = np.log(hmmMult.posteriors[:, 0:2].sum(axis=1))  #   1 - (n,n) 2 - (+,+) 3 - (-,-)
    mask = log_p0 <= np.log(.5)   # p0 - отвергается гипотеза "разницы нет"
    print('mFDR =', hmmMult.estimate_fdr(log_p0, mask))
    # !! mFDR = 2.4886891002e-06 это означает, что 3 раза мы соврали из 1246254.

    # для n_components=4:
    # print("{0:.2f}%".format((1 - len(t[t > 1.]) / len(t)) * 100))

    # control FDR
    # alpha = 1e-06
    # indices = hmmMult.posteriors[:, 0:2].sum(axis=1).argsort()
    # for t in range(1, len(x)):
    #     log_p0 = np.log(hmmMult.posteriors[indices[0:t], 0:2].sum(axis=1))
    #     temp = hmmMult.estimate_fdr(log_p0, mask[0:t])  # mask ?
    #     if temp > alpha:
    #         break
    #     else:
    #         mFDR = temp
    # print('controlled FDR ', t, len(x))