__author__ = 'artiom'

import time
import math
import string

import click
import pysam
import pyximport
from sklearn import cluster
from itertools import count
from collections import Counter
import numpy as np
from scipy.stats import poisson
from scipy.misc import logsumexp
from hmmlearn.base import _BaseHMM

pyximport.install(setup_args={'include_dirs': np.get_include()})
from _speedups import compute_coverage

k = 200

# def read_observations(bamfile):
#     for reference_name in pysam.AlignmentFile(bamfile + ".sorted.bam", "rb").references:
#         np.savetxt('../data/coverages/' + bamfile + '__' + reference_name,
#                    compute_coverage(bamfile + ".sorted.bam", reference_name), fmt='%i')

class Reader(object):
    @staticmethod
    def stepwise_read_observations(bamfile, reference_name):
        return compute_coverage(bamfile + '.sorted.bam', reference_name)

class MultiPoissonHMM(_BaseHMM):
    def __init__(self, *args, **kwargs):
        self.use_null = kwargs.pop('use_null', False)
        super(MultiPoissonHMM, self).__init__(*args, **kwargs)

        self.n_components1d = int(np.sqrt(self.n_components))

    # def _get_rates(self):
    #     """Emission rate for each state."""
    #     return self._rates
    #
    # def _set_rates(self, rates):
    #     rates = np.asarray(rates)
    #     self._rates = rates.copy()
    #
    # rates_ = property(_get_rates, _set_rates)

    def _init(self, obs, params='str'):
        super(MultiPoissonHMM, self)._init(obs, params=params)

        concat_obs = np.concatenate(obs)

        self.n_samples = len(concat_obs)
        self.n_rates = self.n_components1d * self.n_samples

        # self._D = np.array([[0, 1, 0, 1], [2, 3, 3, 2]])
        # self._D = np.array([[0, 2, 1, 0, 1, 0, 2, 2, 1],
        #                     [3, 5, 4, 4, 3, 5, 3, 4, 5]])
        self.D_ = self._compute_D(self.n_samples)

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

            self.rates_ = np.concatenate(rates)

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
                    if self.D_[d, i] == c:
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
                        if self.D_[d, i] == c:
                            stats['post'][c] += posteriors[i].sum(axis=0)
                            stats['obs'][c] += posteriors[i].dot(obs[d])

            self.posteriors = posteriors.copy()

    def _do_mstep(self, stats, params):
        super(MultiPoissonHMM, self)._do_mstep(stats, params)

        if 'r' in params:
            self.rates_ = stats['obs'] / stats['post']

    def estimate_fdr(self, log_p0, mask):
        res = log_p0.T * mask
        res[res == 0] = -np.inf     # exp(f) get 0.0
        acc = logsumexp(res)
        count = mask.sum()
        mFDR = np.exp(acc - np.log(count))
        return mFDR

class Writer(object):
    @staticmethod
    def save(pred, reference_start, chr):
        with open('../intermediate/'+chr+'.bed', 'a') as bed_file:
            pred = (pred > 2).astype(int)
            e = s = -1
            for i in range(len(pred)):
                if s == -1 and pred[i] == 1:
                    s = e = i
                if pred[i] == 0 and e != -1:
                    e = i
                    bed_file.write('{0}\t{1}\t{2}\n'.format(chr, reference_start+k*s, reference_start+k*e))
                    e = s = -1

@click.command()
@click.option('--bam1', prompt='1st bam file name',
              help='1st bam file name without extension .bam')
@click.option('--bam2', prompt='2nd bam file name',
              help='2nd bam file name without extension .bam')
@click.option('--chr', default=None, help='Train specific chromatin(s).\n'
                                          'Write chromatin names separated by commas without spaces\n'
                                          'If none chromatin specifies the tool will analyze a whole file.\n'
                                          'Example: --chr=chr19,chr20,chr21')
def faireanalysis(bam1, bam2, chr):
    """A program for FAIRE-seq analysis and comparison"""

    if chr:
        reference_list = chr.split(',')
    else:
        reference_list = pysam.AlignmentFile(bam1 + '.sorted.bam', 'rb').references

    start = time.time()
    for reference_name in reference_list:
        print(reference_name)
        x = Reader.stepwise_read_observations(bam1, reference_name)
        y = Reader.stepwise_read_observations(bam2, reference_name)

        hmmMult = MultiPoissonHMM(n_components=9, use_null=True)
        x_2dim = np.array([x, y])

        hmmMult.fit([x_2dim])

        t = hmmMult.predict(x_2dim, 'map')

        print('similarity: {0:.2f}%'.format((1 - len(t[t > 2.]) / len(t)) * 100))

        start_position = list(pysam.AlignmentFile('../intermediate/'+bam1+'.sorted.bam').fetch(reference_name))[0].reference_start
        Writer.save(t, reference_start=start_position, chr=reference_name)

        log_p0 = np.log(hmmMult.posteriors[:, 0:2].sum(axis=1))
        mask = log_p0 <= np.log(.5)
        print('mFDR =', hmmMult.estimate_fdr(log_p0, mask))

        end = time.time()
        print('time for {0}: {1} sec'.format(reference_name, round((end - start), 2)))
        start = end

if __name__ == '__main__':
    faireanalysis()

# 1st bam file name: ENCFF000TJR
# 2nd bam file name: ENCFF000TJP
# reference_name:  chr1
# similarity: 81.49%
# mFDR = 1.83796232665e-05
# reference_name:  chr2
# similarity: 93.52%
# mFDR = 0.00260956482779
# reference_name:  chr3
# similarity: 94.16%
# mFDR = 0.00287397879984
# reference_name:  chr4
# similarity: 94.61%
# mFDR = 0.0102689483662