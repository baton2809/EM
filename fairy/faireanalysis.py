# -*- coding: utf-8 -*-

import time

import click
import pysam
import pyximport
import numpy as np
from joblib import Parallel, delayed
from hmmlearn.base import _BaseHMM
from scipy.stats import poisson
from scipy.misc import logsumexp

pyximport.install(setup_args={'include_dirs': np.get_include()})
from ._speedups import compute_coverage  # if ud like to import module
# from _speedups import compute_coverage     # if ud like to run from console

k = 200

# def read_observations(bamfile):
#     for reference_name in pysam.AlignmentFile(bamfile + ".sorted.bam", "rb").references:
#         np.savetxt('../data/coverages/' + bamfile + '__' + reference_name,
#                    compute_coverage(bamfile + ".sorted.bam", reference_name), fmt='%i')


class Reader(object):
    """
    Reader class
    """
    @staticmethod
    def stepwise_read_observations(bamfile, reference_name):
        return compute_coverage(bamfile + '.sorted.bam', reference_name)


class MultiPoissonHMM(_BaseHMM):
    """
    MultiPoissonHMM class
    """
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
    """
    Writer class
    """
    @staticmethod
    def save(pred, reference_start, chr):
        with open(chr+'.bed', 'a') as bed_file:
            pred = (pred > 2).astype(int)
            e = s = -1
            for i in range(len(pred)):
                if s == -1 and pred[i] == 1:
                    s = e = i
                if pred[i] == 0 and e != -1:
                    e = i
                    bed_file.write('{0}\t{1}\t{2}\n'.format(chr, reference_start+k*s, reference_start+k*e))
                    e = s = -1

    @staticmethod
    def WIG_save(X, reference_start, chr, group):
        # TODO if already exists then...
        with open(chr+'_'+group+'.wig', 'a') as wig_file:
            # TODO first row of output file: fixedStep  chrom=chrN  start=position  step=stepInterval
            if X.shape:
                X = X.sum(axis=0) / X.shape[0]
            for i in range(len(X)):
                wig_file.write('%s\n' % X[i])


def train(gr1, gr2, reference_name):
    start = time.time()
    print(reference_name)

    x = []
    y = []
    gr1_sep, gr2_sep = gr1.split(','), gr2.split(',')
    for bam in gr1_sep:
        x.append(Reader.stepwise_read_observations(bam, reference_name))
    for bam in gr2_sep:
        y.append(Reader.stepwise_read_observations(bam, reference_name))

    hmm_mult = MultiPoissonHMM(n_components=9, use_null=True)
    x_2dim = np.array(x + y)
    hmm_mult.fit([x_2dim])
    t = hmm_mult.predict(x_2dim, 'map')

    print('similarity: {0:.2f}%'.format((1 - len(t[t > 2.]) / len(t)) * 100))

    start_position = list(pysam.AlignmentFile(gr1_sep[0]+'.sorted.bam')
                          .fetch(reference_name))[0].reference_start
    Writer.save(t, reference_start=start_position, chr=reference_name)

    gr1_bams = [bam[-3::] for bam in gr1_sep]
    gr1_bams = '_'.join(gr1_bams)
    gr2_bams = [bam[-3::] for bam in gr2_sep]
    gr2_bams = '_'.join(gr2_bams)

    Writer.WIG_save(np.array(x), reference_start=start_position, chr=reference_name, group=gr1_bams)
    Writer.WIG_save(np.array(y), reference_start=start_position, chr=reference_name, group=gr2_bams)

    log_p0 = np.log(hmm_mult.posteriors[:, 0:2].sum(axis=1))
    mask = log_p0 <= np.log(.5)
    print('mFDR =', hmm_mult.estimate_fdr(log_p0, mask))

    end = time.time()
    print('time for {0}: {1} sec'.format(reference_name, round((end - start), 2)))


@click.command()
@click.option('--gr1', prompt='1st group',
              help='1st group of bam files without extension .bam')
@click.option('--gr2', prompt='2nd group',
              help='2nd group of bam files without extension .bam')
@click.option('--chr', default=None, help='Train specific chromatin(s).\n'
                                          'Write chromatin names separated by commas without spaces\n'
                                          'If none chromatin specifies the tool will analyze a whole file.\n'
                                          'Example: --chr=chr19,chr20,chr21')
def faireanalysis(gr1, gr2, chr):
    """A program for FAIRE-seq analysis and comparison"""
    if chr:
        reference_list = chr.split(',')
    else:
        # distiction exclusion
        reference_list = list(set(pysam.AlignmentFile(gr1.split(',')[0] + '.sorted.bam', 'rb').references)
                              .intersection(pysam.AlignmentFile(gr2.split(',')[0] + '.sorted.bam', 'rb').references))
    # exclude a chromosome M.
    reference_list = list(set(reference_list) - set(['chrM']))

    Parallel(n_jobs=2)(delayed(train)(gr1, gr2, reference_name) for reference_name in reference_list)

# if __name__ == '__main__':
#     fairy()
