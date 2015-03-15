__author__ = 'artiom'

"""
    This file is just merge all subtasks written before into one file
"""

import time
import pysam
import math
import numpy as np
from sklearn.cluster import KMeans
import bisect
from scipy.stats import poisson
from statistics import *

K = 2
# N = 500

def calculate_coverages(label=False):
    samfile = pysam.AlignmentFile("ENCFF000TJP.sorted.bam", "rb")
    # samfile = pysam.AlignmentFile("ex40.7.sorted.bam", "rb")
    lengths_chr_list = samfile.lengths
    order, bins = 0, []
    for name_reference in samfile.references:
        bins.append(coverage_vector(name_reference, order, samfile, lengths_chr_list, print_stats=label))
        order += 1
    samfile.close()
    return bins

def coverage_vector(name, order, samfile, lengths_chr_list, k=100000, print_stats=False):
    bin = np.zeros((int(math.ceil(lengths_chr_list[order] / k) + 1)), dtype=int)
    count = 0
    for read in samfile.fetch(name):
        bin[int(math.ceil(read.reference_start / k))] += 1
    for i in range(len(bin)):
        count += bin[i]
    if print_stats:
        print("reference name: %s\n" % name)
        # despite all these printing write methods that occurs and catch exception if not equal
        print('was reads: {}\ngot reads: {}\n'.format(samfile.count(samfile.references[order]), count))
        print("coverage vector:\n{}\n".format(bin))
    return bin

def init_parameters(sample):
    kmeans = KMeans(n_clusters=2, init='k-means++')
    kmeans.fit(sample.reshape(-1, 1))    # one-dimensional
    clusters = kmeans.cluster_centers_
    sum_clusters = int(clusters[0]) + int(clusters[1])
    pi = np.sort(np.array([int(clusters[0])/sum_clusters, int(clusters[1])/sum_clusters]))
    # print("lambda : \n{}\npi : \n{}".format(kmeans.cluster_centers_, pi))
    return kmeans.cluster_centers_, pi

def sample_cat():
    return bisect.bisect_right(np.array(pi).cumsum(), np.random.rand())

def data_generator():
    return np.random.poisson(lambda_parameter[sample_cat()], N)

def loglikelihood():
    for n in range(N):
        poisson_mixture_model = np.sum(pi[k] * poisson.pmf(x[n], lambda_parameter[k]) for k in range(K))
    return np.sum(np.log(poisson_mixture_model))

def e_step():
    for n in range(N):
        for k in range(K):
            division = np.sum(pi[j] * poisson.pmf(x[n], lambda_parameter[j]) for j in range(K))
            # division shouldn't be nan/0.0
            if division == 0 or math.isnan(division):
                print("ERROR : {}, {}".format(pi, lambda_parameter))
            gamma[n][k] = pi[k] * poisson.pmf(x[n], lambda_parameter[k]) / division
    # I don't satisfy with almost the same posterior probabilities on each [n] step.
    # Does the algorithm work correctly?
    return gamma

def m_step():
    Nk = gamma.sum(axis=0)
    for k in range(K):
        lambda_parameter[k] = np.dot(gamma.T[k], x) / Nk[k]
        pi[k] = Nk[k] / N

if __name__ == "__main__":
    bins_vector = calculate_coverages()
    param_bins = []
    for i in range(len(bins_vector)):
        param_bins.append(init_parameters(bins_vector[i]))

    # EM
    start = time.time()

    i = 21
    lambda_parameter = param_bins[i][0].reshape(1, -1)[0]
    pi = param_bins[i][1]
    x = bins_vector[i]
    N = len(x)

    gamma = np.zeros((N, K))
    ll_new = loglikelihood()
    tolerance = 10**-5
    iter = 0
    while True:
        iter += 1
        gamma = e_step()
        m_step()
        ll_old = loglikelihood()
        if np.absolute(ll_new / ll_old - 1) < tolerance:
            break
        else:
            ll_new = ll_old
    end = time.time()
    print('Iteration was made before convergence: {}\ntime : {} sec'.format(iter, round(end-start, 2)))