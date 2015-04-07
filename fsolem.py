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
from scipy.misc import logsumexp
import csv
import warnings
warnings.filterwarnings("ignore")

K = 2
epsilon = 1.0**-7

def calculate_coverages(label=False):
    samfile = pysam.AlignmentFile("ENCFF000TJP.sorted.bam", "rb")
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
    # pi = np.sort(np.array([int(clusters[0])/sum_clusters, int(clusters[1])/sum_clusters]))
    pi = clusters / clusters.sum()
    pi.sort()
    return kmeans.cluster_centers_, pi

def check_gamma(gamma):
    count = 0
    for i in gamma:
        if abs(sum(i) - 1) > epsilon:
            count = count + 1
    print("{}/{}".format(count, len(gamma)-count))

# def sample_cat():
#     return bisect.bisect_right(np.array(pi).cumsum(), np.random.rand())
#
# def data_generator():
#     return np.random.poisson(lambda_parameter[sample_cat()], N)

def loglikelihood():
    if pi[1] == 0 or lambda_parameter[1] == 0:
        raise ZeroDivisionError

    # for i in range(N):
    #     if math.isnan(poisson_mixture_model[i]):
    #         print(poisson_mixture_model)
    #         raise ArithmeticError("Error: poisson mixture model value evaluates to NaN !")

    lpr = np.zeros((N, K))
    for k in range(K):
        lpr[:, k] = np.log(pi[k]) + poisson.logpmf(x, lambda_parameter[k])

    acc = logsumexp(lpr, axis=1)
    return acc.sum()

def e_step(gamma):
    for k in range(K):
        gamma[:, k] = np.log(pi[k]) + poisson.logpmf(x, lambda_parameter[k])

    # der = logsumexp(gamma, axis=1)
    # for k in range(K):
    #     gamma[:, k] = gamma[:, k] - der
    gamma -= logsumexp(gamma, axis=1)[:, np.newaxis]

    np.exp(gamma, gamma)

    check_gamma(gamma)
    # for i in range(N):
    #     if (abs(sum(gamma[i,:]) - 1) < epsilon):
    #         raise ArithmeticError("Error: the sum of posterior probabilities is not equal to 1 !")

    return gamma

def m_step():
    Nk = gamma.sum(axis=0)
    for k in range(K):
        lambda_parameter[k] = np.dot(gamma.T[k], x) / Nk[k]
        pi[k] = Nk[k] / N
    return lambda_parameter, pi

if __name__ == "__main__":

    start = time.time()

    # # i have commented it only to save time for getting coverages vector
    # bins_vector = calculate_coverages()
    # param_bins = []
    # for i in range(len(bins_vector)):
    #     param_bins.append(init_parameters(bins_vector[i]))
    #
    # with open("ENCFF000TJP.csv", "w", newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(bins_vector)

    # i do it to get data quickly(only for debugging and improving code)
    data = []
    with open("ENCFF000TJP.csv", "r") as t:
        u = csv.reader(t)
        for i in u:
            data.append(i)
    bins_vector = data
    param_bins = []
    bins_vector = [np.array([int(t) for t in bins_vector[i]]) for i in range(len(bins_vector))]

    for i in range(len(bins_vector)):
        param_bins.append(init_parameters(bins_vector[i]))

    # EM
    for i in range(len(bins_vector)):
        if i == 12:
            continue
        print("Step: %d" % i)
        lambda_parameter = param_bins[i][0].reshape(1, -1)[0]
        pi = param_bins[i][1]
        x = bins_vector[i]
        N = len(x)

        gamma = np.zeros((N, K))
        ll_new = loglikelihood()
        tolerance = 10**-5
        iter = 0
        print("gamma stats: (not 1 / 1)")
        while True:
            iter += 1
            gamma = e_step(gamma)
            lambda_parameter, pi = m_step()
            ll_old = loglikelihood()
            if np.absolute(ll_new / ll_old - 1) < tolerance:
                break
            else:
                ll_new = ll_old
        end = time.time()
        print('Iteration was made before convergence: {}'.format(iter))
        np.savetxt('./data/gamma/gamma'+str(i), gamma)
        np.savetxt('./data/lambda/lambda'+str(i), lambda_parameter)
        np.savetxt('./data/pi/pi'+str(i), pi)
        print("lambda:\n{}\npi:\n{}".format(lambda_parameter, pi))
    print('---------------------\nExecution time: ', round(end-start, 2))