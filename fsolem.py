__author__ = 'artiom'

"""EM probability model
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

class FAIREseqReader(object):

    def __init__(self, file):
        self.bins_vector = []
        self.file = file

    def write_bins_vector(self):
        with open(self.file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.bins_vector)

    def coverage_vector(self, name, order, samfile, lengths_chr_list, k=200, print_stats=False):
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

    def calculate_coverages(self, label=False):
        samfile = pysam.AlignmentFile(self.file, "rb")
        lengths_chr_list = samfile.lengths
        order, bins = 0, []
        for name_reference in samfile.references:
            bins.append(self.coverage_vector(name_reference, order, samfile, lengths_chr_list, print_stats=label))
            order += 1
        samfile.close()
        return bins

    def read_coverages(self):
        data = []
        with open(self.file, "r") as t:
            u = csv.reader(t)
            for i in u:
                data.append(i)
        bins_vector = data
        bins_vector = [np.array([int(t) for t in bins_vector[i]]) for i in range(len(bins_vector))]
        return bins_vector

class EM(object):

    def __init__(self, bins_vector, FDR_strategy=None):
        if FDR_strategy:
            self.bins_vector = bins_vector
            self.estimate = FDR_strategy

    def init_parameters(self, sample):
        kmeans = KMeans(n_clusters=2, init='k-means++')
        kmeans.fit(sample.reshape(-1, 1))
        clusters = kmeans.cluster_centers_
        pi = clusters / clusters.sum()
        pi.sort()
        return kmeans.cluster_centers_, pi

    def check_gamma(self, gamma):
        t = gamma.sum(axis=1)
        print(np.size(t[t/1 - 1 > epsilon]) == 0)

    def loglikelihood(self):
        if self.pi[1] == 0 or self.lambda_parameter[1] == 0:
            raise ZeroDivisionError
        lpr = np.zeros((self.N, K))
        for k in range(K):
            lpr[:, k] = np.log(self.pi[k]) + poisson.logpmf(self.x, self.lambda_parameter[k])
        acc = logsumexp(lpr, axis=1)
        return acc.sum()

    def e_step(self, gamma):
        for k in range(K):
            gamma[:, k] = np.log(self.pi[k]) + poisson.logpmf(self.x, self.lambda_parameter[k])
        gamma -= logsumexp(gamma, axis=1)[:, np.newaxis]
        np.exp(gamma, gamma)
        # self.check_gamma(gamma)
        return gamma

    def m_step(self):
        Nk = self.gamma.sum(axis=0)
        for k in range(K):
            self.lambda_parameter[k] = np.dot(self.gamma.T[k], self.x) / Nk[k]
            self.pi[k] = Nk[k] / self.N
        return self.lambda_parameter, self.pi

    def train(self):
        param_bins = []
        for i in range(len(bins_vector)):
            param_bins.append(model.init_parameters(bins_vector[i]))
        for i in range(len(bins_vector)):
            print("Step: %d" % i)
            self.lambda_parameter = param_bins[i][0].reshape(1, -1)[0]
            self.pi = param_bins[i][1]
            self.x = bins_vector[i]
            self.N = len(self.x)
            self.gamma = np.zeros((self.N, K))
            ll_old = model.loglikelihood()
            tolerance = 10**-3
            iter = 0
            while True:
                iter += 1
                self.gamma = model.e_step(self.gamma)
                self.lambda_parameter, self.pi = model.m_step()
                ll_new = model.loglikelihood()
                if ll_new - ll_old < tolerance:
                    break
                else:
                    ll_old = ll_new
            print('Iteration was made before convergence: {}'.format(iter))
            np.savetxt('./data/gamma/gamma'+str(i), self.gamma)
            np.savetxt('./data/lambda/lambda'+str(i), self.lambda_parameter)
            np.savetxt('./data/pi/pi'+str(i), self.pi)

            print("lambda:\n{}\npi:\n{}".format(self.lambda_parameter, self.pi))

class StrategicAlternative(object):
    """Interface "Strategy"
    """
    def get_estimation(self):
        pass

class FDREstimate(StrategicAlternative):
    """Only FDR estimation
    """
    def get_estimation(self):
        print("Only FDR estimation")

class FDRControl(StrategicAlternative):
    """FDR estimation and control
    """
    def get_estimation(self):
        print("FDR estimation and control")


if __name__ == "__main__":

    start = time.time()

    # reader = FAIREseqReader("ENCFF000TJP.sorted.bam")
    reader = FAIREseqReader("ENCFF000TJP.csv")
    bins_vector = reader.read_coverages()

    model = EM(bins_vector, FDR_strategy=FDREstimate)
    model.train()

    end = time.time()
    print('Iteration was made before convergence: {}\ntime : {} sec'.format(iter, round(end-start, 2)))

    # i have commented it only to save time for getting coverages vector
    # bins_vector = model.calculate_coverages()
    # param_bins = []
    # for i in range(len(bins_vector)):
    #     param_bins.append(model.init_parameters(bins_vector[i]))
    # with open("ENCFF000TJP.csv", "w", newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(bins_vector)

    # i do it to get data quickly(only for debugging and improving code)
    # data = []
    # with open("ENCFF000TJP.csv", "r") as t:
    #     u = csv.reader(t)
    #     for i in u:
    #         data.append(i)
    # bins_vector = data
    # param_bins = []
    # bins_vector = [np.array([int(t) for t in bins_vector[i]]) for i in range(len(bins_vector))]

    # for i in range(len(bins_vector)):
    #     param_bins.append(model.init_parameters(bins_vector[i]))
    # for i in range(len(bins_vector)):
    #     lambda_parameter = param_bins[i][0].reshape(1, -1)[0]
    #     pi = param_bins[i][1]
    #     x = bins_vector[i]
    #     N = len(x)
    #
    #     gamma = np.zeros((N, K))
    #     ll_old = model.loglikelihood()
    #     tolerance = 10**-3
    #     iter = 0
    #     while True:
    #         iter += 1
    #         gamma = model.e_step(gamma)
    #         lambda_parameter, pi = model.m_step()
    #         ll_new = model.loglikelihood()
    #         if ll_new - ll_old < tolerance:
    #             break
    #         else:
    #             ll_old = ll_new
    #     end = time.time()
    #     np.savetxt('./data/gamma/gamma'+str(i), gamma)
    #     np.savetxt('./data/lambda/lambda'+str(i), lambda_parameter)
    #     np.savetxt('./data/pi/pi'+str(i), pi)

# Step: 0
# Iteration was made before convergence: 4
# lambda:
# [  1.78037997e+00   3.88033333e+03]
# pi:
# [[  9.99968706e-01]
#  [  3.12937561e-05]]
# Step: 1
# Iteration was made before convergence: 111
# lambda:
# [ 1.03770418  5.25823267]
# pi:
# [[ 0.92859689]
#  [ 0.07140311]]
# Step: 2
# Iteration was made before convergence: 151
# lambda:
# [ 1.0785057   4.35822749]
# pi:
# [[ 0.84164239]
#  [ 0.15835761]]
# Step: 3
# Iteration was made before convergence: 164
# lambda:
# [ 0.79063939  4.45367501]
# pi:
# [[ 0.96003628]
#  [ 0.03996372]]
# Step: 4
# Iteration was made before convergence: 5
# lambda:
# [   1.83951377  873.35555063]
# pi:
# [[  9.99971256e-01]
#  [  2.87438267e-05]]
# Step: 5
# Iteration was made before convergence: 2
# lambda:
# [    1.43461867  1042.03703704]
# pi:
# [[  9.99968442e-01]
#  [  3.15576506e-05]]
# Step: 6
# Iteration was made before convergence: 13
# lambda:
# [   1.57212732  317.97562591]
# pi:
# [[  9.99933444e-01]
#  [  6.65558616e-05]]
# Step: 7
# Iteration was made before convergence: 122
# lambda:
# [ 1.11090322  4.68235571]
# pi:
# [[ 0.87345499]
#  [ 0.12654501]]
# Step: 8
# Iteration was made before convergence: 77
# lambda:
# [ 0.68941571  4.48991586]
# pi:
# [[ 0.7660422]
#  [ 0.2339578]]
# Step: 9
# Iteration was made before convergence: 12
# lambda:
# [   1.55123151  770.58764825]
# pi:
# [[  9.99864064e-01]
#  [  1.35936432e-04]]
# Step: 10
# Iteration was made before convergence: 6
# lambda:
# [   1.83330246  288.61245894]
# pi:
# [[  9.99937715e-01]
#  [  6.22851018e-05]]
# Step: 11
# Iteration was made before convergence: 103
# lambda:
# [ 1.31071348  5.54606458]
# pi:
# [[ 0.885639]
#  [ 0.114361]]
# Step: 12
# Iteration was made before convergence: 2
# lambda:
# [   0.98730396  630.5       ]
# pi:
# [[  9.99996527e-01]
#  [  3.47312065e-06]]
# Step: 13
# Iteration was made before convergence: 128
# lambda:
# [ 0.61940583  3.36467655]
# pi:
# [[ 0.72205659]
#  [ 0.27794341]]
# Step: 14
# Iteration was made before convergence: 78
# lambda:
# [ 3.93284909  0.57000782]
# pi:
# [[ 0.34811444]
#  [ 0.65188556]]
# Step: 15
# Iteration was made before convergence: 7
# lambda:
# [   1.66259071  221.57259737]
# pi:
# [[  9.99568645e-01]
#  [  4.31355317e-04]]
# Step: 16
# Iteration was made before convergence: 2
# lambda:
# [    2.20727036  1840.        ]
# pi:
# [[  9.99997537e-01]
#  [  2.46318766e-06]]
# Step: 17
# Iteration was made before convergence: 4
# lambda:
# [   1.35011639  434.39818935]
# pi:
# [[  9.99859114e-01]
#  [  1.40886188e-04]]
# Step: 18
# Iteration was made before convergence: 5
# lambda:
# [   2.0847355   899.53124024]
# pi:
# [[  9.99891762e-01]
#  [  1.08237556e-04]]
# Step: 19
# Iteration was made before convergence: 77
# lambda:
# [ 1.10114754  5.28553212]
# pi:
# [[ 0.81353892]
#  [ 0.18646108]]
# Step: 20
# Iteration was made before convergence: 83
# lambda:
# [ 3.34072221  0.43130867]
# pi:
# [[ 0.25446876]
#  [ 0.74553124]]
# Step: 21
# Iteration was made before convergence: 67
# lambda:
# [ 0.28039341  2.78424645]
# pi:
# [[ 0.65783293]
#  [ 0.34216707]]
# Step: 22
# Iteration was made before convergence: 51
# lambda:
# [   1.09327342  136.22125151]
# pi:
# [[  9.99916099e-01]
#  [  8.39009271e-05]]
# Step: 23
# Iteration was made before convergence: 2
# lambda:
# [  5372.89473684  10873.56521739]
# pi:
# [[ 0.45238095]
#  [ 0.54761905]]
# Iteration was made before convergence: <built-in function iter>
# time : 720.68 sec