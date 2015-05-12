"""HMM with 2 states"""

__author__ = 'artiom'

import numpy as np
import bisect
from scipy.stats import poisson
from scipy.misc import logsumexp
import glob
import time
import warnings
from sklearn.cluster import KMeans
warnings.filterwarnings("error")

GAMMA_FILE = './data/gamma/gamma20'
LAMBDA_FILE = './data/lambda/lambda20'
PI_FILE = './data/pi/pi20'

tolerance = 10**-3
epsilon = 1.0**-7
iter = -1

def forward_backward():
    alpha[0, :] = np.log(pi) + pois[0, ]  # basis
    for i in range(1, N):   # recursion
        alpha[i, ] = pois[i, ] + logsumexp(alpha[i-1, ] + np.log(A), axis=1)

    beta[N-1] = np.log(np.ones(K))  # basis
    for i in range(N-2, -1, -1):     # recursion
        beta[i, ] = logsumexp(beta[i+1, ] + np.log(A.T) + pois[i+1, ], axis=1)

    return alpha, beta  # actually it is log alpha and log beta

def e_step(log_alpha, log_beta):
    gamma = log_alpha + log_beta
    gamma -= logsumexp(gamma, axis=1)[:, np.newaxis]
    np.exp(gamma, gamma)

    for t in range(1, N):
        for i in range(K):
            for j in range(K):
                xi[t, i, j] = (log_alpha[t-1, i] + np.log(A[i, j]) +
                               pois[t, j] +
                               log_beta[t, j])

        xi[t] -= logsumexp(xi[t])
    np.exp(xi[1:], xi[1:])
    return gamma, xi

def baum_welch(gamma, xi):
    pi[:] = gamma[0, :].copy()  # copy slice
    A = np.sum(xi[1:, ], axis=0)
    A /= A.sum(axis=1)[np.newaxis].T
    Nk = np.sum(gamma, axis=0)
    for k in range(K):
        lambda_parameter[k] = np.dot(gamma.T[k], x) / Nk[k]
    return pi, A, lambda_parameter

def ll():
    return logsumexp(alpha[-1])

def poissons(Num_states=2):
    if Num_states == 2:
        for k in range(K):
            pois[:, k] = poisson.logpmf(x, lambda_parameter[k])
        return pois
    elif Num_states == 3:
        logpmf = np.zeros((N, K))
        logpmf[:, 0] = x.copy()
        logpmf[logpmf[:, 0] != 0, 0] = -np.inf
        logpmf[logpmf[:, 0] == 0, 0] = 1
        for k in range(1, K):
            logpmf[:, k] = poisson.logpmf(x, lambda_parameter[k])
        return logpmf

def kmeans_param(x):
    kmeans = KMeans(n_clusters=2, init='k-means++')
    kmeans.fit(x.reshape(-1, 1))
    clusters = kmeans.cluster_centers_
    pi = clusters / clusters.sum()
    pi.sort()
    return kmeans.cluster_centers_.reshape(-1,), pi.reshape(-1,)

def init_param(Num_states=2):
    if Num_states == 2:
        return kmeans_param(x)
    elif Num_states == 3:
        lambda_NULL = 0
        pi_NULL = (x == 0).sum()/N
        x_01 = x[x != 0]
        lambda_parameter, pi = [], []
        lambda_parameter_01, pi_01 = kmeans_param(x_01)
        pi_01 = (1. - pi_NULL) * pi_01 # аккуратно, проверь!
        lambda_parameter.append(lambda_NULL), lambda_parameter.extend(lambda_parameter_01)
        pi.append(pi_NULL), pi.extend(pi_01)
        return lambda_parameter, pi

if __name__ == "__main__":
    start = time.time()

    K = 2
    x = np.loadtxt("covvec", dtype=int)     # 20 bin
    N = len(x)
    pois = np.zeros((N, K))
    alpha = np.zeros((N, K))
    beta = np.zeros((N, K))
    gamma = np.zeros((N, K))
    xi = np.zeros((N, K, K))

    lambda_parameter, pi = init_param(K)
    A = np.full((K, K), 1/K)
    pois = poissons(K)  # <- lambda_old
    alpha, beta = forward_backward()  # old
    ll_old = ll()  # old

    while True:
        iter += 1
        print("step: {}".format(iter))
        gamma, xi = e_step(alpha, beta)  # new <- alpha_old
        pi, A, lambda_parameter = baum_welch(gamma, xi)  # new
        pois = poissons(K)  # <- lambda_new
        alpha, beta = forward_backward()  # new
        ll_new = ll()

        if iter and ll_new - ll_old < tolerance:
            break
        else:
            print('old: {}\nnew: {}'.format(ll_old, ll_new))
            ll_old = ll_new

    end = time.time()
    print('Iteration was made before convergence: {}\ntime : {} sec'.format(iter, round(end-start, 2)))
    print('convergences: {}\n              {}'.format(ll_old, ll_new))
    print(A)
    print(lambda_parameter)
    print(pi)



# ========================================= #

# step: 0
# old: -414746.69618496817
# new: -358278.61531018716
# step: 1
# old: -358278.61531018716
# new: -355842.9847561983
# step: 2
# Iteration was made before convergence: 2
# time : 101.1 sec
# convergences: -355842.9847561983
#               -360396.534245229         <--- bother me