__author__ = 'artiom'

import numpy as np
import bisect
from scipy.stats import poisson
from scipy.misc import logsumexp
import glob
import time
import warnings
warnings.filterwarnings("error")

GAMMA_FILE = './data/gamma/gamma20'
LAMBDA_FILE = './data/lambda/lambda20'
PI_FILE = './data/pi/pi20'

K = 2

def forward_backward(pois):
    # alpha[0, :] = np.log(pi) + poisson.logpmf(x[0], lambda_parameter)  # basis
    # for i in range(1, N):   # recursion
    #     alpha[i, ] = poisson.logpmf(x[i], lambda_parameter) + logsumexp(alpha[i-1, ] + np.log(A), axis=1)
    alpha[0, :] = np.log(pi) + pois[0, ]  # basis
    for i in range(1, N):   # recursion
        alpha[i, ] = pois[i, ] + logsumexp(alpha[i-1, ] + np.log(A), axis=1)

    beta[N-1] = np.log(np.ones(K))  # basis
    # for i in range(N-2, -1, -1):     # recursion
    #     beta[i, ] = logsumexp(beta[i+1, ] + np.log(A.T) + poisson.logpmf(x[i+1], lambda_parameter), axis=1)
    for i in range(N-2, -1, -1):     # recursion
        beta[i, ] = logsumexp(beta[i+1, ] + np.log(A.T) + pois[i+1, ], axis=1)

    return alpha, beta  # actually it is log alpha and log beta

def e_step(log_alpha, log_beta, pois):
    gamma = log_alpha + log_beta
    gamma -= logsumexp(gamma, axis=1)[:, np.newaxis]
    np.exp(gamma, gamma)

    for t in range(1, N):
        for i in range(K):
            for j in range(K):
                # xi[t, i, j] = (log_alpha[t-1, i] + np.log(A[i, j]) +
                #                poisson.logpmf(x[t], lambda_parameter[j]) +
                #                log_beta[t, j])
                xi[t, i, j] = (log_alpha[t-1, i] + np.log(A[i, j]) +
                               pois[t, j] +
                               log_beta[t, j])

        xi[t] -= logsumexp(xi[t])
    np.exp(xi[1:], xi[1:])
    return gamma, xi

def baum_welch(gamma, xi):
    pi = gamma[0, ]
    print(pi.sum())
    A = np.sum(xi[1:, ], axis=0)
    A /= A.sum(axis=1)[np.newaxis].T
    Nk = np.sum(gamma, axis=0)
    for k in range(K):
        lambda_parameter[k] = np.dot(gamma.T[k], x) / Nk[k]

    return pi, A, lambda_parameter

def ll():
    return logsumexp(alpha[-1])

def poissons():
    """
    To ease calculation and avoid repeated actions evaluating the Poisson's function in some methods
    :return: avector of poisson's probabilities
    """
    pois = np.zeros((N, K))
    for k in range(K):
        pois[:, k] = poisson.logpmf(x[:], lambda_parameter[k])
    return pois

def check_gamma():
    """
    For debugging
    :return: True if each summation over states for gamma get 1.
    """
    t = gamma.sum(axis=1)
    print(np.size(t[t/1 - 1 > 10**-5]) == 0)

if __name__ == "__main__":
    start = time.time()
    x = np.loadtxt("covvec", dtype=int)     # 20 bin
    N = len(x)
    alpha = np.zeros((N, K))
    beta = np.zeros((N, K))
    gamma = np.zeros((N, K))
    xi = np.zeros((N, K, K))

    lambda_parameter = [4.45536015, 0.54416903]

    pi = [0.89115594, 0.10884406]
    A = np.full((K, K), 1/K)
    pois = poissons()

    ll_new = ll()
    tolerance = 10**-3
    iter = 0
    while True:
        print("step: {}, likelihood = {}\n".format(iter, ll_new))
        # print("pi = {}\n A = {}\nlambda = {}".format(pi, A, lambda_parameter))
        iter += 1
        alpha, beta = forward_backward(pois)
        gamma, xi = e_step(alpha, beta, pois)
        check_gamma()
        pi, A, lambda_parameter = baum_welch(gamma, xi)

        ll_old = ll()
        if np.absolute(ll_new / ll_old - 1) < tolerance:
            break
        else:
            ll_new = ll_old
    end = time.time()
    print('Iteration was made before convergence: {}\ntime : {} sec'.format(iter, round(end-start, 2)))

# TROUBLE in loglikelihood

# True                                          gamma.sum(axis=1) = 1.
# 1.00000000001                                 pi.sum()
# step: 1, likelihood = -414746.69618496817
#
# True                                          gamma.sum(axis=1) = 1.
# 1.0                                           pi.sum()
# step: 2, likelihood = -365862.073677846
#
# True                                          gamma.sum(axis=1) = 1.
# 1.00000000001                                 pi.sum()
# step: 3, likelihood = -360476.2153981096
#
# True                                          gamma.sum(axis=1) = 1.
# 0.999999999974                                pi.sum()
# step: 4, likelihood = -367195.59363218537
#
# True                                          gamma.sum(axis=1) = 1.
# 1.0                                           pi.sum()
# step: 5, likelihood = -377342.47137207724