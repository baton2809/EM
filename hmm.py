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

def forward_backward():
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

def e_step(log_alpha, log_beta):
    gamma = log_alpha + log_beta
    gamma -= logsumexp(gamma, axis=1)[:, np.newaxis]
    np.exp(gamma, gamma)

    # states_prob[:, 1:2] = gamma[:, 0:1]
    # for t in range(N):
    #     if x[t] == 0:
    #         states_prob[t] = [1, 0, 0]

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
    A = np.sum(xi[1:, ], axis=0)
    A /= A.sum(axis=1)[np.newaxis].T
    Nk = np.sum(gamma, axis=0)
    for k in range(K):
        lambda_parameter[k] = np.dot(gamma.T[k], x) / Nk[k]
    return pi, A, lambda_parameter

def ll():
    """
    вычисляем с новыми параметрами, поэтому
    :return:
    """
    # acc = 0
    # for i in range(K):
    #     acc += gamma[0, i] * np.log(pi[i])
    #
    # for n in range(1, N):
    #     for j in range(K):
    #         for k in range(K):
    #             acc += xi[n, j, k] * np.log(A[j, k])
    #
    # for n in range(N):
    #     for k in range(K):
    #         acc += gamma[n, k] * poisson.logpmf(x[n], lambda_parameter[k])
    #
    # return acc

    return logsumexp(alpha[-1])

    # c = alpha[-1].min()
    # return (logsumexp(alpha[-1] - c) + c)

def poissons():
    """
    To ease calculation and avoid repeated actions evaluating the Poisson's function in some methods
    :return: avector of poisson's probabilities
    """
    for k in range(K):
        pois[:, k] = poisson.logpmf(x, lambda_parameter[k])
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
    pois = np.zeros((N, K))
    alpha = np.zeros((N, K))
    beta = np.zeros((N, K))
    gamma = np.zeros((N, K))
    states_prob = np.zeros((N, K+1))
    xi = np.zeros((N, K, K))

    lambda_parameter = [4.45536015, 0.54416903]
    pi = [0.89115594, 0.10884406]

    A = np.full((K, K), 1/K)

    tolerance = 10**-3
    iter = -1

    pois = poissons()  # <- lambda_old
    alpha, beta = forward_backward()  # old
    ll_old = ll()  # old

    while True:
        iter += 1
        print("step: {}".format(iter))
        gamma, xi = e_step(alpha, beta)  # new <- alpha_old
        pi, A, lambda_parameter = baum_welch(gamma, xi)  # new
        pois = poissons()  # <- lambda_new
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

# ========================================= #

# step: 0
# [ 1.  1.  1. ...,  1.  1.  1.]
# 1.00000000001
# [ 1.  1.]
# [3.2749389746035282, 0.41106573343486802]
# old: -414746.69618496817
# new: -358278.61531018716
# step: 1
# [ 1.  1.  1. ...,  1.  1.  1.]
# 0.999999999982
# [ 1.  1.]
# [3.2201923366240224, 0.38724268531150302]
# old: -358278.61531018716
# new: -355842.9847561983
# step: 2
# [ 1.  1.  1. ...,  1.  1.  1.]
# 0.999999999987
# [ 1.  1.]
# [3.2705364719911167, 0.40685958082627571]
# Iteration was made before convergence: 2
# time : 101.72 sec
# convergences: -355842.9847561983
#               -360396.534245229

# step: 0
# True
# 1.00000000001
# -422688.549041
# step: 1
# True
# 0.999999999982
# -392652.89707
# step: 2
# True
# 0.999999999987
# -376866.502458
# step: 3
# True
# 0.999999999999
# -370423.229493
# step: 4
# True
# 0.999999999973
# -368784.492558
# step: 5
# True
# 1.0
# Iteration was made before convergence: 5
# time : 221.51 sec
# convergences: -368784.4925575147
#               -368939.7492457621


# step: 0
# True
# 1.00000000001
# -414746.696185
# step: 1
# True
# 0.999999999982
# -358278.61531
# step: 2
# True
# 0.999999999987
# -355842.984756
# step: 3
# True
# 0.999999999999
# -360396.534245
# step: 4
# True
# 0.999999999973
# -368748.98594
# step: 5
# True
# 1.0
# -379870.852984
# step: 6
# True
# 1.0
# -391579.02999
# step: 7
# True
# 1.0
# -400336.805302
# step: 8
# True
# 1.0
# -404312.855803
# step: 9
# True
# 1.0
# -405351.150991
# step: 10
# True
# 1.0
# -405335.811665
# step: 11
# True
# 1.0
# -404579.244542
# step: 12
# True
# 1.0
# -403223.625268
# step: 13
# True
# 1.0
# -401478.796646
# step: 14
# True
# 1.0
# -399814.070339
# step: 15
# True
# 1.0
# -398634.332979
# step: 16
# True
# 1.0
# -398014.221814
# step: 17
# True
# 1.0
# -397829.510208
# step: 18
# True
# 1.0
# -397893.864989
# step: 19
# True
# 1.0
# -398050.595535
# step: 20
# True
# 1.0
# -398203.86401
# step: 21