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
# warnings.filterwarnings("error")

tolerance = 10**-3
epsilon = 10**-12
iter = -1

def forward_backward():
    alpha[0, :] = np.log(pi) + pois[0, ]  # basis
    for i in range(1, N):   # recursion
        alpha[i, ] = pois[i, ] + logsumexp(alpha[i-1, ] + np.log(A), axis=1)

    beta[N-1] = np.log(np.ones(K))  # basis
    for i in range(N-2, -1, -1):     # recursion
        beta[i, ] = logsumexp(beta[i+1, ] + np.log(A.T) + pois[i+1, ], axis=1)

    return alpha, beta  # actually it is log alpha and log beta

def check_gamma():
    """
    For debugging
    :return: True if each summation over states for gamma get 1.
    """
    t = gamma.sum(axis=1)
    print("gamma criteria is ", np.size(t[np.abs(t - 1.) > epsilon]) == 0)

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
    # np.exp(xi[1:], xi[1:])
    return gamma, xi    # xi - is actually log_xi

def baum_welch(gamma, xi):
    pi[:] = gamma[0, :].copy()  # copy slice
    A = logsumexp(xi[1:, ], axis=0)
    A -= logsumexp(A, axis=1)[np.newaxis].T
    np.exp(A, A)
    # A = np.sum(xi[1:, ], axis=0)
    # A /= A.sum(axis=1)[np.newaxis].T
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
    x = np.loadtxt("covvec20", dtype=int)     # 20 bin
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

        check_gamma()
        pi, A, lambda_parameter = baum_welch(gamma, xi)  # new
        print("A.sum(axis=1) get : ", A.sum(axis=1))
        print("pi.sum() get : ", pi.sum())
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
    print("A.sum(axis=1) get : ", A.sum(axis=1))
    print(lambda_parameter)
    print("pi.sum() get : ", pi.sum())



# ========================================= #
# # covvec20
# step: 0
# True
# A.sum(axis=1) get :  [ 1.  1.]
# pi.sum() get :  1.00000000002
# old: -414746.69635317405
# new: -358278.6153405261
# step: 1
# True
# A.sum(axis=1) get :  [ 1.  1.]
# pi.sum() get :  1.00000000003
# old: -358278.6153405261
# new: -355842.9848058823
# step: 2
# True
# A.sum(axis=1) get :  [ 1.  1.]
# pi.sum() get :  0.999999999987
# Iteration was made before convergence: 2
# time : 110.01 sec
# convergences: -355842.9848058823
#               -360396.5343190091
# A.sum(axis=1) get :  [ 1.  1.]
# [ 0.40685958  3.27053648]
# pi.sum() get :  0.999999999987


# # covvec15
# step: 0
# gamma criteria is  True
# A.sum(axis=1) get :  [ 1.  1.]
# pi.sum() get :  1.0
# old: -1257554.5380569273
# new: -939774.5100641939
# step: 1
# gamma criteria is  True
# A.sum(axis=1) get :  [ 1.  1.]
# pi.sum() get :  1.0
# old: -939774.5100641939
# new: -939107.8801286012
# step: 2
# gamma criteria is  True
# A.sum(axis=1) get :  [ 1.  1.]
# pi.sum() get :  1.0
# /Users/artiom/PycharmProjects/em-mixture/thesis/hmm.py:20: RuntimeWarning: divide by zero encountered in log
#   alpha[0, :] = np.log(pi) + pois[0, ]  # basis
# old: -939107.8801286012
# new: -939034.5526369443
# step: 3
# gamma criteria is  True
# A.sum(axis=1) get :  [ 1.  1.]
# pi.sum() get :  1.0
# old: -939034.5526369443
# new: -939032.1907665791
# step: 4
# gamma criteria is  True
# A.sum(axis=1) get :  [ 1.  1.]
# pi.sum() get :  1.0
# old: -939032.1907665791
# new: -939032.1017008605
# step: 5
# gamma criteria is  True
# A.sum(axis=1) get :  [ 1.  1.]
# pi.sum() get :  1.0
# old: -939032.1017008605
# new: -939032.0983463117
# step: 6
# gamma criteria is  True
# A.sum(axis=1) get :  [ 1.  1.]
# pi.sum() get :  1.0
# Iteration was made before convergence: 6
# time : 428.78 sec
# convergences: -939032.0983463117
#               -939032.0981913857
# A.sum(axis=1) get :  [ 1.  1.]
# [   1.66240934  219.86058331]
# pi.sum() get :  1.0