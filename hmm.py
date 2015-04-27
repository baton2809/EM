__author__ = 'artiom'

import numpy as np
import bisect
from scipy.stats import poisson
from scipy.misc import logsumexp
import glob

GAMMA_FILE = './data/gamma/gamma20'
LAMBDA_FILE = './data/lambda/lambda20'
PI_FILE = './data/pi/pi20'

# N = 300
K = 2

def forward_backward():
    # alpha[0, :] = pi * poisson.pmf(x[0], lambda_parameter)  # basis
    # for i in range(1, N):   # recursion
    #     alpha[i, :] = poisson.pmf(x[i], lambda_parameter) * (alpha[i-1, 0] * A[0, ] + alpha[i-1, 1] * A[1, ])
    #
    # beta[N-1] = np.ones(K)
    # for i in range(N-2, 0, -1):     # recursion
    #     beta[i, ] = np.sum(beta[i+1, ] * poisson.pmf(x[i+1], lambda_parameter) * A, axis=1)

    alpha[0, :] = np.log(pi) + poisson.logpmf(x[0], lambda_parameter)  # basis
    for i in range(1, N):   # recursion
        alpha[i, ] = poisson.logpmf(x[i], lambda_parameter) + logsumexp(alpha[i-1, ] + np.log(A), axis=1)

    beta[N-1] = np.log(np.ones(K))  # basis
    for i in range(N-2, 0, -1):     # recursion
        beta[i, ] = logsumexp(beta[i+1, ] + np.log(A) + poisson.logpmf(x[i+1], lambda_parameter), axis=1)

    return alpha, beta  # actually it is log alpha and log beta

def e_step(log_alpha, log_beta):
    # gamma = alpha * beta / np.array(alpha * beta).sum(axis=1)[:, np.newaxis]

    gamma = log_alpha + log_beta - logsumexp(log_alpha + log_beta, axis=1)[:, np.newaxis]
    np.exp(gamma, gamma)

    denom = np.zeros(N)

    # log_A = np.log(A)
    # log_pois = np.array([poisson.logpmf(x, lambda_parameter[t]) for t in range(K)]).T
    # for t in range(N):
    #     for i in range(K):
    #         for j in range(K):
    #             denom[t] += log_alpha[t-1, i] + log_A[i, j] + log_pois[t, j] + log_beta[t, j]
    #
    #     for i in range(K):
    #         for j in range(K):
    #             xi[t, i, j] = log_alpha[t-1, i] + log_A[i, j] + log_pois[t, j] + log_beta[t, j] - logsumexp(denom[t])
    # np.exp(xi, xi)

    alpha = np.exp(log_alpha)
    beta = np.exp(log_beta)
    for t in range(N):
        for i in range(K):
            for j in range(K):
                denom[t] += alpha[t-1, i] * A[i, j] * poisson.pmf(x[t], lambda_parameter[j]) * beta[t, j]
                # TROUBLE: denom = [0. .. 0.] <= вырожденные alpha, beta
        for i in range(K):
            for j in range(K):
                xi[t, i, j] = alpha[t-1, i] * A[i, j] * poisson.pmf(x[t], lambda_parameter[j]) * beta[t, j] / denom[t]

    return gamma, xi

def baum_welch(gamma, xi):

    pi = gamma[0, ]
    A = np.sum(xi[1:, ], axis=0) / np.sum(gamma[1:, ], axis=0)
    Nk = np.sum(gamma, axis=0)
    for k in range(K):
        lambda_parameter[k] = np.dot(gamma.T[k], x) / Nk[k]

    return pi, A, lambda_parameter

def ll():
    pass

if __name__ == "__main__":
    x = np.loadtxt("covvec", dtype=int)     # 20 bin

    N = len(x)
    alpha = np.zeros((N, K))
    beta = np.zeros((N, K))
    gamma = np.zeros((N, K))
    xi = np.zeros((N, K, K))

    lambda_parameter = [267.69811321, 1192.93333333]
    pi = [0.18327561, 0.81672439]

    A = np.array([[0.3, 0.7], [0.9, 0.1]])  # how to init correctly transition probability?

    alpha, beta = forward_backward()
    gamma, xi = e_step(alpha, beta)
    pi, A, lambda_parameter = baum_welch(gamma, xi)
    print(A)