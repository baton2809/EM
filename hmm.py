__author__ = 'artiom'

import numpy as np
import bisect
from scipy.stats import poisson

N = 300
K = 2
alpha = np.zeros((N, K))
beta = np.zeros((N, K))
gamma = np.zeros((N, K))
xi = np.zeros((N, K, K))

def sample_cat(pi):
    return bisect.bisect_right(pi.cumsum(), np.random.rand())

def data_gen(lambda_parameter, pi):
    lx = []
    [lx.append(np.random.poisson(lambda_parameter[sample_cat(pi)], 1).item(0)) for i in range(N)]
    x = np.array(lx)
    return x

def e_step(alpha, beta):
    # got nans
    gamma = alpha * beta / np.array(alpha * beta).sum(axis=1)[:, np.newaxis]

def forward_backward():
    # got values that tends to zero with increasing of steps(N)
    alpha[0, :] = pi * poisson.pmf(x[0], lambda_parameter)  # basis
    for i in range(1, N):   # recursion
        alpha[i, :] = poisson.pmf(x[i], lambda_parameter) * (alpha[i-1, 0] * A[0, ] + alpha[i-1, 1] * A[1, ])

    beta[N-1] = np.ones(K)  # basis
    for i in range(N-2, 0, -1):     # recursion
        beta[i, ] = beta[i+1, 0] * poisson.pmf(x[i+1], lambda_parameter[0]) * A[:, 0] + \
                    beta[i+1, 1] * poisson.pmf(x[i+1], lambda_parameter[1]) * A[:, 1]
    return alpha, beta

def ll():
    pass

if __name__ == "__main__":
    pi = np.array([0.4, 0.6])
    lambda_parameter = np.array([2, 100])
    A = np.array([[0.3, 0.7], [0.9, 0.1]])
    x = data_gen(lambda_parameter, pi)
    alpha, beta = forward_backward()
    gamma = e_step(alpha, beta)