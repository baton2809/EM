__author__ = 'artiom'

import numpy as np
import bisect
from scipy.stats import poisson

K = 2
N = 100

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
            gamma[n][k] = pi[k] * poisson.pmf(x[n], lambda_parameter[k]) / \
                np.sum(pi[j] * poisson.pmf(x[n], lambda_parameter[j]) for j in range(K))
    # I don't satisfy with almost the same posterior probabilities on each [n] step.
    # Does the algorithm work correctly?
    return gamma

def m_step():
    Nk = gamma.sum(axis=0)
    for k in range(K):
        lambda_parameter[k] = np.dot(gamma.T[k], x) / Nk[k]

    # \hat \pi to be continued...

if __name__ == "__main__":
    lambda_parameter = np.random.uniform(0.0, 5.0, K)
    pi = np.random.dirichlet(np.ones(K), size=None)
    x = data_generator()
    # print("The Log Likelihood : ", loglikelihood())
    gamma = np.zeros((N, K))
    gamma = e_step()
    print('old lambda: %a' % lambda_parameter)
    m_step()
    print('new lambda: %a' % lambda_parameter)