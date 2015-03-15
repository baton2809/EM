__author__ = 'artiom'

import numpy as np
import bisect
from scipy.stats import poisson
import time
import math

K = 2
N = 500

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
    # I don't satisfy with the nan posterior probabilities on each [n] step.
    # Does the algorithm work correctly?
    return gamma

def m_step():
    Nk = gamma.sum(axis=0)
    for k in range(K):
        lambda_parameter[k] = np.dot(gamma.T[k], x) / Nk[k]
        pi[k] = Nk[k] / N

if __name__ == "__main__":
    start = time.time()
    # lambda_parameter = np.random.uniform(0.0, 5.0, K)
    # pi = np.random.dirichlet(np.ones(K), size=None)

    # # bin = 21
    # lambda_parameter = [130.74609375,  997.02702703]
    # pi = [0.11535049, 0.88464951]

    # i've got and pasted parameters after executing kmeanspp.py
    # bin = 20
    lambda_parameter = [267.69811321, 1192.93333333]
    pi = [0.18300206, 0.81699794]

    # # bin = 12
    # lambda_parameter = [14.51829268,  537.2]
    # pi = [0.02540835, 0.97459165]

    # x = data_generator()
    x = np.loadtxt("covvec", dtype=int)
    print(x)
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