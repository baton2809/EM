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
    pass

def m_step():
    pass

if __name__ == "__main__":
    lambda_parameter = np.random.uniform(0.0, 5.0, K)
    pi = np.random.dirichlet(np.ones(K), size=None)
    x = data_generator()
    print("The Log Likelihood : ", loglikelihood())