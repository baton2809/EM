__author__ = 'artiom'

import numpy as np
import bisect
from scipy.stats import poisson
from scipy.misc import logsumexp
import time
import math
import warnings
warnings.filterwarnings("ignore")

K = 2
N = 500

def sample_cat():
    return bisect.bisect_right(np.array(pi).cumsum(), np.random.rand())

def data_generator():
    return np.random.poisson(lambda_parameter[sample_cat()], N)

def loglikelihood():
    if pi[1] == 0 or lambda_parameter[1] == 0:
        raise ZeroDivisionError

    poisson_mixture_model = np.zeros(N)
    val = np.zeros((N, K))
    for k in range(K):
        val[:, k] = np.log(pi[k]) + poisson.logpmf(x, lambda_parameter[k])    # val is a 2D array
        poisson_mixture_model[:] = logsumexp(val)   # pmm is a 1D array

    for i in range(N):
        if math.isnan(poisson_mixture_model[i]):
            raise ArithmeticError("Error: poisson mixture model value evaluates to NaN !")

    return np.sum(poisson_mixture_model)

def e_step(gamma):
    divisions = np.sum(pi[j] * poisson.pmf(x, lambda_parameter[j]) for j in range(K))
    for k in range(K):
        gamma[:, k] = np.log(pi[k]) + poisson.logpmf(x, lambda_parameter[k]) - np.log(divisions)
    for k in range(K):
        gamma[:, k] = gamma[:, k] - logsumexp(gamma, axis=1)
    for i in range(N):
        if (sum(gamma[i,:]) > 1):
            raise ArithmeticError("Error: the sum of posterior probabilities is not equal to 1 !")
    return np.exp(gamma, gamma)

def m_step():
    Nk = gamma.sum(axis=0)
    for k in range(K):
        lambda_parameter[k] = np.dot(gamma.T[k], x) / Nk[k]
        pi[k] = Nk[k] / N

def test0():
    x = data_generator()
    lambda_parameter = np.random.uniform(0.0, 5.0, K)
    pi = np.random.dirichlet(np.ones(K), size=None)
    return lambda_parameter, pi

def test1():
    # bin = 21
    lambda_parameter = [130.74609375,  997.02702703]
    pi = [0.11535049, 0.88464951]
    return lambda_parameter, pi

def test2():
    # bin = 20
    lambda_parameter = [267.69811321, 1192.93333333]
    pi = [0.18300206, 0.81699794]
    return lambda_parameter, pi

def test3():
    # bin = 12
    lambda_parameter = [14.51829268,  537.2]
    pi = [0.02540835, 0.97459165]
    return lambda_parameter, pi


if __name__ == "__main__":
    start = time.time()

    x = np.loadtxt("covvec", dtype=int)
    N = len(x)
    lambda_parameter, pi = test3()

    try:
        gamma = np.zeros((N, K))
        ll_new = loglikelihood()
        tolerance = 10**-5
        iter = 0
        while True:
            iter += 1
            gamma = e_step(gamma)
            m_step()
            ll_old = loglikelihood()
            if np.absolute(ll_new / ll_old - 1) < tolerance:
                break
            else:
                ll_new = ll_old
        end = time.time()
        print('Iteration was made before convergence: {}\ntime : {} sec'.format(iter, round(end-start, 2)))
    except ZeroDivisionError:
        print("Divide by zero!")
    except ArithmeticError as e:
        print(e)
        print('Iteration was made before halting: %d' % iter)