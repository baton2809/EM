__author__ = 'artiom'

import numpy as np
import bisect
from scipy.stats import poisson
from scipy.misc import logsumexp
from sklearn import hmm
import time
import math
import csv
import warnings
warnings.filterwarnings("ignore")

K = 2
N = 10000
epsilon = 1.0**-7
tolerance = 10**-3
LAMBDA_FILE = '../data/lambda/lambda21'
PI_FILE = '../data/pi/pi21'

def sample_cat(var):
    return bisect.bisect_right(np.array(var).cumsum(), np.random.rand())

def data_generator(lambda_parameter, pi):
    lx = []
    for i in range(N):
        x = np.random.poisson(lambda_parameter[sample_cat(pi)], 1)
        lx.append(x.item(0))
    x = np.array(lx)
    return x

def loglikelihood():
    lpr = np.zeros((N, K))
    for k in range(K):
        lpr[:, k] = np.log(pi[k]) + poisson.logpmf(x, lambda_parameter[k])
    acc = logsumexp(lpr, axis=1)
    return acc.sum()

def e_step(gamma):
    for k in range(K):
        gamma[:, k] = np.log(pi[k]) + poisson.logpmf(x, lambda_parameter[k])
    gamma -= logsumexp(gamma, axis=1)[:, np.newaxis]
    np.exp(gamma, gamma)
    return gamma

def m_step():
    Nk = gamma.sum(axis=0)
    for k in range(K):
        lambda_parameter[k] = np.dot(gamma.T[k], x) / Nk[k]
        pi[k] = Nk[k] / N
    return lambda_parameter, pi

def estimate_fdr(log_p0, labels):
    acc, count = 0.0, 0
    res = log_p0 * labels
    res[res == 0] = -np.inf
    acc = logsumexp(res)
    count = labels.sum()
    mFDR = np.exp(acc - np.log(count))
    return mFDR

if __name__ == "__main__":
    start = time.time()
    lambda_parameter = np.loadtxt(LAMBDA_FILE)
    pi = np.loadtxt(PI_FILE)

    y = np.zeros(N, dtype=int)  # label/states
    x = np.zeros(N, dtype=int)
    for t in range(N):
        y[t] = sample_cat(pi)
        x[t] = math.floor(np.random.poisson(lambda_parameter[y[t]], 1))

    try:
        # section - 1
        gamma = np.zeros((N, K))
        ll_new = loglikelihood()
        iter = 0
        while True:

            iter += 1
            gamma = e_step(gamma)
            m_step()
            ll_old = loglikelihood()
            if ll_old - ll_new < tolerance:
                break
            else:
                print('old: {}\nnew: {}'.format(ll_new, ll_old))
                ll_new = ll_old
        end = time.time()
        print(lambda_parameter)
        print(pi)
        print('Iteration was made before convergence: {}\ntime : {} sec'.format(iter, round(end-start, 2)))

        # section - 2
        log_p0 = np.log(gamma[:, 0])

        # mFDR
        mask = log_p0 <= np.log(.5)
        print(estimate_fdr(log_p0, mask))

        # true FDR
        print(estimate_fdr(log_p0, y))  # y - label/states

        # section 3
        # control FDR
        alpha = 6e-2
        p0 = gamma[:, 0].sort()
        for t in range(1, N):
            log_p0 = np.log(gamma[0:t, 0])
            temp = estimate_fdr(log_p0, y[0:t])
            if temp > alpha:
                break
            else:
                mFDR = temp
        # print(mFDR)  # 0.0599130650251 <= alpha = 0.06


    except ZeroDivisionError:
        print("Divide by zero!")
    except ArithmeticError as e:
        print(e)
        print('Iteration was made before halting: %d' % iter)
    finally:
        pass
