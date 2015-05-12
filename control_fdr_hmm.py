__author__ = 'artiom'

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
N = 500
epsilon = 1.0**-7

def sample_cat(var):
    return bisect.bisect_right(np.array(var).cumsum(), np.random.rand())

if __name__ == "__main__":
    start = time.time()

    A = [[0.48596109,  0.51403891], [0.04019019,  0.95980981]]
    lambda_parameter = [3.2705364719911167, 0.40685958082627571]
    pi = [1.3266356030895764e-05, 0.9999867336307543]

    N = 10000
    y = np.zeros(N, dtype=int)  # label/states
    x = np.zeros(N, dtype=int)
    y[0] = (sample_cat(pi))
    x[0] = (math.floor(np.random.poisson(lambda_parameter[y[0]], 1)))
    for t in range(1, N):
        y[t] = (sample_cat(A[y[t - 1]]))
        x[t] = (math.floor(np.random.poisson(lambda_parameter[y[t]], 1)))
    print(y)
    print(x)