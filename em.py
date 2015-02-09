__author__ = 'artiom'

"""
The program do:
1) generate a two-dimensional data
2) Evaluate the log-likelihood function of the Gaussian mixture before convergence
3) Re-estimate the parameters
"""

import numpy as np
import bisect
import matplotlib.pyplot as plt
import warnings
import scipy
# from scipy.stats import multivariate_normal

warnings.filterwarnings('error')

K = 2
D = 2   # const!
N = 140

def init():
    """
    Generate initial mean and covariance
    :return: arrays of pi(s), means and covariances
    """
    pi = np.random.dirichlet(np.ones(K), size=None)    # sum(pi) = 1 - properties of dirichlet
    mean = np.random.rand(K, D)
    cov = []
    for k in range(K):
        cov.append(np.random.rand(D, D))
        cov[k] = cov[k].dot(cov[k].T)
    return pi, mean, cov

def sample_cat():
    """
    Binary search of the cumulative sum
    :return: k
    """
    return bisect.bisect_right(cs, np.random.rand())

# whether it is possible to simplify ???
def genData():
    """
    Generate D-dimensional array of points of Gaussian mixture

    x[t] = sample_gaussian(means[k], variances[k]);

    :return: array of points
    """
    lx, ly = [], []
    for i in range(N):
        k = sample_cat()
        x, y = np.random.multivariate_normal(mean[k], cov[k], 1).T
        lx.append(x.item(0)), ly.append(y.item(0))
    x, y = np.array(lx), np.array(ly)
    ndArray = np.array([[x[i], y[i]] for i in range(x.size)])
    np.savetxt('data', ndArray)
    return ndArray

def plotData(ndArray):
    """
    Plot image
    :param ndArray: array of points
    :return:
    """
    x = [[ndArray[i][0]] for i in range(ndArray.shape[0])]
    y = [[ndArray[i][1]] for i in range(ndArray.shape[0])]
    plt.plot(x, y, 'x')
    plt.axis('equal')
    plt.show()

def gausDistr(x, mean, cov):
    """
    Usage: gaussian(X[t], mean[k], cov[k])
    :param x: point in D-dimensional space
    :param mean: mean
    :param cov: covariance
    :return: result of calculating the Gaussian distribution
    """
    # return multivariate_normal.pdf(x, mean, cov)
    return (1 / (2 * np.pi) ** (mean.ndim / 2)) * (1 / np.linalg.det(cov) ** .5) * \
        np.exp(-0.5 * np.dot(np.dot(x - mean, np.linalg.inv(cov)), (x - mean)))

def respons():
    """
    Evaluate the responsibilities (E-step)
    :return: array gamma[N][K]
    """
    gamma = np.zeros((N, K))

    for n in range(N):
        denominator = 0
        for j in range(K):
            denominator += pi[j] * gausDistr(ndArray[n], mean[j], cov[j])   # can be simplified?

        for k in range(K):
            numerator = pi[k] * gausDistr(ndArray[n], mean[k], cov[k])
            gamma[n][k] = numerator / denominator

    return gamma


def reestimate():
    """
    Re-estimate the parameters using the current responsibilities
    :return:
    """

    Nk = gamma.sum(axis=1)

    # print(mean)                                               # old mean
    for k in range(K):
        sum = gamma[:, k].dot(ndArray)
        mean[k] = sum / Nk[k]
    # print('{}\n---------------------------' . format(mean))   # new mean

    # print(cov)                                                # old cov
    sum = np.zeros((K, K, K))
    for k in range(K):
        # print(gamma[:, k].dot(ndArray[:] - mean[k])) # 2x1
        # print((ndArray[:] - mean[k]).T.dot(ndArray[:] - mean[k])) # 2x2
        for n in range(N):
            sum[k] += gamma[n][k] * np.matrix(ndArray[n] - mean[k]) * np.matrix(ndArray[n] - mean[k]).T
        cov[k] = sum[k] / Nk[k]
    # print('{}\n---------------------------' . format(cov))    # new cov

    # print(pi)                                                 # old mixing coefficient
    for k in range(K):
        pi[k] = Nk[k] / N
    # print('{}\n---------------------------' . format(pi))     # new mixing coefficient

if __name__ == "__main__":
    # print(scipy.version.full_version)
    while True:
        try:
            pi, mean, cov = init()
            cs = (np.array(pi)).cumsum()
            # print('{}\n{}'.format(cs, pi))
            sample_cat()
            ndArray = genData()
            gamma = respons()
            # plotData(ndArray)                                 # doesn't work in miniconda python 3
            reestimate()                                      # print an old mean and a new one
            break
        except Warning:
            pass