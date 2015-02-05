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

K = 2
D = 2   # const!
N = 40

def init():
    """
    Generate initial mean and covariance
    :return: arrays of pi(s), means and covariances
    """
    pi = np.random.dirichlet(np.ones(K), size=1)    # sum(pi) = 1 - properties of dirichlet
    mean = np.random.rand(K, D)
    cov = range(K)
    for k in range(K):
        cov[k] = np.random.rand(D, D)               # is it important that cov belongs to (0,1) ?
    return pi, mean, cov

def cumsum():
    """
    Preprocess a cumulative sum
    :return: ndArray of the cumulative sum
    """
    cums = pi
    for i in range(1, pi.size):
        cums[0][i] += cums[0][i - 1]
    return cums

def sample_cat():
    """
    Binary search of the cumulative sum
    :return: k
    """
    return bisect.bisect_right(cums[0], np.random.rand())

def genData(mean, cov, dataSize):
    """
    Generate D-dimensional array of point

    x[t] = sample_gaussian(means[k], variances[k]);

    :param mean: mean
    :param cov: covariance
    :param dataSize: length of array-points
    :return: array of points
    """

    k = sample_cat()

    x,y = np.random.multivariate_normal(mean[k], cov[k], dataSize).T
    ndArray = np.array([[x[i],y[i]] for i in range(x.size)])
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
    return (1 / (2 * np.pi) ** (mean.ndim / 2)) * (1 / np.linalg.det(cov) ** .5) * \
        np.exp(-0.5 * np.dot(np.dot(x - mean, np.linalg.inv(cov)), (x - mean)))

def reestimate():
    print("do smt")

if __name__ == "__main__":
    pi, mean, cov = init()

    cums = cumsum()
    sample_cat()

    ndArray = genData(mean, cov, N)
    # print(ndArray)
    plotData(ndArray)



