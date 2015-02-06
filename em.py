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

warnings.filterwarnings('error')

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
    cums = pi.copy()    # to resolve a problem of "if cums changes, pi will also change"
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

    x, y = np.random.multivariate_normal(mean[k], cov[k], dataSize).T
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
    # surely we get sometimes an exception while calculating a determinator of the covariance matrix
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
            denominator += pi[0][j] * gausDistr(ndArray[n], mean[j], cov[j])

        # print(denominator)             # uncomment it and run some times

        for k in range(K):
            # print(gausDistr(ndArray[n], mean[k], cov[k]))  # uncomment it and run some times
            numerator = pi[0][k] * gausDistr(ndArray[n], mean[k], cov[k])
            gamma[n][k] = numerator / denominator

    return gamma


def reestimate():
    """
    Re-estimate the parameters using the current responsibilities
    :return:
    """
    Nk = np.zeros(K)
    for i in range(K):
        for j in range(N):
            Nk[i] += gamma[j][i]

    print(mean)                                               # old mean
    for k in range(K):
        sum = 0
        for n in range(N):
            # print(ndArray[n])
            # print(gamma[n][k])
            sum += gamma[n][k] * ndArray[n]
        mean[k] = sum / Nk[k]
    print('{}\n---------------------------' . format(mean))   # new mean

    print(cov)                                                # old cov
    sum = range(K)
    for k in range(K):
        sum[k] = np.zeros((K, K))

    for k in range(K):
        for n in range(N):
            sum[k] += np.array(gamma[n][k] * np.matrix(ndArray[n] - mean[k]) * np.matrix(ndArray[n] - mean[k]).T)
        cov[k] = sum[k] / Nk[k]
    print('{}\n---------------------------' . format(cov))    # new cov

    print(pi)                                                 # old mixing coefficient
    for k in range(K):
        pi[0][k] = Nk[k] / N
    print('{}\n---------------------------' . format(pi))     # new mixing coefficient


if __name__ == "__main__":
    while True:
        try:
            pi, mean, cov = init()
            cums = cumsum()
            sample_cat()

            ndArray = genData(mean, cov, N)
            # print(ndArray)

            gamma = respons()
            plotData(ndArray)
            # print(gamma)
            reestimate()                                      # print an old mean and a new one
            break
        except Warning:
            pass