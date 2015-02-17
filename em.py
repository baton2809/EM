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
from scipy.misc import logsumexp
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
    mean, cov = np.random.rand(K, D), []
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

def genData():
    """
    Generate D-dimensional array of points of Gaussian mixture

    x[t] = sample_gaussian(means[k], variances[k]);

    :return: array of points
    """
    # whether it is possible to simplify ???
    lx, ly = [], []
    for i in range(N):
        k = sample_cat()
        x, y = np.random.multivariate_normal(mean[k], cov[k], 1).T
        lx.append(x.item(0)), ly.append(y.item(0))
    x, y = np.array(lx), np.array(ly)
    X = np.array([[x[i], y[i]] for i in range(x.size)])
    np.savetxt('data', X)

    a = np.max(X)

    return X

def plotData(ndArray):
    """
    Plot image
    :param ndArray: array of points
    :return:
    """
    x = [[ndArray[i][0]] for i in range(N)]
    y = [[ndArray[i][1]] for i in range(N)]
    plt.plot(x, y, 'x')
    plt.axis('equal')
    plt.show()

def gausDistr(X, mean, cov):
    """
    Usage: gaussian(X[t], mean[k], cov[k])
    :param x: point in D-dimensional space
    :param mean: mean
    :param cov: covariance
    :return: result of calculating the Gaussian distribution
    """
    return (1 / (2 * np.pi) ** (mean.ndim / 2)) * (1 / np.linalg.det(cov) ** .5) * \
        np.exp(-0.5 * np.dot(np.dot(X - mean, np.linalg.inv(cov)), (X - mean)))

def respons():
    """
    Evaluate the responsibilities (E-step)
    :return: array gamma[N][K]
    """
    gamma = np.zeros((N, K))

    # I have uninstall SciPy package from my ide in chance. :(
    # But it is impossible to install this one in the Project Interpreter for some reason.
    # This replacement doesn't work: multivariate_normal.pdf(X[n], mean[j], cov[j])

    for n in range(N):
        for k in range(K):
            gamma[n][k] = pi[k] * gausDistr(X[n], mean[k], cov[k]) / \
                          np.sum([pi[j] * gausDistr(X[n], mean[j], cov[j]) for j in range(K)])

    return gamma

def reestimate():
    """
    Re-estimate the parameters using the current responsibilities (M-step)
    :return:
    """

    Nk = gamma.sum(axis=0)

    print('mean:\n{}\ncov:\n{}\npi:\n{}\n' . format(mean, cov, pi))

    for k in range(K):
        mean[k] = np.dot(gamma.T[k], X) / Nk[k]

        mu_k = mean[k, np.newaxis]
        cov[k] = np.dot(gamma.T[k] * X.T, X) - np.dot(mu_k, mu_k.T)

        pi[k] = Nk[k] / N

    # print('mean:\n{}\ncov:\n{}\npi:\n{}\n' . format(mean, cov, pi))

def likelihood():
    """
    Evaluate the log likelihood function
    gmd is a Gaussian mixture distribution - p(X)
    :return: log likelihood value
    """
    gmd = np.zeros(N)

    for n in range(N):
        gmd = np.sum(pi[k] * gausDistr(X[n], mean[k], cov[k]) for k in range(K))

    return np.sum(np.log(gmd))

if __name__ == "__main__":
    # print(scipy.version.full_version)
    while True:
        try:
            tol = 10**-5    # convergence criteria
            pi, mean, cov = init()
            cs = (np.array(pi)).cumsum()
            sample_cat()
            X = genData()
            l_new = likelihood()
            iter = 0
            while True:
                iter += 1
                gamma = respons()
                # plotData(X)   # doesn't work in miniconda python 3
                reestimate()
                l_old = likelihood()
                print('log likelihood\nold: {}\nnew: {}\n'.format(l_old, l_new))
                if np.absolute(l_new/l_old - 1) < tol:
                    break
                else:
                    l_new = l_old

            print "Iteration was made before convergence: ", iter

            break
        except Warning:
            pass