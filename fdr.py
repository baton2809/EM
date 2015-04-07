__author__ = 'artiom'

import numpy as np
from scipy.misc import logsumexp
from scipy.stats import poisson
import time
import glob
import warnings
warnings.filterwarnings("ignore")

GAMMA_FILES = './data/gamma/gamma*'
LAMBDA_FILES = './data/lambda/lambda*'
PI_FILES = './data/pi/pi*'
K = 2

if __name__ == "__main__":
    gamma_files_size = len(glob.glob(GAMMA_FILES))
    lambda_files_size = len(glob.glob(LAMBDA_FILES))
    pi_files_size = len(glob.glob(PI_FILES))
    if gamma_files_size == lambda_files_size == pi_files_size:
        l_files = list(zip(glob.glob(GAMMA_FILES), glob.glob(LAMBDA_FILES), glob.glob(PI_FILES)))
    for g, l, p in l_files:
        lambda_parameter = np.loadtxt(l)
        pi = np.loadtxt(p)
        gamma = np.loadtxt(g)
        N = len(gamma)

        p0 = gamma[:, 0]
        p1 = gamma[:, 1]

        E_FP, E_TP = 0, 0
        for i in range(N):
            if (p1[i] > 0.5):
                E_FP += p1[i]*p0[i]
                E_TP += p1[i]*p1[i]

        mFDR = E_TP / (E_TP + E_FP)
        nums = l[-2:]
        if 'a' in nums:
            nums = nums.replace('a', '')
        print('chr{} - {}'.format(nums, mFDR))