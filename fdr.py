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
        log_p0 = np.log(gamma[:, 0])

        acc, count = 0.0, 0
        mask = log_p0 <= np.log(.5)
        res = log_p0 * mask
        res[res == 0] = -np.inf
        acc = logsumexp(res)
        count = mask.sum()
        mFDR = np.exp(acc - np.log(count))

        # acc = logsumexp()
        # count = mask.sum()
        # for i in range(N):
        #     if p0[i] <= 0.5:
        #         acc = np.logaddexp(acc, log_p0[i])
        #         # acc += p0[i]
        #         count += 1
        # acc = np.exp(acc)
        # mFDR = acc / count


        nums = l[-2:]
        if 'a' in nums:
            nums = nums.replace('a', '')
        print('chr{} - {}'.format(nums, mFDR))

# chr0 - 1.1995865205217623e-42
# chr1 - 0.13558515922491945
# chr10 - 1.2940796768728552e-05
# chr11 - 0.17127440111106051
# chr13 - 0.08388966940863529
# chr14 - 0.05473103217927941
# chr15 - 0.000200033381191717
# chr16 - 0.0
# chr17 - 2.298567884015194e-16
# chr18 - 4.349472225605853e-59
# chr19 - 0.14482518546808315
# chr2 - 0.1610207087122315
# chr20 - 0.21212532115651572
# chr21 - 0.11659640739539245
# chr22 - 7.792327121953088e-05
# chr23 - 1.5684180698918506e-31
# chr3 - 0.11478039970067258
# chr4 - 4.210468489140651e-10
# chr5 - 3.5086242009348787e-175
# chr6 - 0.007639271299803457
# chr7 - 0.10104421208138556
# chr8 - 0.13787491163044366
# chr9 - 2.2472573134346117e-07