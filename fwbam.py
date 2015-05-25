__author__ = 'artiom'

"""
A Program calculates coverages vectors
"""

import time
import pysam
import math
import numpy as np
from itertools import count

import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})

from _speedups import compute_coverage

k = 200

if __name__ == "__main__":
    start = time.time()

    for reference_name in pysam.AlignmentFile("ENCFF000TJK.sorted.bam", "rb").references:
            # compute_coverage("ENCFF000TJP.sorted.bam", reference_name)
            np.savetxt('../data/coverages/ENCFF000TJK__'+reference_name,
                       compute_coverage("ENCFF000TJK.sorted.bam", reference_name), fmt='%i')


    # files = ["ENCFF000TJP.sorted.bam", "ENCFF000TJR.sorted.bam"]
    #
    # sample_files = [pysam.AlignmentFile("ENCFF000TJP.sorted.bam", "rb"),
    #                 pysam.AlignmentFile("ENCFF000TJR.sorted.bam", "rb")]
    #
    # for i in range(len(sample_files)):
    #     for reference_name in sample_files[i].references:
    #         # compute_coverage("ENCFF000TJP.sorted.bam", reference_name)
    #         np.savetxt('../data/coverages/biologiсal_sample_'+reference_name+'_'+str(i),
    #                    compute_coverage(files[i], reference_name), fmt='%i')

    end = time.time()
    print('time: {} sec'.format(round((end - start), 2)))