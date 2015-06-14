cimport numpy as np
import numpy as np
import pysam

# cython _speedups.pyx
def compute_coverage(sample_file, reference_name, int k=200):
    cdef int order
    cdef int length
    cdef np.ndarray[np.int32_t, ndim=1] X
    with pysam.AlignmentFile(sample_file) as sam:
        order = sam.references.index(reference_name)
        length = sam.lengths[order]
        X = np.zeros(length // k + 1, dtype=np.int32)
        for read in sam.fetch(reference_name):
            X[read.reference_start // k] += 1
        return X