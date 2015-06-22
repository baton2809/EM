
import numpy as np
from nose.tools import assert_equal
from fairy.faireanalysis import MultiPoissonHMM

def test_compute_D_single():
    assert MultiPoissonHMM._compute_D(n_samples=2) == np.array([[0, 1, 2, 0, 0, 1, 2, 2, 1],
                                                                [0, 1, 2, 2, 1, 0, 0, 1, 2]])

# if __name__ == "__main__":
#     import nose
#     nose.runmodule()