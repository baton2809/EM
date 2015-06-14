#! /usr/bin/env python
#
# Copyright (C) 2015 Artiom Butomov <butapro7@gmail.com>

"""A tool for comparing of biological data in Python with HMM."""

import sys

from setuptools import setup, Extension


DISTNAME = "fairy"
DESCRIPTION = "A tool of comparing and analyzing"
LONG_DESCRIPTION = open("README.rst").read()
MAINTAINER = "Artiom Butomov"
MAINTAINER_EMAIL = "butapro7@gmail.com"
LICENSE = "new BSD"

CLASSIFIERS = [
               "Development Status :: 3 - Alpha",
               "License :: OSI Approved",
               "Intended Audience :: Developers",
               "Intended Audience :: Science/Research",
               "Topic :: Software Development",
               "Topic :: Scientific/Engineering",
               "Programming Language :: Cython",
               "Programming Language :: Python",
               "Programming Language :: Python :: 2",
               "Programming Language :: Python :: 2.6",
               "Programming Language :: Python :: 2.7",
               "Programming Language :: Python :: 3",
               "Programming Language :: Python :: 3.3",
               "Programming Language :: Python :: 3.4",
               ]

import fairy

VERSION = fairy.__version__

setup_options = dict(
                     name="fairy",
                     version=VERSION,
                     description=DESCRIPTION,
                     long_description=LONG_DESCRIPTION,
                     maintainer=MAINTAINER,
                     maintainer_email=MAINTAINER_EMAIL,
                     license=LICENSE,
                     url="https://github.com/baton2809/thesis",
                     packages=["fairy"],
                     classifiers=CLASSIFIERS,
                     ext_modules=[
                                  Extension("fairy._speedups", ["fairy/_speedups.c"])
                                  ]
                     )


# For these actions, NumPy is not required. We want them to succeed without,
# for example when pip is used to install seqlearn without NumPy present.
NO_NUMPY_ACTIONS = ('--help-commands', 'egg_info', '--version', 'clean')
if not ('--help' in sys.argv[1:]
        or len(sys.argv) > 1 and sys.argv[1] in NO_NUMPY_ACTIONS):
    import numpy as np
    setup_options['include_dirs'] = [np.get_include()]

setup(**setup_options)