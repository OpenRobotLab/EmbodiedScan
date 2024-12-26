# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# hacky way to find numpy include path
# replace with actual path if this does not work
np_include_path = np.__file__.replace('__init__.py', 'core/include/')
INCLUDE_PATH = [np_include_path]

setup(ext_modules=cythonize(
    Extension('box_intersection',
              sources=['box_intersection.pyx'],
              include_dirs=INCLUDE_PATH)), )
