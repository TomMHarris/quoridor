"""
Build the Cython engine:
    python setup.py build_ext --inplace
"""

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(
        "engine_fast.pyx",
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
    include_dirs=[np.get_include()],
)
