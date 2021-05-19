# to compile invert_flow function, type this...
# python cythonize_invert_flow.py build_ext --inplace
from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("invert_flow.pyx", annotate=True))
