from setuptools import setup, Extension
import os, numpy

with open("README.md", "r") as fh:
    long_description = fh.read()

module = Extension('data_significance', sources = ['main.c', 'ridge.c', 'util.c', 'glm.c'], libraries = ['gsl', 'gslcblas'])

setup(
    name = 'data_significance',
    version = '1.3',
    author="Peng Jiang",
    author_email="pengj@alumni.princeton.edu",
    description = 'A few C functions for significance test in Python',
    
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    url="https://github.com/data2intelligence/data_significance",
    
    include_dirs = [os.path.join(numpy.get_include(), 'numpy')],
    
    install_requires=['numpy', 'pandas'],
    
    ext_modules = [module],
    )
