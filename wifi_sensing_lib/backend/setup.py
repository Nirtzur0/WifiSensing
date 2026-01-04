from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

ext = Extension("csi_backend", sources=["csi_backend.pyx"], include_dirs=[get_include()])
setup(name = "csi_backend", ext_modules = cythonize(ext))
