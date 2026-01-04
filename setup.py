from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

ext = Extension(
    "wifi_sensing_lib.backend.csi_backend",
    sources=["wifi_sensing_lib/backend/csi_backend.pyx"],
    include_dirs=[numpy.get_include()]
)

setup(
    name="wifi_sensing_lib",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "pyyaml",
        "pyshark",
        "tqdm",
        "matplotlib",
        "CSIKit",
        # WiPiCap/RF_CRATE are handled as local sources for now
    ],
    ext_modules=cythonize([ext]),
    entry_points={
        'console_scripts': [
            'wifi-sensing=wifi_sensing_lib.cli:main',
        ],
    },
)
