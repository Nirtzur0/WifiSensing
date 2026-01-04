from setuptools import setup, find_packages

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
    entry_points={
        'console_scripts': [
            'wifi-sensing=wifi_sensing_lib.cli:main',
        ],
    },
)
