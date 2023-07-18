from setuptools import setup

setup(
    name="wellfactor",
    version="0.1.0",
    description="Python Implementation of NMF algorithms for WellFactor",
    author="Dongjin Choi",
    author_email="dchoi85@gatech.edu",
    packages=["wellfactor"],
    install_requires=["numpy", "scipy"],
)