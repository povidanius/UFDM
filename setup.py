from setuptools import setup, find_packages

setup(
    name="ufdm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "numpy>=1.20",
        "matplotlib>=3.5",
        "scipy>=1.7",
        "dcor>=0.5",
        "pandas>=1.5",
        "seaborn>=0.11",
    ],
    author="Your Name",
    description="Uniform Fourier Dependence Measure implementation",
    url="https://github.com/povidanius/UFDM",
)