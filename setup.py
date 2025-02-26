from setuptools import setup, find_packages

setup(
    name="ts",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
    ],
    author="Kartik Subramanian",
    author_email="karu@uchicago.edu",
    description="A library for time series modeling with ARMA and GARCH",
    keywords="time series, ARMA, GARCH, econometrics",
    python_requires=">=3.8",
) 