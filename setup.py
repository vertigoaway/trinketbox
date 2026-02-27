from setuptools import setup, find_packages
setup(
    name="trinketbox",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "numpy==2.2.6",
        "setuptools==59.6.0",
        "torch==2.10.0"
    ],
)