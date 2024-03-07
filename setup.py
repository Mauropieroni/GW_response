# setup.py
from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name="gw_response",
    version="0.0.1",
    packages=find_packages(),
    author="Mauro Pieroni",
    author_email="mauro.pieroni@cern.ch",
    description="A python code to compute the response function of a GW interferometer.",
    long_description=long_description,
    long_description_content_type="text/markdown",
)