from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setup(
    name="gw_response",
    version="1.0.0",
    description="A package for handling LISA GW response",
    author="Mauro Pieroni, James Alvey",
    author_email="mauro.pieroni@cern.ch, j.b.g.alvey@uva.nl",
    packages=find_packages(),
    install_requires=required_packages,
)
