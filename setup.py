from setuptools import setup, find_packages

setup(
    name="gw_response",
    version="1.0.0",
    description="A package for handling LISA GW response",
    author="Mauro Pieroni, James Alvey",
    author_email="mauro.pieroni@cern.ch, j.b.g.alvey@uva.nl",
    packages=find_packages(),
    install_requires=[
        "python>=3.9;pla" "numpy>=1.26.4;python_version>='3.9'",
        "numpy==1.24.4;python_version<'3.9'",
        "jax>=0.4.25;python_version>='3.9'",
        "jax==0.4.13;python_version<'3.9'",
        "jaxlib>=0.4.25;python_version>='3.9'",
        "jaxlib==0.4.13;python_version<'3.9'",
        "chex>=0.1.85;python_version>='3.9'",
        "chex==0.1.7;python_version<'3.9'",
        "healpy>=1.16.6;python_version>='3.9'",
        "healpy==1.16.2;python_version<'3.9'",
        "typing_extensions==4.10.0;python_version=='3.11'",
    ],
)
