"""
SuperDARN Canada© -- Engineering Diagnostic Tools Kit: (setup)

Author: Adam Lozinsky
Date: October 28, 2021
Affiliation: University of Saskatchewan

Disclaimer: EDTK is under the GPL v3 license found in the root directory LICENSE.md.
"""

import os
from codecs import open
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Setup information
setup(
    name='edtk',
    version='0.0.1',
    description='Utilities for processing, imaging, and plotting ICEBEAR data',
    long_description=long_description,
    url='https://github.com/SuperDARNCanada/edtk',
    author='Adam Lozinsky, Remington Rohel',
    author_email='adam.lozinsky@usask.ca, remington.rohel@usask.ca',
    license='GPL v3.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: GNU General Public License v3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'],
    keywords='superdarn superdarncanada borealis engineering diagnostics toolkit radar',
    packages=find_packages(exclude=['docs', 'examples']),
    install_requires=['numpy', 'h5py', 'matplotlib', 'PyYaml', 'Pandas'],
    entry_points={
        'console_scripts': [
            'edtk-zvh=toolkit.rhodeschwarz.zvh_plots:main',
        ],
    },
)
