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

extensions = []
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

packages = find_packages(exclude=['etc', 'docs', 'tests'])

# Setup information
setup(
    name='edtk',
    version='0.0.1',
    description='Utilities for processing, imaging, and plotting ICEBEAR data',
    long_description=long_description,
    url='https://github.com/SuperDARNCanada/edtk',

    # Author details
    author='Adam Lozinsky, Remington Rohel',
    author_email='adam.lozinsky@usask.ca, remington.rohel@usask.ca',

    # Choose your license
    license='GPL v3.0',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU Lesser General Public License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],

    # What does your project relate to?
    keywords='icebear icebear-3d e-region ionosphere meteor radar',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=packages,

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html

    install_requires=['numpy', 'h5py', 'digital-rf', 'numba', 'matplotlib', 'imageio', 'scipy', 'opencv-python',
                      'PyYaml', 'pymap3d'],

    ext_modules=extensions,

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    # extras_require={
    #     'dev': ['check-manifest'],
    #     'test': ['coverage'],
    # },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.

    # package_data={'icebear': ['dat/*'],},
    # include_package_data=True,

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    # entry_points={
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },
