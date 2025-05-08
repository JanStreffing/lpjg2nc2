#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# Read the long description from README.md
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="lpjg2nc2",
    version="1.0.0",
    description="Convert LPJ-GUESS output files (.out) to NetCDF format",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jan Streffing",
    author_email="jan.streffing@awi.de",  # Replace with your email
    url="https://github.com/JanStreffing/lpjg2nc2",
    project_urls={
        "Documentation": "https://lpjg2nc2.readthedocs.io/",
        "Source": "https://github.com/JanStreffing/lpjg2nc2",
        "Issues": "https://github.com/JanStreffing/lpjg2nc2/issues",
    },
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "xarray>=0.19.0",
        "netCDF4>=1.5.7",
        "tqdm>=4.61.0",
    ],
    entry_points={
        "console_scripts": [
            "lpjg2nc=lpjg2nc:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    include_package_data=True,
    keywords="climate, lpj-guess, netcdf, conversion",
)
