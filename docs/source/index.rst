.. lpjg2nc2 documentation master file

Welcome to lpjg2nc2's documentation
===================================

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0
   :alt: License: GPLv3

lpjg2nc2 is a powerful tool for converting LPJ-GUESS output files (.out) to NetCDF format.
The tool searches for run* folders in a given path, finds .out files inside their output
folders, and converts them to the widely-used NetCDF format for easier analysis and visualization.

Key Features
-----------

* **Efficiently process LPJ-GUESS output**: Convert .out files to NetCDF format with proper coordinates and metadata
* **Parallel processing**: Utilize multi-core processing for faster conversion
* **Grid interpolation**: Remap to regular global grids using CDO
* **Data Analysis**: Analyze NaN values to understand data sparsity
* **Flexible configuration**: Control various aspects of the conversion process

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   examples
   api/modules
   performance
   utilities

Getting Started
--------------

Check the :doc:`installation` guide and the :doc:`usage` page to get started quickly. 
For more detailed examples, see the :doc:`examples` section.

Indices and Tables
-----------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
