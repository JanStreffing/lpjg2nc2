.. _api_netcdf_converter:

NetCDF Converter Module
=====================

The NetCDF converter module handles the transformation of LPJ-GUESS output data into NetCDF format.

.. automodule:: lpjg2nc.netcdf_converter
   :members:
   :undoc-members:
   :show-inheritance:

Main Processing Functions
-----------------------

.. autofunction:: process_file
   :noindex:

The main entry point for processing LPJ-GUESS output files. This function determines the type of file (1D or 2D) and delegates to the appropriate handler.

.. autofunction:: process_2d_file
   :noindex:

Processes 2D output files (the most common type in LPJ-GUESS output), which contain spatial data across a grid.

Parallelization Utilities
-----------------------

.. autofunction:: get_parallel_config
   :noindex:

Functions that handle the configuration and execution of parallel processing, optimizing performance for large datasets.

Data Transformation
-----------------

.. autofunction:: expand_data_to_full_grid
   :noindex:

Functions that transform sparse data into a complete gridded dataset, properly handling missing values.
