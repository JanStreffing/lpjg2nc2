.. _api_netcdf_converter:

NetCDF Converter Module
=====================

The NetCDF converter module handles the transformation of LPJ-GUESS output data into NetCDF format.

.. note::
   This module provides vectorized data processing capabilities, resulting in significant 
   performance improvements for large datasets.

Main Processing Functions
-----------------------

The module provides these key functions:

* ``process_file(file_paths, output_path, grid_info, verbose=False, inner_jobs=None, chunk_size=None, pattern_filter=None)``
  
  The main entry point for processing LPJ-GUESS output files. This function determines 
  the type of file (1D or 2D) and delegates to the appropriate handler.

* ``process_2d_file(file_paths, output_path, grid_info, verbose=False, inner_jobs=None, chunk_size=None, pattern_filter=None)``
  
  Processes 2D output files (the most common type in LPJ-GUESS output), which 
  contain spatial data across a grid.

Parallelization Utilities
-----------------------

* ``get_parallel_config(verbose=False, requested_jobs=0, requested_chunk_size=0)``
  
  Functions that handle the configuration and execution of parallel processing, 
  optimizing performance for large datasets.

Data Transformation
-----------------

* ``expand_data_to_full_grid(data, lat_idx, lon_idx, grid_info)``
  
  Functions that transform sparse data into a complete gridded dataset, properly 
  handling missing values and using efficient vectorized operations.
