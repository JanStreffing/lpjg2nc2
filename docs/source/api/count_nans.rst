.. _api_count_nans:

NaN Analysis Module
=================

The NaN analysis module provides utilities for analyzing the sparsity of data in NetCDF files by counting NaN (Not a Number) values. This module was added to help better understand data sparsity in global datasets, especially for land-only data on global grids.

Analysis Functions
----------------

* ``analyze_netcdf(file_path, verbose=True, return_stats=False)``
  
  The main function for analyzing NetCDF files. It counts valid data points vs. NaN values and calculates sparsity statistics.
  
  This function processes a NetCDF file and computes:
  
  - Total number of data points
  - Number of valid (non-NaN) data points
  - Number of NaN values
  - Percentage of valid data and NaN values
  - Variable-specific statistics

Reporting Functions
-----------------

* ``print_short_summary(stats)``
  
  Provides a concise summary of NaN statistics, including:
  
  - Percentage of valid data points
  - Percentage of NaN values
  - Warning indicators for very sparse datasets (>95% NaN)

For land-only data on global grids, this analysis is particularly valuable, as such datasets typically contain many NaN values over ocean grid cells. In one analyzed dataset, the analysis revealed 99.82% NaN values, which is expected for land-only data on a global grid.
