.. _api_count_nans:

NaN Analysis Module
=================

The NaN analysis module provides utilities for analyzing the sparsity of data in NetCDF files by counting NaN (Not a Number) values.

.. automodule:: lpjg2nc.count_nans
   :members:
   :undoc-members:
   :show-inheritance:

Analysis Functions
----------------

.. autofunction:: analyze_netcdf
   :noindex:

The main function for analyzing NetCDF files. It counts valid data points vs. NaN values and calculates sparsity statistics.

Reporting Functions
-----------------

.. autofunction:: print_short_summary
   :noindex:

Provides a concise summary of NaN statistics, including:
- Percentage of valid data points
- Percentage of NaN values
- Warning indicators for very sparse datasets (>95% NaN)

This analysis is particularly useful for land-only data on global grids, which typically contain a large number of NaN values over ocean grid cells.
