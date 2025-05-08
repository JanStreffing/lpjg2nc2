.. _api_modules:

API Reference
============

This section provides an overview of the lpjg2nc2 Python modules and their key functions.

.. note::
   This documentation uses a simplified manual approach rather than autodoc to ensure
   compatibility with ReadTheDocs.

Key Components
-------------

* **Main Module** (:doc:`lpjg2nc`): Command-line interface and orchestration
* **File Parser** (:doc:`file_parser`): Finding and parsing LPJ-GUESS files
* **Grid Utilities** (:doc:`grid_utils`): Coordinate system management
* **NetCDF Converter** (:doc:`netcdf_converter`): Core conversion functionality with vectorized processing
* **CDO Interpolation** (:doc:`cdo_interpolation`): Remapping to regular grids
* **NaN Analysis** (:doc:`count_nans`): Analysis of data sparsity

.. toctree::
   :maxdepth: 2
   :hidden:

   lpjg2nc
   file_parser
   grid_utils
   netcdf_converter
   cdo_interpolation
   count_nans
