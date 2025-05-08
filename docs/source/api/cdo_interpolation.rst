.. _api_cdo_interpolation:

CDO Interpolation Module
========================

The CDO interpolation module provides utilities for remapping LPJ-GUESS output to regular global grids using Climate Data Operators (CDO).

Key Functions
-------------

* ``remap_to_regular_grid(input_file, grid_param, output_file=None, verbose=False)``

  Remap an unstructured-grid NetCDF file to a regular latitude–longitude grid. Supports both resolution-based and grid-dimension specifications.

* ``create_grid_file(resolution, xsize=None, ysize=None, output_path=None)``

  Create a grid description text file that CDO uses during remapping operations.

* ``is_cdo_available()``

  Check whether the CDO command-line tool is available in the current environment.
