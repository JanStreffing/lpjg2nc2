.. _api_grid_utils:

Grid Utilities Module
=====================

The grid utilities module handles reading and manipulating coordinate grid information.

Key Functions
-------------

* ``read_grid_information(base_path)``

  Read grid information from ``grids.nc`` and return a dictionary describing a reduced Gaussian grid.

* ``match_coordinates_to_grid(lons, lats, grid_info)``

  Match longitude/latitude arrays from LPJ-GUESS output to the nearest points on the reduced Gaussian grid.

Grid Reading Functions
--------------------

This function reads grid information from the grids.nc file, which defines the spatial coordinates for LPJ-GUESS output.

Coordinate Matching Functions
---------------------------

These functions map irregular coordinate points to a structured grid, which is essential for creating properly formatted NetCDF files.
