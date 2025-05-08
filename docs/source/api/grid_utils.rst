.. _api_grid_utils:

Grid Utilities Module
===================

The grid utilities module handles reading and manipulating coordinate grid information.

.. automodule:: lpjg2nc.grid_utils
   :members:
   :undoc-members:
   :show-inheritance:

Grid Reading Functions
--------------------

.. autofunction:: read_grid_information
   :noindex:

This function reads grid information from the grids.nc file, which defines the spatial coordinates for LPJ-GUESS output.

Coordinate Matching Functions
---------------------------

.. autofunction:: match_coordinates_to_grid
   :noindex:

These functions map irregular coordinate points to a structured grid, which is essential for creating properly formatted NetCDF files.
