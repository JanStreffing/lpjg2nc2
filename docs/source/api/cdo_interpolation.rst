.. _api_cdo_interpolation:

CDO Interpolation Module
======================

The CDO interpolation module provides utilities for remapping LPJ-GUESS output to regular global grids using Climate Data Operators (CDO).

.. automodule:: lpjg2nc.cdo_interpolation
   :members:
   :undoc-members:
   :show-inheritance:

Grid Remapping Functions
----------------------

.. autofunction:: remap_to_regular_grid
   :noindex:

The main function for remapping irregular grid data to a regular latitude-longitude grid. Supports both:
- Resolution-based specification (e.g., 1 degree)
- Grid dimension specification (e.g., 360x180 points)

Grid File Creation
----------------

.. autofunction:: create_grid_file
   :noindex:

Creates grid description files required by CDO for remapping operations.

Utility Functions
---------------

.. autofunction:: is_cdo_available
   :noindex:

Checks whether the CDO command-line tools are available in the current environment.
