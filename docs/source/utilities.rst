.. _utilities:

Utility Functions and Tools
=========================

This page documents additional utilities and tools that are part of the lpjg2nc2 package.

NaN Value Analysis
----------------

The ``count_nans`` module provides tools for analyzing the sparsity of data in NetCDF files, which is particularly useful for land-only data on global grids.

.. code-block:: python

    from lpjg2nc.count_nans import analyze_netcdf, print_short_summary
    
    # Analyze a NetCDF file for NaN values
    stats = analyze_netcdf('/path/to/file.nc', verbose=True, return_stats=True)
    
    # Print a concise summary
    print_short_summary(stats)

The analysis provides:
- Percentage of valid data points
- Percentage of NaN values
- Per-variable statistics
- Warnings for highly sparse datasets (>95% NaN)

CDO Grid Remapping
----------------

The ``cdo_interpolation`` module offers functionality to remap output to regular global grids using Climate Data Operators (CDO).

.. code-block:: python

    from lpjg2nc.cdo_interpolation import remap_to_regular_grid
    
    # Remap a NetCDF file to a 1-degree global grid
    remapped_file = remap_to_regular_grid(
        input_file='/path/to/input.nc',
        grid_param='360x180',  # Can also use resolution: 1.0
        verbose=True
    )

Key features:
- Support for both resolution-based (e.g., 1.0) and grid dimension-based (e.g., 360x180) specifications
- Automatic grid file generation
- Nearest-neighbor interpolation for unstructured grids

File Structure Detection
---------------------

The ``file_parser`` module provides tools for automatically detecting the structure of LPJ-GUESS output files.

.. code-block:: python

    from lpjg2nc.file_parser import detect_file_structure
    
    # Detect the structure of an LPJ-GUESS output file
    structure = detect_file_structure('/path/to/output.out')
    
    # Access the detected structure
    columns = structure['columns']
    has_time_index = structure['has_time_index']
    is_yearly = structure['is_yearly']

The detected structure includes:
- Column names and their meanings
- Whether the file has a time index
- Whether the data is yearly or contains sub-annual values
- Data types for each column

Grid Utilities
------------

The ``grid_utils`` module provides functions for working with spatial grids and coordinates.

.. code-block:: python

    from lpjg2nc.grid_utils import read_grid_information, match_coordinates_to_grid
    
    # Read grid information from a grids.nc file
    grid_info = read_grid_information('/path/to/base_dir/')
    
    # Match coordinates to a grid
    grid_index = match_coordinates_to_grid(lat, lon, grid_info)

These utilities help convert between irregular coordinate points and structured grids, which is essential for creating properly formatted NetCDF files.

Command-line Tools
---------------

In addition to the main ``lpjg2nc.py`` script, the package includes several command-line utilities:

1. **Test-run with ifs_input files:**

   .. code-block:: bash
   
       ./lpjg2nc.py -p /path/to/data/ --test ifs_input

2. **Run only for specific file pattern:**

   .. code-block:: bash
   
       ./lpjg2nc.py -p /path/to/data/ --pattern vegc.out

3. **Run with custom parallelization:**

   .. code-block:: bash
   
       ./lpjg2nc.py -p /path/to/data/ -j 16 --inner-jobs 32
