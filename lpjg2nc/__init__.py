"""
lpjg2nc2 - Package containing utilities for converting LPJ-GUESS output files to NetCDF format.
"""

# Define version
__version__ = '1.0.0'

# Try to import modules, but allow documentation to build even if they fail
try:
    from lpjg2nc.file_parser import find_out_files, detect_file_structure
    from lpjg2nc.grid_utils import read_grid_information
    from lpjg2nc.netcdf_converter import process_file
    from lpjg2nc.count_nans import analyze_netcdf, print_short_summary
    from lpjg2nc.cdo_interpolation import remap_to_regular_grid
except ImportError:
    # When building docs, these imports may fail if dependencies aren't installed
    pass
