.. _api_file_parser:

File Parser Module
================

The file parser module provides utilities for finding and parsing LPJ-GUESS output files.

File Finding Functions
--------------------

* ``find_out_files(base_path)``
  
  Locates all .out files in the LPJ-GUESS run directories and groups them by pattern.
  This function scans all run*/output directories for .out files and organizes them
  by file pattern to enable efficient batch processing.

File Structure Detection
----------------------

* ``detect_file_structure(file_path)``
  
  Analyzes the structure of an LPJ-GUESS output file, identifying:
  
  - Header lines
  - The number of columns
  - Column names and their meanings
  - The data type of each column
  - Whether the file contains time series data

Data Reading Functions
--------------------

* ``read_and_combine_files(file_paths, structure=None, verbose=False)``
  
  Reads data from multiple LPJ-GUESS output files and combines them into a unified dataset.
  This function uses optimized data structures and vectorized operations for efficiently
  handling large datasets, contributing to the overall 8.8× performance improvement.
