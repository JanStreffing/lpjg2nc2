.. _api_file_parser:

File Parser Module
================

The file parser module provides utilities for finding and parsing LPJ-GUESS output files.

.. automodule:: lpjg2nc.file_parser
   :members:
   :undoc-members:
   :show-inheritance:

File Finding Functions
--------------------

.. autofunction:: find_out_files
   :noindex:

These functions locate the .out files in the LPJ-GUESS run directories and group them by pattern.

File Structure Detection
----------------------

.. autofunction:: detect_file_structure
   :noindex:

This function analyzes the structure of an LPJ-GUESS output file, identifying:
- Header lines
- The number of columns
- Column names and their meanings
- The data type of each column

Data Reading Functions
--------------------

.. autofunction:: read_and_combine_files
   :noindex:

These functions read data from multiple LPJ-GUESS output files and combine them into a unified dataset.
