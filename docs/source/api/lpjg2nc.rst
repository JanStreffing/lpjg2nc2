.. _api_lpjg2nc:

Main Module (lpjg2nc)
====================

The main module provides the command-line interface and orchestrates the overall conversion process.

Command-line Interface
---------------------

* ``parse_args()``: Parses command-line arguments and provides help information

The main script handles several key operations:

1. Parsing command-line arguments
2. Finding and grouping output files by pattern
3. Coordinating parallel processing of files (with significant performance improvements through vectorization)
4. Handling the test mode for specific patterns
5. Optional remapping to regular grids using CDO

Processing Functions
------------------

* ``process_ifs_input_test(path, output_path, verbose=False, n_jobs=1, remap=None)``: Processes ifs_input.out test files

* ``main()``: Main entry point for the command-line tool

Utility Functions
---------------

* ``run_subprocess(cmd)``: Executes and monitors external commands
