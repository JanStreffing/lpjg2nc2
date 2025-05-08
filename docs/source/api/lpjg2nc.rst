.. _api_lpjg2nc:

Main Module (lpjg2nc)
====================

The main module provides the command-line interface and orchestrates the overall conversion process.

.. automodule:: lpjg2nc
   :members:
   :undoc-members:
   :show-inheritance:

Command-line Interface
---------------------

.. autofunction:: parse_args

The main script handles several key operations:

1. Parsing command-line arguments
2. Finding and grouping output files by pattern
3. Coordinating parallel processing of files
4. Handling the test mode for specific patterns
5. Optional remapping to regular grids using CDO

Processing Functions
------------------

.. autofunction:: process_ifs_input_test
   :noindex:

.. autofunction:: main
   :noindex:

Utility Functions
---------------

.. autofunction:: run_subprocess
   :noindex:
