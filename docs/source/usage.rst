.. _usage:

Usage Guide
==========

Command Line Interface
---------------------

lpjg2nc2 is primarily used as a command-line tool. Here's the basic syntax:

.. code-block:: bash

    ./lpjg2nc.py -p /path/to/lpj_guess_runs/ [options]

Required Arguments
~~~~~~~~~~~~~~~~

* ``-p PATH, --path PATH``: Path to the directory containing run* folders (required)

Optional Arguments
~~~~~~~~~~~~~~~~

* ``-f FILE, --file FILE``: Process a specific file (for testing purposes)
* ``-o OUTPUT, --output OUTPUT``: Output directory for NetCDF files (default: "../../outdata/lpj_guess" relative to input path)
* ``-v, --verbose``: Increase output verbosity
* ``--remap RES``: Remap output to a regular global grid using CDO. Specify either:
    * Resolution in degrees (e.g., 0.5, 1, 2)
    * Grid dimensions as XxY (e.g., 360x180 for 1° grid)
* ``--test {ifs_input}``: Test with specific file pattern (e.g., ifs_input.out)
* ``-j JOBS, --jobs JOBS``: Number of parallel jobs for outer parallelization (patterns). Default: 8
* ``--inner-jobs INNER_JOBS``: Number of parallel jobs for inner parallelization (within patterns). Default: 16
* ``--chunk-size CHUNK_SIZE``: Chunk size for processing data arrays. Default: 50000
* ``--pattern PATTERN``: Specific pattern to process (used internally for parallelization)

Example Commands
--------------

Basic Conversion
~~~~~~~~~~~~~~~

To convert all LPJ-GUESS output files in a directory:

.. code-block:: bash

    ./lpjg2nc.py -p /path/to/lpj_guess_runs/

This will:
1. Search for all run* folders in the specified path
2. Find .out files in each run's output directory
3. Convert them to NetCDF format
4. Save the output to "../../outdata/lpj_guess" relative to the input path

Verbose Output with Custom Output Directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more detailed progress information and a specific output directory:

.. code-block:: bash

    ./lpjg2nc.py -p /path/to/lpj_guess_runs/ -v -o /path/to/output/

Remapping to Regular Grid
~~~~~~~~~~~~~~~~~~~~~~~

To convert and remap output to a regular 1-degree global grid (360x180):

.. code-block:: bash

    ./lpjg2nc.py -p /path/to/lpj_guess_runs/ --remap 360x180

Alternatively, you can specify the resolution in degrees:

.. code-block:: bash

    ./lpjg2nc.py -p /path/to/lpj_guess_runs/ --remap 1

Testing with a Specific Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To test the conversion with just the ifs_input.out files:

.. code-block:: bash

    ./lpjg2nc.py -p /path/to/lpj_guess_runs/ --test ifs_input

Processing a Single File
~~~~~~~~~~~~~~~~~~~~~~

To process a specific .out file:

.. code-block:: bash

    ./lpjg2nc.py -p /path/to/lpj_guess_runs/ -f /path/to/specific/file.out

Fine-tuning Performance
~~~~~~~~~~~~~~~~~~~~~

To adjust parallelization settings for optimal performance:

.. code-block:: bash

    ./lpjg2nc.py -p /path/to/lpj_guess_runs/ -j 16 --inner-jobs 32 --chunk-size 100000

This increases:
- The number of output patterns processed in parallel to 16
- The number of variables processed in parallel within each pattern to 32
- The chunk size for data arrays to 100,000 (uses more memory but may be faster)

Output Files
-----------

The tool generates NetCDF files with the following naming pattern:
- For standard output: ``<pattern>.nc``
- For remapped output: ``<pattern>_<resolution>deg.nc`` or ``<pattern>_<dimensions>.nc``

Each NetCDF file contains:
- Variables from the original LPJ-GUESS output
- Proper coordinate information (latitude, longitude, time)
- Time dimension (if applicable)
- Metadata from the original files
