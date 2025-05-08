.. _examples:

Examples
========

Basic Data Conversion Example
----------------------------

This example demonstrates how to convert standard LPJ-GUESS output files to NetCDF format.

.. code-block:: bash

    # Convert all output patterns from an LPJ-GUESS run
    ./lpjg2nc.py -p /work/projects/lpj_guess/run_1950_2010/ -v
    
    # Results will be saved to ../../outdata/lpj_guess/ by default
    # Each output pattern will be converted to a separate NetCDF file

Example Output:

.. code-block:: text

    Found 5 unique output patterns across 128 run directories
    Processing output patterns: vegc.out, soilc.out, npp.out, gpp.out, lai.out
    Processing: vegc.out [1/5]
    Processing 12 variables across 3,246,752 data points...
    ...
    Successfully processed all 5 output patterns
    NetCDF files saved to: /work/projects/lpj_guess/outdata/lpj_guess/

Working with Sparse Datasets
---------------------------

This example shows how to process LPJ-GUESS data that's very sparse (many NaN values), which is common when working with land-only data on a global grid.

.. code-block:: bash

    # Process ifs_input.out files with verbose output to see NaN statistics
    ./lpjg2nc.py -p /work/climate/lpj_guess_run/ --test ifs_input -v
    
Example Output:

.. code-block:: text

    Found 128 ifs_input.out files
    Processing 8 variables across 8,950,165 data points...
    Successfully processed ifs_input.out files
    Output saved to: /work/climate/lpj_guess/outdata/lpj_guess/ifs_input.nc
    
    Data coverage analysis:
    📊 Data coverage: 27.58% valid data, 72.42% NaN values
    
    This level of sparsity is normal for land-only data on a global grid

Remapping to Regular Grids for Earth System Models
-------------------------------------------------

When preparing data for use in Earth System Models, it's often necessary to remap the output to a regular global grid. This example demonstrates how to create a 1° (360x180) grid.

.. code-block:: bash

    # Convert LPJ-GUESS output and remap to 1-degree global grid
    ./lpjg2nc.py -p /work/coupled_model/lpj_guess_data/ -v --remap 360x180
    
Example Output:

.. code-block:: text

    Processing: ifs_input.out
    ...
    Successfully processed ifs_input.out files
    Output saved to: /work/coupled_model/lpj_guess_data/outdata/lpj_guess/ifs_input.nc
    
    Remapping with CDO: cdo remapnn,/path/to/grid_360x180_global.txt ...
    Successfully remapped to 1.0° grid: /work/coupled_model/lpj_guess_data/outdata/lpj_guess/ifs_input_360x180.nc

Performance Optimization for Large Datasets
------------------------------------------

This example shows how to adjust parallel processing parameters for optimal performance with large datasets.

.. code-block:: bash

    # Process a large dataset with custom parallelization settings
    ./lpjg2nc.py -p /work/archive/lpj_guess_global/ -j 24 --inner-jobs 32 --chunk-size 100000
    
Example Output:

.. code-block:: text

    Using 24 parallel jobs for outer parallelization (patterns)
    Using 32 parallel jobs for inner parallelization (variables)
    Using chunk size of 100000 for data arrays
    
    Processing 8 patterns in parallel...
    Combined 52,483,456 data points from 512 files in 145.2 seconds
    Total processing time: 312.45 seconds (5.21 minutes)
    
    Performance: 12.5 variables/second (8x faster than sequential processing)

Extracting Data from a Specific Time Period
------------------------------------------

This example demonstrates how to extract data for a specific time period from LPJ-GUESS output.

.. code-block:: bash

    # First convert the output files to NetCDF
    ./lpjg2nc.py -p /work/project/lpj_guess/spinup_1850_2000/ -v
    
    # Then use CDO to extract a specific time period
    cdo seldate,2000-01-01,2000-12-31 /work/project/lpj_guess/outdata/lpj_guess/vegc.nc vegc_2000.nc
    
This two-step approach:
1. Converts the LPJ-GUESS output to NetCDF format with lpjg2nc2
2. Uses CDO to extract just the data for the year 2000
