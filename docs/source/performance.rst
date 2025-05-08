.. _performance:

Performance Optimization
======================

This page provides guidance on optimizing lpjg2nc2's performance for different environments and dataset sizes.

Parallelization Parameters
------------------------

lpjg2nc2 uses a two-level parallelization strategy:

1. **Outer Parallelization** (``-j`` or ``--jobs``)
   Controls how many output patterns are processed in parallel. For example, if you have multiple output types (vegc.out, soilc.out, etc.), this parameter determines how many are processed concurrently.

2. **Inner Parallelization** (``--inner-jobs``)
   Controls the number of parallel jobs for processing variables within each output pattern. This affects how many variables are processed simultaneously within a single file type.

Memory Usage Considerations
-------------------------

The memory usage of lpjg2nc2 is primarily affected by:

1. **Number of parallel jobs**: Higher values for ``-j`` and ``--inner-jobs`` will increase memory usage.
2. **Chunk size**: The ``--chunk-size`` parameter controls how many data points are processed at once.

Memory usage can be approximated as:

.. math::

   \text{Memory (GB)} \approx \text{jobs} \times \text{inner-jobs} \times \text{chunk-size} \times 8 \times 10^{-9}

Recommended Settings
------------------

Based on extensive testing, here are recommended settings for different environments:

+------------------------+------------------+------------------+----------------+
| Environment            | --jobs (-j)      | --inner-jobs     | --chunk-size   |
+========================+==================+==================+================+
| Desktop (16GB RAM)     | 4                | 8                | 25000          |
+------------------------+------------------+------------------+----------------+
| Workstation (32GB RAM) | 8                | 16               | 50000          |
+------------------------+------------------+------------------+----------------+
| HPC Node (128GB+ RAM)  | 8                | 64               | 75000          |
+------------------------+------------------+------------------+----------------+

Performance Benchmarks
--------------------

The following benchmarks show performance improvements from the vectorized implementation compared to the original point-by-point implementation:

+-----------------+-------------------+------------------+---------------+
| Dataset Size    | Original (vars/s) | Vectorized (v/s) | Speedup       |
+=================+===================+==================+===============+
| Small (~1GB)    | ~1.30             | ~8.5             | 6.5×          |
+-----------------+-------------------+------------------+---------------+
| Medium (~10GB)  | ~1.25             | ~25.5            | 20.4×         |
+-----------------+-------------------+------------------+---------------+
| Large (~100GB)  | ~0.90             | ~18.2            | 20.2×         |
+-----------------+-------------------+------------------+---------------+

Vectorization improvements are particularly significant for larger datasets. The optimizations include:

1. Replacing point-by-point operations with NumPy's vectorized operations
2. Optimizing time index creation with more efficient data structures
3. Using efficient hashing for coordinate matching
4. Chunking large arrays to balance memory usage and performance

Profiling the Conversion Process
------------------------------

When running with the verbose flag (``-v``), lpjg2nc2 provides timing information for different stages of the conversion process:

.. code-block:: text

    Processing variables: 100%|█████████████████████████| 8/8 [00:46<00:00, 5.87s/it]
    Processed all 8 variables, stored in dataset: ['co2', 'par', 'LW', 'precip', 'airT', 'soilT', 'surfW', 'deepW']
    Creating xarray dataset with 1D coordinates...
    Creating xarray dataset with complete grid expansion...
    Expanding from 24521 processed points to 88838 total grid points...
        Building optimized grid mapping with efficient hashing...
        Created mapping in 0.75 seconds
        Found 24501 matching points between processed and full grid
        Expanding variables to full grid...
    Preserved all variables in dataset: ['co2', 'par', 'LW', 'precip', 'airT', 'soilT', 'surfW', 'deepW']
    Coordinate processing completed in 50.05 seconds
        Grid expansion took 1.19 seconds
        Final grid has 88838 points with 24501 data points
        Expansion speed: 74525 points/sec
    
    ⏱️ Total processing time: 89.19 seconds (1.49 minutes)

This information can help identify bottlenecks and optimize settings for your specific dataset.
