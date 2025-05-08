.. _installation:

Installation
===========

Requirements
-----------

lpjg2nc2 requires:

* Python 3.8 or above
* NumPy
* pandas
* xarray
* netCDF4
* CDO (Climate Data Operators) - for grid remapping functionality

Basic Installation
-----------------

You can install lpjg2nc2 directly from the GitHub repository:

.. code-block:: bash

    git clone https://github.com/JanStreffing/lpjg2nc2.git
    cd lpjg2nc2
    pip install -e .

This will install the package in development mode, so any changes you make to the code will be immediately available.

Installation with Conda
----------------------

For HPC environments or systems where you prefer to use conda, you can create a dedicated environment:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/JanStreffing/lpjg2nc2.git
    cd lpjg2nc2
    
    # Create and activate a conda environment
    conda create -n lpjg2nc2 python=3.9 numpy pandas xarray netcdf4 tqdm
    conda activate lpjg2nc2
    
    # Install CDO (Climate Data Operators) - required for remapping
    conda install -c conda-forge cdo
    
    # Install the package in development mode
    pip install -e .

Verifying the Installation
-------------------------

After installation, you can verify that lpjg2nc2 is working correctly by running:

.. code-block:: bash

    python -c "import lpjg2nc; print(lpjg2nc.__version__)"

You can also check the help message of the command-line tool:

.. code-block:: bash

    ./lpjg2nc.py -h

This should display the help message with all available command-line options.
