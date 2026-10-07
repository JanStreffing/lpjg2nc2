# LPJ-GUESS to NetCDF Converter (lpjg2nc2)

[![PyPI version](https://badge.fury.io/py/lpjg2nc2.svg)](https://badge.fury.io/py/lpjg2nc2)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Documentation Status](https://readthedocs.org/projects/lpjg2nc2/badge/?version=latest)](https://lpjg2nc2.readthedocs.io/en/latest/?badge=latest)
[![Documentation Build](https://img.shields.io/badge/docs%20build-passing-brightgreen)](https://lpjg2nc2.readthedocs.io/)

lpjg2nc2 is a powerful tool for converting LPJ-GUESS output files (.out) to NetCDF format. It efficiently searches for run* folders in a given path, finds .out files inside their output directories, and converts them to the widely-used NetCDF format for easier analysis and visualization.

## 🚀 Key Features

* **Efficient Processing**: Convert LPJ-GUESS .out files to NetCDF with proper coordinates and metadata
* **High Performance**: Vectorized implementation delivers up to 20× speedup compared to point-by-point processing
* **Parallel Processing**: Two-level parallelization optimizes conversion on multi-core systems
* **3D Depth-Layer Support**: Full support for 3D files with multiple soil depth layers (e.g., mrsll, mrsol, tsl)
* **Flexible Configuration**: Control various aspects of the conversion process
* **Grid Handling**: Process both 2D and 3D data (with depth levels), preserving spatial coordinates
* **Time Dimensions**: Handle time series data (years, months, and days) with proper cftime support for historical dates
* **Error Logging**: Comprehensive error tracking and logging during parallel processing
* **Grid Remapping**: Option to remap to regular global grids using CDO
* **Variable Metadata**: `long_name` and `units` attributes taken from LPJ-GUESS's own output definitions (`modules/*.cpp`)
* **Data Analysis**: Built-in analysis of NaN values to understand data sparsity in global datasets
* **Test Mode**: Option to process specific files for testing

## 📦 Installation

### From PyPI (recommended)

```bash
pip install lpjg2nc2
```

### From Source

```bash
git clone https://github.com/JanStreffing/lpjg2nc2.git
cd lpjg2nc2
pip install -e .
```

## 📋 Requirements

* Python 3.8+
* Dependencies (automatically installed):
  * numpy>=1.20.0
  * pandas>=1.3.0
  * xarray>=0.19.0
  * netCDF4>=1.5.7
  * tqdm>=4.61.0
* Optional: Climate Data Operators (CDO) for remapping capabilities

## 🔍 Usage

### Basic Conversion

```bash
lpjg2nc -p /path/to/lpj_guess_runs/
```

This will:
1. Search for all run* folders in the specified path
2. Find .out files in each run's output directory
3. Convert them to NetCDF format
4. Save the output to "../../outdata/lpj_guess" relative to the input path

### Verbose Output with Custom Output Directory

```bash
lpjg2nc -p /path/to/lpj_guess_runs/ -v -o /path/to/output/
```

### Remapping to Regular Grid

```bash
lpjg2nc -p /path/to/lpj_guess_runs/ --remap
# same as the default: conservative remapping to a 1 degree grid
lpjg2nc -p /path/to/lpj_guess_runs/ --remap remapcon,r360x180
# any CDO remap operator and grid
lpjg2nc -p /path/to/lpj_guess_runs/ --remap remapnn,global_1
```

After a successful remap the unstructured NetCDF file is deleted, unless
`--nc_keep_unstruc` is given.

### Converting a Single Variable

```bash
lpjg2nc -p /path/to/lpj_guess_runs/ -f netAtmosLandCO2Flux_monthly.out -v
```

All files of that name under `<path>/run*/output` are combined into one global file.

## 🛠️ Command Line Arguments

### Required Arguments
* `-p PATH, --path PATH`: Path to the directory containing run* folders

### Optional Arguments
* `-f FILE, --file FILE`: Process a single variable globally: a basename (e.g. `netAtmosLandCO2Flux_monthly.out`) or a path to one such file; all matching files under `<path>/run*/output` are combined
* `-g GRIDS, --grids GRIDS`: Path to a `grids.nc` file providing the grid geometry (default: `<path>/grids.nc`, if it exists)
* `-o OUTPUT, --output OUTPUT`: Output directory for NetCDF files (default: `../../outdata/lpj_guess` relative to the input path)
* `-v, --verbose`: Increase output verbosity
* `--remap [OPERATOR,GRID]`: Remap output using CDO, e.g. `remapcon,r360x180` (default if no value is given) or `remapnn,global_1`
* `--nc_keep_unstruc`: Keep the unstructured (pre-remap) NetCDF file alongside the remapped one (only with `--remap`)
* `--nc_rm_ascii`: Delete the `.out` input files of a variable once its NetCDF file was written successfully (and remapped, if `--remap` is given); by default they are kept
* `--test {ifs_input}`: Test with specific file pattern (e.g., ifs_input.out)
* `-j JOBS, --jobs JOBS`: Number of parallel jobs for outer parallelization (patterns), default: 8
* `--pattern PATTERN`: Specific pattern to process (internal use for parallelization)

## 📚 Documentation

For complete documentation, visit [lpjg2nc2.readthedocs.io](https://lpjg2nc2.readthedocs.io/).

## 📜 License

This project is licensed under the GNU General Public License v3 (GPLv3) - see the [license](license) file for details.
