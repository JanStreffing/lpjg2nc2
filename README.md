# LPJ-GUESS to NetCDF Converter

This script converts LPJ-GUESS output files (.out) to NetCDF format.

## Features

- Finds and processes all `.out` files in `run*/output` folders
- Creates a single NetCDF file per `.out` file pattern
- Handles both 2D and 3D data (for files with depth levels)
- Preserves spatial coordinates (longitude, latitude)
- Handles time dimensions (years and days, if present)
- Includes a test mode for processing specific files

## Requirements

- Python 3.6+
- Dependencies: pandas, numpy, xarray, netCDF4

## Usage

### Converting all files in a path

```bash
./lpjg2nc.py -p /path/to/runs -o ./netcdf_output -v
```

### Testing with a specific file

```bash
./lpjg2nc.py -p /path/to/runs -f /path/to/runs/run1/output/somefile.out -o ./netcdf_output -v
```

## Command Line Arguments

- `-p`, `--path`: Path to the directory containing run* folders (required)
- `-f`, `--file`: Specific file to process (for testing)
- `-o`, `--output`: Output directory for NetCDF files (default: ./netcdf_output)
- `-v`, `--verbose`: Increase output verbosity

## Example

```bash
./lpjg2nc.py -p /work/bb1469/a270092/runtime/awiesm3-v3.4/AWIESM3_NTest141_LPJG_RESTART/run_20010101-20011231/work -o ./netcdf -v
```
