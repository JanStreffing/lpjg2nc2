#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert LPJ-GUESS output files (.out) to NetCDF format.

This script searches for all run* folders in a given path, finds .out files 
inside their output folders, and converts them to NetCDF format.
"""

import os
import sys
import glob
import time
import argparse
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import re
from tqdm import tqdm

# Import from our modules
from grid_utils import read_grid_information
from file_parser import find_out_files, detect_file_structure
from netcdf_converter import process_file


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert LPJ-GUESS output files (.out) to NetCDF format.'
    )
    parser.add_argument(
        '-p', '--path', type=str, required=True,
        help='Path to the directory containing run* folders'
    )
    parser.add_argument(
        '-f', '--file', type=str,
        help='Specific file to process (for testing)'
    )
    parser.add_argument(
        '-o', '--output', type=str, default=None,
        help='Output directory for NetCDF files (default: "../../outdata/lpj_guess" relative to input path)'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Increase output verbosity'
    )
    parser.add_argument(
        '--test', type=str, choices=['ifs_input'],
        help='Test with specific file pattern (e.g., ifs_input.out)'
    )
    return parser.parse_args()




















def process_ifs_input_test(path, output_path, verbose=False):
    """Process ifs_input.out files as a test case."""
    total_start_time = time.time()
    
    print("Starting ifs_input.out test case...")
    
    # Find all ifs_input.out files
    if verbose:
        print("Searching for ifs_input.out files...")
    out_files = find_out_files(path)
    if 'ifs_input.out' not in out_files or not out_files['ifs_input.out']:
        print(f"No ifs_input.out files found in {path}/run*/output")
        sys.exit(1)
    
    if verbose:
        print(f"Found {len(out_files['ifs_input.out'])} ifs_input.out files")
    
    # Read grid information from grids.nc
    if verbose:
        print(f"Reading grid information from {path}/grids.nc...")
    grid_info = read_grid_information(path)
    if grid_info and verbose:
        print(f"Found {len(grid_info['lat'])} unique latitudes and {len(grid_info['lon'])} unique longitudes")
    
    # Process the ifs_input.out files
    output_file = process_file(out_files['ifs_input.out'], output_path, grid_info, verbose)
    
    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time
    
    print(f"✅ Successfully processed ifs_input.out files")
    print(f"📁 Output saved to: {output_file}")
    print(f"⏱️ Total processing time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    
    return output_file


def main():
    """Main function."""
    total_start_time = time.time()
    
    args = parse_args()
    
    # Set default output path if not specified
    if args.output is None:
        # Use "../../outdata/lpj_guess" relative to the input path
        input_path = os.path.abspath(args.path)
        output_path = os.path.normpath(os.path.join(input_path, "../../outdata/lpj_guess"))
        args.output = output_path
        if args.verbose:
            print(f"Using default output path: {args.output}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Check if we're in test mode
    if args.test:
        if args.test == 'ifs_input':
            process_ifs_input_test(args.path, args.output, args.verbose)
            return
    
    if args.file:
        # Process a specific file
        if not os.path.isfile(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        
        # Read grid information if available
        if args.verbose:
            print(f"Reading grid information from {args.path}/grids.nc...")
        grid_info = read_grid_information(args.path)
        
        output_file = process_file([args.file], args.output, grid_info, args.verbose)
        
        total_end_time = time.time()
        total_elapsed = total_end_time - total_start_time
        
        print(f"✅ Successfully processed: {args.file}")
        print(f"📁 Output saved to: {output_file}")
        print(f"⏱️ Total processing time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    else:
        # Find and process all .out files in run*/output folders
        if args.verbose:
            print(f"Searching for .out files in {args.path}/run*/output...")
        out_files = find_out_files(args.path)
        
        if not out_files:
            print(f"No .out files found in {args.path}/run*/output")
            sys.exit(1)
        
        # Read grid information if available
        if args.verbose:
            print(f"Reading grid information from {args.path}/grids.nc...")
        grid_info = read_grid_information(args.path)
        if grid_info and args.verbose:
            print(f"Found {len(grid_info['lat'])} unique latitudes and {len(grid_info['lon'])} unique longitudes")
        
        processed_files = []
        # Show progress bar for processing multiple files
        file_items = list(out_files.items())
        for file_name, file_paths in tqdm(file_items, desc="Processing file patterns", disable=not args.verbose):
            if args.verbose:
                print(f"Processing {len(file_paths)} files for pattern: {file_name}")
            
            output_file = process_file(file_paths, args.output, grid_info, args.verbose)
            processed_files.append(output_file)
        
        total_end_time = time.time()
        total_elapsed = total_end_time - total_start_time
        
        print(f"✅ Successfully processed {len(processed_files)} output patterns")
        print(f"📁 NetCDF files saved to: {args.output}")
        print(f"⏱️ Total processing time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")


if __name__ == "__main__":
    main()
