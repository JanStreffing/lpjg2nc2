#!/usr/bin/env python3
"""
Combine LPJ-GUESS output files from all run folders.
Reads from: <exp_path>/run_YYYYMMDD-YYYYMMDD/work/runX/output/
Outputs to: <exp_path>/outdata/lpj_guess/out/ (combined .out files, split by year)
           <exp_path>/outdata/lpj_guess/nc/  (NetCDF files, via lpjg2nc2)

Usage:
    python combine_runs.py <run_path>
    
Example:
    python combine_runs.py /work/bb1469/a270092/runtime/awiesm3-develop/LR_spinup_test_01/run_13500101-13691231/work
"""

import os
import sys
import glob
import re
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Combine LPJ-GUESS output files from run folders, split by year.'
    )
    parser.add_argument(
        'run_path', type=str,
        help='Path to run_YYYYMMDD-YYYYMMDD/work directory containing runX/output folders'
    )
    parser.add_argument(
        '-j', '--jobs', type=int, default=8,
        help='Number of parallel workers (default: 8, lower to avoid file handle exhaustion)'
    )
    return parser.parse_args()


def derive_paths(run_path):
    """
    Derive input/output paths from the run_path.
    
    Input:  <exp>/run_YYYYMMDD-YYYYMMDD/work/runX/output/
    Output: <exp>/outdata/lpj_guess/out/
            <exp>/outdata/lpj_guess/nc/
    """
    run_path = os.path.abspath(run_path)
    
    # Extract date range from parent directory name (run_YYYYMMDD-YYYYMMDD)
    # run_path is .../run_YYYYMMDD-YYYYMMDD/work
    parent = os.path.dirname(run_path)  # .../run_YYYYMMDD-YYYYMMDD
    run_dir_name = os.path.basename(parent)  # run_YYYYMMDD-YYYYMMDD
    
    # Extract date range
    match = re.match(r'run_(\d{8}-\d{8})', run_dir_name)
    if match:
        date_range = match.group(1)
    else:
        raise ValueError(f"Could not extract date range from {run_dir_name}")
    
    # Derive experiment base path
    exp_path = os.path.dirname(parent)  # <exp>
    
    # Define output paths
    out_dir = os.path.join(exp_path, 'outdata', 'lpj_guess', 'out')
    nc_dir = os.path.join(exp_path, 'outdata', 'lpj_guess', 'nc')
    
    return {
        'run_path': run_path,
        'date_range': date_range,
        'exp_path': exp_path,
        'out_dir': out_dir,
        'nc_dir': nc_dir
    }


def get_run_folders(run_path):
    """Get all runX/output folders sorted numerically."""
    pattern = os.path.join(run_path, 'run[0-9]*', 'output')
    run_dirs = glob.glob(pattern)
    # Sort numerically by extracting the run number
    run_dirs = sorted(run_dirs, key=lambda x: int(os.path.basename(os.path.dirname(x)).replace('run', '')))
    return run_dirs


def get_file_patterns(run_output_dir):
    """Get all unique .out file patterns."""
    pattern = os.path.join(run_output_dir, '*.out')
    files = glob.glob(pattern)
    # Extract base names
    patterns = set(os.path.basename(f) for f in files)
    return patterns

def combine_files(run_folders, pattern_name, output_dir):
    """Combine files from all run folders for a given pattern, split by year.
    
    Uses streaming approach: writes directly to year-specific files to avoid memory issues.
    """
    header = None
    year_col_idx = None
    year_files = {}  # year -> file handle
    base_name = pattern_name.replace('.out', '')
    
    try:
        for run_output_dir in run_folders:
            input_file = os.path.join(run_output_dir, pattern_name)
            if not os.path.exists(input_file):
                continue
                
            with open(input_file, 'r') as f:
                for i, line in enumerate(f):
                    if i == 0:
                        if header is None:
                            header = line
                            # Find Year column index
                            cols = line.split()
                            try:
                                year_col_idx = cols.index('Year')
                            except ValueError:
                                year_col_idx = None
                        continue
                    
                    if year_col_idx is not None:
                        # Extract year from this line
                        parts = line.split()
                        if len(parts) > year_col_idx:
                            year = parts[year_col_idx]
                            
                            # Open file for this year if not already open
                            if year not in year_files:
                                output_file = os.path.join(output_dir, f'{base_name}_{year}.out')
                                year_files[year] = open(output_file, 'w')
                                year_files[year].write(header)
                            
                            year_files[year].write(line)
                    else:
                        # No year column, write to single file
                        if 'all' not in year_files:
                            output_file = os.path.join(output_dir, pattern_name)
                            year_files['all'] = open(output_file, 'w')
                            year_files['all'].write(header)
                        year_files['all'].write(line)
    finally:
        # Close all file handles
        for fh in year_files.values():
            fh.close()
    
    return list(year_files.keys())


def process_pattern(args):
    """Wrapper for parallel processing."""
    run_folders, pattern, output_dir = args
    combine_files(run_folders, pattern, output_dir)
    return pattern


def main():
    args = parse_args()
    
    # Derive paths
    paths = derive_paths(args.run_path)
    print(f"Run path: {paths['run_path']}")
    print(f"Date range: {paths['date_range']}")
    print(f"Output .out dir: {paths['out_dir']}")
    print(f"Output .nc dir: {paths['nc_dir']}")
    
    # Create output directories
    os.makedirs(paths['out_dir'], exist_ok=True)
    os.makedirs(paths['nc_dir'], exist_ok=True)
    
    # Get run folders
    run_folders = get_run_folders(paths['run_path'])
    if not run_folders:
        print(f"No run folders found in {paths['run_path']}")
        sys.exit(1)
    print(f"\nFound {len(run_folders)} run folders")
    
    # Get all file patterns from first run folder
    patterns = get_file_patterns(run_folders[0])
    if not patterns:
        print(f"No .out files found in {run_folders[0]}")
        sys.exit(1)
    print(f"Found {len(patterns)} .out file patterns")
    
    # Prepare arguments for parallel processing
    n_workers = min(multiprocessing.cpu_count(), args.jobs)
    print(f"\nCombining files into {paths['out_dir']} using {n_workers} workers")
    
    args_list = [(run_folders, p, paths['out_dir']) for p in sorted(patterns)]
    
    # Process in parallel with progress bar
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_pattern, a): a[1] for a in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Combining files"):
            future.result()  # Raise any exceptions
    
    print(f"\n✅ Successfully combined {len(patterns)} file patterns from {len(run_folders)} run folders")
    print(f"📁 Combined .out files saved to: {paths['out_dir']}")
    print(f"📁 NetCDF output directory: {paths['nc_dir']} (run lpjg2nc separately)")


if __name__ == "__main__":
    main()
