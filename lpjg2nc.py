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
import multiprocessing
import subprocess
import json
import shutil
import psutil


def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / (1024 ** 3)
    return mem_gb


def get_system_memory():
    """Get system memory info in GB."""
    mem = psutil.virtual_memory()
    return {
        'total': mem.total / (1024 ** 3),
        'available': mem.available / (1024 ** 3),
        'used': mem.used / (1024 ** 3),
        'percent': mem.percent
    }


def format_eta(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"

# Import from our modules
from lpjg2nc.grid_utils import read_grid_information
from lpjg2nc.file_parser import find_out_files, detect_file_structure
from lpjg2nc.netcdf_converter import process_file
from lpjg2nc.count_nans import analyze_netcdf, print_short_summary
from lpjg2nc.cdo_interpolation import remap_to_regular_grid


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
        '--remap', type=str, metavar='RES',
        help='Remap output to a regular global grid using CDO. Specify either resolution in degrees (e.g., 0.5, 1, 2) or grid dimensions as XxY (e.g., 360x180 for 1° grid)'
    )
    parser.add_argument(
        '--test', type=str, choices=['ifs_input'],
        help='Test with specific file pattern (e.g., ifs_input.out)'
    )
    parser.add_argument(
        '-j', '--jobs', type=int, default=8,
        help='Number of parallel jobs for outer parallelization (patterns). Default (8) based on performance testing.'
    )
    parser.add_argument(
        '--inner-jobs', type=int, default=16,
        help='Number of parallel jobs for inner parallelization (within patterns). Default (16) based on performance testing.'
    )
    parser.add_argument(
        '--chunk-size', type=int, default=50000,
        help='Chunk size for processing data arrays. Larger values use more memory but may be faster. Default (50000) based on performance testing.'
    )
    parser.add_argument(
        '--pattern', type=str, default=None,
        help='Specific pattern to process (used internally for parallelization)'
    )
    return parser.parse_args()


def process_ifs_input_test(path, output_path, verbose=False, n_jobs=1, remap=None):
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
    
    # Analyze NaN values in the output file
    if output_file and os.path.exists(output_file):
        if verbose:
            print(f"\nAnalyzing NaN values in {output_file}...")
        # Use the external count_nans module for analysis
        nan_stats = analyze_netcdf(output_file, verbose=False, return_stats=True)
        
        if verbose:
            print(f"NaN analysis complete: {nan_stats['total_valid_pct']:.2f}% valid data")
    else:
        if verbose:
            print(f"\nSkipping NaN analysis - output file not found: {output_file}")
        nan_stats = None
    
    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time
    
    print(f"✅ Successfully processed ifs_input.out files")
    print(f"📁 Output saved to: {output_file}")
    
    # Display NaN statistics summary if available
    if nan_stats:
        print_short_summary(nan_stats)
    
    # Remap to regular grid if requested
    if remap and output_file:
        # Handle either resolution in degrees or grid dimensions format
        remapped_file = remap_to_regular_grid(output_file, remap, verbose=verbose)
        if remapped_file:
            # Format the grid description based on the remap parameter format
            if 'x' in str(remap).lower():
                print(f"📊 Created {remap} grid file: {remapped_file}")
            else:
                try:
                    resolution = float(remap)
                    print(f"📊 Created {resolution}° regular grid file: {remapped_file}")
                except ValueError:
                    print(f"📊 Created remapped grid file: {remapped_file}")
    
    print(f"⏱️ Total processing time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    
    return output_file



def run_subprocess(cmd):
    """Run a subprocess and return True if it succeeded."""
    try:
        # Execute the command and log in the background
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
            universal_newlines=True, shell=True
        )
        # Wait for process to complete
        stdout, stderr = process.communicate()
        
        # Check if process succeeded
        if process.returncode != 0:
            print(f"Error running command: {cmd}")
            print(stderr)
            return False
        return True
    except Exception as e:
        print(f"Exception running command {cmd}: {str(e)}")
        return False

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
            process_ifs_input_test(args.path, args.output, args.verbose, remap=args.remap)
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
        
        # Analyze NaN values in the output file
        if output_file and os.path.exists(output_file):
            if args.verbose:
                print(f"\nAnalyzing NaN values in {output_file}...")
            # Use the external count_nans module for analysis
            nan_stats = analyze_netcdf(output_file, verbose=False, return_stats=True)
        else:
            if args.verbose:
                print(f"\nSkipping NaN analysis - output file not found: {output_file}")
            nan_stats = None
        
        # Remap to regular grid if requested
        if args.remap and output_file:
            try:
                resolution = float(args.remap)
                if resolution <= 0:
                    print(f"⚠️ Invalid resolution: {args.remap}. Must be a positive number.")
                else:
                    remapped_file = remap_to_regular_grid(output_file, resolution, verbose=args.verbose)
                    if remapped_file:
                        print(f"📊 Created {resolution}° regular grid file: {remapped_file}")
            except ValueError:
                print(f"⚠️ Invalid resolution: {args.remap}. Must be a number.")
        
        total_end_time = time.time()
        total_elapsed = total_end_time - total_start_time
        
        print(f"✅ Successfully processed: {args.file}")
        print(f"📁 Output saved to: {output_file}")
        
        # Display NaN statistics summary if available
        if nan_stats:
            print_short_summary(nan_stats)
            
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
        
        # Specific pattern processing (used when this script is run in parallel mode)
        if args.pattern:
            pattern_name = args.pattern
            if pattern_name in out_files:
                file_paths = out_files[pattern_name]
                # Process just this pattern
                output_file = process_file(file_paths, args.output, grid_info, args.verbose, 
                                          inner_jobs=args.inner_jobs, chunk_size=args.chunk_size)
                if output_file:
                    print(f"Successfully processed: {pattern_name} -> {os.path.basename(output_file)}")
                    return 0
                else:
                    print(f"Failed to process: {pattern_name}")
                    return 1
            else:
                print(f"Pattern not found: {pattern_name}")
                return 1
        
        # Determine number of parallel jobs to use
        n_jobs = args.jobs
        if n_jobs <= 0:
            # Auto-detect based on system resources
            n_jobs = max(1, min(multiprocessing.cpu_count() - 1, 8))  # Use N-1 cores up to max 8
            
        print(f"Processing {len(out_files)} output patterns with {n_jobs} parallel jobs")
        
        # Get a list of all patterns
        file_items = list(out_files.keys())
        total_patterns = len(file_items)
        
        # Process files in parallel using subprocesses
        if n_jobs > 1:
            # Prepare the command template
            script_path = os.path.abspath(sys.argv[0])
            base_cmd = f"{sys.executable} {script_path} -p {args.path} -o {args.output}"
            if args.verbose:
                base_cmd += " -v"
            if args.inner_jobs > 0:
                base_cmd += f" --inner-jobs {args.inner_jobs}"
            if args.chunk_size > 0:
                base_cmd += f" --chunk-size {args.chunk_size}"
                
            # Start processing patterns in parallel
            sys_mem = get_system_memory()
            print(f"Starting parallel processing with {n_jobs} workers")
            print(f"System memory: {sys_mem['used']:.1f}GB used / {sys_mem['total']:.1f}GB total ({sys_mem['percent']:.0f}%)")
            
            running_procs = {}
            completed = set()
            errors = []
            error_details = {}  # Store error messages for each failed pattern
            pattern_idx = 0
            start_time = time.time()
            last_status_time = start_time
            
            # Create log file for errors
            log_file = os.path.join(args.output, 'conversion_errors.log')
            os.makedirs(args.output, exist_ok=True)
            
            # Process patterns in batches
            while pattern_idx < len(file_items) or running_procs:
                # Start new processes up to the job limit
                while len(running_procs) < n_jobs and pattern_idx < len(file_items):
                    pattern = file_items[pattern_idx]
                    cmd = f"{base_cmd} --pattern '{pattern}'"
                    
                    # Start the subprocess
                    process = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                        universal_newlines=True, shell=True
                    )
                    
                    running_procs[pattern] = process
                    pattern_idx += 1
                
                # Check for completed processes
                still_running = {}
                for pattern, proc in running_procs.items():
                    returncode = proc.poll()
                    if returncode is not None:  # Process has finished
                        stdout, stderr = proc.communicate()
                        if returncode != 0:
                            errors.append(pattern)
                            error_msg = stderr.strip() if stderr else stdout.strip()
                            error_details[pattern] = {
                                'returncode': returncode,
                                'stderr': error_msg[:500]  # Limit error message length
                            }
                            # Log error immediately
                            print(f"\n❌ ERROR in {pattern}: exit code {returncode}")
                            if error_msg:
                                print(f"   {error_msg[:200]}")
                            # Write to log file
                            with open(log_file, 'a') as f:
                                f.write(f"\n{'='*80}\n")
                                f.write(f"Pattern: {pattern}\n")
                                f.write(f"Exit code: {returncode}\n")
                                f.write(f"Error output:\n{error_msg}\n")
                        completed.add(pattern)
                    else:
                        still_running[pattern] = proc
                
                running_procs = still_running
                
                # Status update every 5 seconds
                current_time = time.time()
                if current_time - last_status_time >= 5.0:
                    elapsed = current_time - start_time
                    done = len(completed)
                    if done > 0:
                        rate = done / elapsed  # patterns per second
                        remaining = total_patterns - done
                        eta_seconds = remaining / rate if rate > 0 else 0
                        eta_str = format_eta(eta_seconds)
                    else:
                        eta_str = "calculating..."
                    
                    sys_mem = get_system_memory()
                    print(f"[{done}/{total_patterns}] Done | {len(running_procs)} running | "
                          f"ETA: {eta_str} | Mem: {sys_mem['used']:.1f}/{sys_mem['total']:.1f}GB ({sys_mem['percent']:.0f}%) | "
                          f"Errors: {len(errors)}")
                    last_status_time = current_time
                
                # Brief pause
                if running_procs:
                    time.sleep(0.5)
            
            # Calculate success rate
            success_count = len(completed) - len(errors)
            error_count = len(errors)
            print(f"\nCompleted {len(completed)} out of {total_patterns} patterns in parallel mode")
            
            # All done in parallel mode
            total_end_time = time.time()
            total_elapsed = total_end_time - total_start_time
            
            if error_count > 0:
                print(f"\n⚠️  {error_count} patterns failed during conversion:")
                for i, pattern in enumerate(errors[:10], 1):  # Show first 10
                    print(f"   {i}. {pattern}")
                if error_count > 10:
                    print(f"   ... and {error_count - 10} more")
                print(f"\n📋 Full error details saved to: {log_file}")
            
            print(f"\n✅ Successfully processed {success_count}/{total_patterns} output patterns using {n_jobs} parallel jobs")
            if error_count > 0:
                print(f"❌ Failed: {error_count} patterns (see {log_file} for details)")
            print(f"📁 NetCDF files saved to: {args.output}")
            print(f"⏱️ Total processing time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
            
            return 0 if error_count == 0 else 1
        
        # Sequential processing as fallback or if n_jobs=1
        if n_jobs == 1:
            processed_files = []
            # Process each file pattern sequentially
            for i, (file_name, file_paths) in enumerate(file_items):
                current_pattern = i + 1
                output_file = process_file(file_paths, args.output, grid_info, args.verbose,
                                          current_pattern=current_pattern, total_patterns=total_patterns,
                                          inner_jobs=args.inner_jobs, chunk_size=args.chunk_size)
                if output_file:
                    processed_files.append(output_file)
        
        total_end_time = time.time()
        total_elapsed = total_end_time - total_start_time
        
        print(f"\n✅ Successfully processed {len(processed_files)} output patterns using {n_jobs} parallel jobs")
        print(f"📁 NetCDF files saved to: {args.output}")
        print(f"⏱️ Total processing time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
        
if __name__ == "__main__":
    main()
