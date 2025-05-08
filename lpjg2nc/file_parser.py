#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File parsing utilities for LPJ-GUESS to NetCDF conversion.
"""

import os
import glob
import pandas as pd
import numpy as np
import re

def find_out_files(base_path):
    """
    Find all .out files in run*/output folders.
    
    Parameters
    ----------
    base_path : str
        Path to the directory containing run* folders.
        
    Returns
    -------
    dict
        Dictionary with file basename as key and list of file paths as value.
    """
    run_dirs = glob.glob(os.path.join(base_path, 'run*'))
    all_out_files = {}
    
    for run_dir in run_dirs:
        output_dir = os.path.join(run_dir, 'output')
        if os.path.isdir(output_dir):
            out_files = glob.glob(os.path.join(output_dir, '*.out'))
            for out_file in out_files:
                # Use the filename as the key, without the path
                file_basename = os.path.basename(out_file)
                if file_basename not in all_out_files:
                    all_out_files[file_basename] = []
                all_out_files[file_basename].append(out_file)
    
    return all_out_files

def detect_file_structure(file_path):
    """
    Detect the structure of the .out file (2D or 3D).
    
    Parameters
    ----------
    file_path : str
        Path to the .out file.
        
    Returns
    -------
    dict
        Dictionary with file structure information.
    """
    with open(file_path, 'r') as f:
        header = f.readline().strip()
    
    columns = [col.strip() for col in header.split()]
    
    # Check for depth columns
    depth_cols = [col for col in columns if col.startswith('Depth')]
    has_day = 'Day' in columns
    
    # Skip the first columns (Lon, Lat, Year, Day if present)
    start_idx = 3  # Default: Lon, Lat, Year
    if has_day:
        start_idx = 4  # Lon, Lat, Year, Day
    
    # Get the variable columns (all columns after the fixed ones)
    if depth_cols:
        # This is a 3D file with depth levels
        var_cols = columns[start_idx:]
        is_3d = True
    else:
        # This is a 2D file
        var_cols = columns[start_idx:]
        is_3d = False
    
    return {
        'columns': columns,
        'is_3d': is_3d,
        'has_day': has_day,
        'var_cols': var_cols,
        'depth_cols': depth_cols
    }

def extract_depths(depth_cols):
    """
    Extract depth values from depth column names.
    
    Parameters
    ----------
    depth_cols : list
        List of depth column names.
        
    Returns
    -------
    numpy.ndarray
        Array of depth values.
    """
    depths = []
    for col in depth_cols:
        match = re.search(r'Depth(\d+\.?\d*)', col)
        if match:
            depths.append(float(match.group(1)))
    return np.array(depths)

def read_and_combine_files(file_paths):
    """
    Read and combine multiple .out files.
    
    Parameters
    ----------
    file_paths : list
        List of paths to .out files.
        
    Returns
    -------
    pandas.DataFrame
        Combined data from all files.
    """
    all_data = []
    
    for file_path in file_paths:
        # Read the data
        df = pd.read_csv(file_path, delim_whitespace=True, comment='#')
        all_data.append(df)
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df
