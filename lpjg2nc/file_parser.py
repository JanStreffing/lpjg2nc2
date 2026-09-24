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
    Find all .out files anywhere under base_path, grouped by basename.

    Handles any layout by searching recursively: files directly in
    base_path (flat/already-combined mode), the traditional
    base_path/run*/output/*.out layout (parallel domain-decomposed runs),
    base_path/run*/*.out (no 'output' subfolder), or multiple date-range
    segments each with their own run dir, e.g.
    base_path/<date_range>/run*/*.out. Every match for a given pattern
    name is combined into one entry, regardless of how deep it sits.

    Parameters
    ----------
    base_path : str
        Path to the directory to search (may contain run* folders, at any
        nesting depth, or .out files directly).

    Returns
    -------
    dict
        Dictionary with file basename as key and list of file paths as value.

    Raises
    ------
    ValueError
        If the same basename occurs more than once within one run* folder
        (e.g. both run1/x.out and run1/output/x.out), or more than once
        outside any run* folder. Such files would be combined as duplicate
        data.
    """
    all_out_files = {}
    seen = {}  # (run dir or base_path, basename) -> first path found
    duplicates = []
    for out_file in sorted(glob.glob(os.path.join(base_path, '**', '*.out'), recursive=True)):
        file_basename = os.path.basename(out_file)
        key = (_run_dir(out_file, base_path), file_basename)
        if key in seen:
            duplicates.append((seen[key], out_file))
        else:
            seen[key] = out_file
        all_out_files.setdefault(file_basename, []).append(out_file)
    if duplicates:
        raise ValueError(
            f"Found {len(duplicates)} .out file(s) with the same basename in the same "
            f"run directory under {base_path}; remove the extra copies:\n"
            + "\n".join(f"  {a}\n  {b}" for a, b in duplicates))
    return all_out_files


def _run_dir(out_file, base_path):
    """Innermost run<N> folder containing out_file, or base_path if none."""
    rel_dirs = os.path.relpath(os.path.dirname(out_file), base_path).split(os.sep)
    for i in range(len(rel_dirs), 0, -1):
        if re.fullmatch(r'run\d+', rel_dirs[i - 1]):
            return os.path.join(base_path, *rel_dirs[:i])
    return base_path

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
    has_month = 'Month' in columns or 'Mth' in columns
    
    # Skip the first columns (Lon, Lat, Year, Day/Month if present)
    start_idx = 3  # Default: Lon, Lat, Year
    if has_day:
        start_idx = 4  # Lon, Lat, Year, Day
    elif has_month:
        start_idx = 4  # Lon, Lat, Year, Month/Mth
    
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
        df = pd.read_csv(file_path, sep=r'\s+', comment='#')
        all_data.append(df)
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df
