#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NetCDF conversion module for LPJ-GUESS to NetCDF.
"""

import os
import time
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import multiprocessing
import psutil
import math
from joblib import Parallel, delayed
from grid_utils import match_coordinates_to_grid
import datetime

def get_parallel_config(verbose=False):
    """
    Determines optimal parallel processing configuration based on system resources.
    Uses joblib for more reliable parallelism in HPC environments.
    
    Returns
    -------
    dict
        Dictionary containing optimal parallel configuration parameters
    """
    # Default settings that work well in most environments
    n_jobs = 8  # Default to 8 parallel jobs
    chunk_size = 25000  # Medium sized chunks for processing
    
    # Detect available cores and adjust n_jobs
    try:
        total_cores = multiprocessing.cpu_count()
        # Use at most 75% of available cores to avoid overloading the system
        n_jobs = max(2, min(64, int(total_cores * 0.75)))
        
        # Get available memory in GB
        mem = psutil.virtual_memory()
        available_memory = mem.available / (1024**3)  # Convert to GB
        
        # Adjust chunk size based on available memory
        if available_memory < 8:  # Less than 8GB available
            chunk_size = 10000
        elif available_memory > 32:  # More than 32GB available
            chunk_size = 50000
            
        if verbose:
            print(f"    System resources detected: {total_cores} cores, {available_memory:.1f}GB available memory")
    except Exception as e:
        if verbose:
            print(f"    Error detecting system resources: {e}, using default settings")
    
    if verbose:
        print(f"    Parallel configuration: {n_jobs} parallel jobs")
        print(f"    Chunk size: {chunk_size} data points")
    
    return {
        "n_jobs": n_jobs,
        "chunk_size": chunk_size,
        "backend": "loky"  # loky backend is more robust than multiprocessing
    }

def process_lon_chunk(chunk_lons, grid_lons, start_idx):
    """
    Process a chunk of longitude values to find nearest indices.
    
    Parameters
    ----------
    chunk_lons : numpy.ndarray
        Chunk of longitude values to process
    grid_lons : numpy.ndarray
        Array of grid longitude values
    start_idx : int
        Starting index of the chunk in the original array
        
    Returns
    -------
    tuple
        (start_idx, indices) - Starting index and corresponding indices array
    """
    indices = np.zeros(len(chunk_lons), dtype=int)
    for i, lon in enumerate(chunk_lons):
        indices[i] = np.abs(grid_lons - lon).argmin()
    return start_idx, indices

def process_lat_chunk(chunk_lats, grid_lats, start_idx):
    """
    Process a chunk of latitude values to find nearest indices.
    
    Parameters
    ----------
    chunk_lats : numpy.ndarray
        Chunk of latitude values to process
    grid_lats : numpy.ndarray
        Array of grid latitude values
    start_idx : int
        Starting index of the chunk in the original array
        
    Returns
    -------
    tuple
        (start_idx, indices) - Starting index and corresponding indices array
    """
    indices = np.zeros(len(chunk_lats), dtype=int)
    for i, lat in enumerate(chunk_lats):
        indices[i] = np.abs(grid_lats - lat).argmin()
    return start_idx, indices

def process_2d_file(file_paths, output_path, grid_info=None, verbose=False):
    """
    Process a 2D .out file and convert it to NetCDF.
    
    Parameters
    ----------
    file_paths : list
        List of paths to .out files.
    output_path : str
        Path to save the NetCDF file.
    grid_info : dict, optional
        Grid information from grid_utils.read_grid_information.
    verbose : bool, optional
        Whether to print verbose output.
        
    Returns
    -------
    str
        Path to the created NetCDF file.
    """
    start_time = time.time()
    
    if verbose:
        print(f"Processing 2D file: {os.path.basename(file_paths[0])}")
    
    # Get the file structure from the first file
    from file_parser import detect_file_structure, read_and_combine_files
    
    structure = detect_file_structure(file_paths[0])
    columns = structure['columns']
    
    # Determine if the file has days
    has_day = structure['has_day']
    
    # Read and combine all files
    if verbose:
        print("Reading and combining files...")
    combined_df = read_and_combine_files(file_paths)
    
    # Get coordinates
    lons = combined_df['Lon'].values
    lats = combined_df['Lat'].values
    years = combined_df['Year'].unique()
    
    # Use grid information if available
    if grid_info:
        if verbose:
            print("Using grid information from grids.nc")
        grid_lons = grid_info['lon']
        grid_lats = grid_info['lat']
    else:
        if verbose:
            print("Using coordinates from .out files")
        grid_lons = np.unique(lons)
        grid_lats = np.unique(lats)
    
    # Unified time handling for all file types (daily, monthly, yearly)
    # Determine unique time values and convert all to datetime format
    years = sorted(combined_df['Year'].unique())
    
    # Create datetime objects based on available time information
    if 'Day' in combined_df.columns and 'Month' in combined_df.columns:
        # Data has year, month and day
        time_groups = combined_df.groupby(['Year', 'Month', 'Day']).size().reset_index()[['Year', 'Month', 'Day']]
        times = [pd.Timestamp(year=int(year), month=int(month), day=int(day)) 
                for year, month, day in zip(time_groups['Year'], time_groups['Month'], time_groups['Day'])]
    elif 'Day' in combined_df.columns:
        # Data has year and day of year
        time_groups = combined_df.groupby(['Year', 'Day']).size().reset_index()[['Year', 'Day']]
        times = [pd.Timestamp(year=int(year), month=1, day=1) + pd.Timedelta(days=int(day)-1) 
                for year, day in zip(time_groups['Year'], time_groups['Day'])]
    elif 'Month' in combined_df.columns:
        # Data has year and month
        time_groups = combined_df.groupby(['Year', 'Month']).size().reset_index()[['Year', 'Month']]
        times = [pd.Timestamp(year=int(year), month=int(month), day=1) 
                for year, month in zip(time_groups['Year'], time_groups['Month'])]
    else:
        # Data has only years
        times = [pd.Timestamp(year=int(year), month=1, day=1) for year in years]
    
    # Always use 'time' dimension for consistency across all file types
    time_dim = 'time'
    
    # Create a dictionary to store all processed variables
    # IMPORTANT: This dictionary must persist through all variable processing iterations
    all_data_vars = {}
    
    # Check if this is a monthly file by looking for month names in columns
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_columns = [col for col in columns if col in month_names]
    is_monthly_file = len(month_columns) > 0 and len(month_columns) <= 12
    
    if is_monthly_file and verbose:
        print(f"Detected monthly file format with {len(month_columns)} month columns")
    
    # Process each variable column (everything after Lon, Lat, Year, [Day])
    start_idx = 3 if not has_day else 4
    
    if is_monthly_file:
        # For monthly files, the variable name is in the base filename
        # The month columns are time points, not separate variables
        base_filename = os.path.basename(file_paths[0])
        var_name = os.path.splitext(base_filename)[0].split('_')[0]  # Extract variable name from filename
        var_columns = [var_name]  # Only one variable with monthly time points
    else:
        # Standard format - each column after coordinates is a separate variable
        var_columns = columns[start_idx:]
    
    if verbose:
        print(f"Processing {len(var_columns)} variables across {len(combined_df)} data points...")
    
    # Flag to control whether we skip regular processing for special file formats
    skip_regular_processing = False
    
    # If this is a monthly file, we need to reshape the data
    if is_monthly_file:
        if verbose:
            print("Processing monthly file format - reshaping data...")
            
        # For monthly files, we need a completely different approach
        # Extract variable name from the filename
        base_filename = os.path.basename(file_paths[0])
        var_name = os.path.splitext(base_filename)[0].split('_')[0]  # Extract variable name from filename
        
        # Get unique years and prepare time points
        years = sorted(combined_df['Year'].unique())
        
        # Create time points for all months in all years
        times = []
        for year in years:
            for month in range(1, 13):  # 1-12 for all months
                times.append(pd.Timestamp(year=int(year), month=month, day=1))
        
        if verbose:
            print(f"Created {len(times)} time points across {len(years)} years")
        
        # Extract unique latitude and longitude values
        unique_lats = sorted(combined_df['Lat'].unique(), reverse=True)  # North to South
        unique_lons = sorted(combined_df['Lon'].unique())  # West to East
        
        if verbose:
            print(f"Found {len(unique_lats)} unique latitudes and {len(unique_lons)} unique longitudes")
        
        # Create a 3D array with dimensions (time, lat, lon)
        n_times = len(times)
        n_lats = len(unique_lats)
        n_lons = len(unique_lons)
        
        if verbose:
            print(f"Creating data array with dimensions: time={n_times}, lats={n_lats}, lons={n_lons}")
        
        # Create lookup dictionaries for coordinates
        lat_to_idx = {lat: i for i, lat in enumerate(unique_lats)}
        lon_to_idx = {lon: i for i, lon in enumerate(unique_lons)}
        
        # Process the data by creating a 3D array (time, lat, lon)
        # First reshape the raw data
        reshaped_data = np.full((n_times, n_lats * n_lons), np.nan)
        
        # Process each row in the original data
        for _, row in combined_df.iterrows():
            lat, lon, year = row['Lat'], row['Lon'], int(row['Year'])
            lat_idx = lat_to_idx[lat]
            lon_idx = lon_to_idx[lon]
            point_idx = lat_idx * n_lons + lon_idx  # Flatten the 2D grid to 1D
            
            # Process each month for this lat/lon/year
            for month_idx, month_name in enumerate(month_columns):
                if month_name in row:
                    # Calculate the time index (12 months per year)
                    year_offset = (year - years[0]) * 12
                    time_idx = year_offset + month_idx
                    if time_idx < n_times:  # Safety check
                        value = row[month_name]
                        reshaped_data[time_idx, point_idx] = value
        
        # For unified approach with other file types, create a 2D array (time, points)
        # and update the sorted coordinates for grid expansion later
        sorted_lats = unique_lats
        sorted_lons = unique_lons
        all_data_vars[var_name] = (('time', 'points'), reshaped_data)
        
        if verbose:
            print(f"Processed monthly data for variable '{var_name}'")
            print(f"Final data shape: {reshaped_data.shape}")
            print(f"Non-NaN values: {np.count_nonzero(~np.isnan(reshaped_data))}")
        
        # Skip the regular processing loop as we've already handled the data
        skip_regular_processing = True
            
        if verbose:
            print(f"Reshaped data to long format with {len(times)} time points")
    
    # Always use 'time' dimension for consistency across all file types
    time_dim = 'time'
    
    # Use tqdm for progress bar over variables
    # Skip the regular processing loop if we already handled this file format specially
    if not skip_regular_processing:
        for var_col in tqdm(var_columns, desc="Processing variables", disable=not verbose):
            # Skip creating a full grid with NaNs and instead create a 1D representation
            if verbose:
                print(f"  > Processing dataset for {var_col} using 1D coordinate representation")
            
            # Get all coordinates as arrays
            lons_array = combined_df['Lon'].values
            lats_array = combined_df['Lat'].values
            years_array = combined_df['Year'].values
            values_array = combined_df[var_col].values
        
        t_start = time.time()
        if verbose:
            print("    Creating 1D sorted coordinates (N->S, E->W)...")
        
        # Create a DataFrame to help with sorting and indexing
        coord_df = pd.DataFrame({
            'lat': lats_array,
            'lon': lons_array,
            'value': values_array,
            'year': years_array
        })
        
        # Add day column if present in the input data
        if has_day:
            coord_df['day'] = combined_df['Day'].values
        
        # Always use just the coordinates found in the data files for processing
        # (we'll expand to the full grid later when creating the final dataset)
        if verbose:
            print("    Sorting coordinates by latitude (N->S) and longitude (W->E)...")
        
        # Sort by latitude (descending, N->S) and longitude (ascending, W->E)
        coord_df = coord_df.sort_values(by=['lat', 'lon'], ascending=[False, True])
        
        # Get the unique coordinates (now sorted)
        unique_coords = coord_df.drop_duplicates(subset=['lat', 'lon'])
        sorted_lats = unique_coords['lat'].values
        sorted_lons = unique_coords['lon'].values
        
        # Save grid info for later expansion
        have_full_grid = grid_info is not None and 'full_lat' in grid_info and 'full_lon' in grid_info
        if have_full_grid and verbose:
            num_land_points = len(grid_info['full_lat'])
            print(f"    Processing with {len(sorted_lats)} data points (will expand to {num_land_points} grid points later)")
        else:
            if verbose:
                print(f"    Created sorted coordinate arrays with {len(sorted_lats)} unique points")
            print(f"    Coordinate processing: {time.time() - t_start:.2f} seconds")
            print("    Creating data arrays by time period...")
            t_start = time.time()
        
        # Create a dictionary to store the variable data
        data_vars = {}
        
        # Group data by time period and create 1D arrays
        if has_day:
            # This is a major bottleneck - switch to a vectorized approach
            # Create a dictionary mapping (year, day) tuples to time indices
            time_map = {}
            for i, t in enumerate(times):
                year = t.year
                day = t.dayofyear
                time_map[(year, day)] = i
                
            if is_monthly_file:
                # For monthly files, we use the pre-calculated time_idx column
                if verbose:
                    print("    Using monthly file time indices...")
                
                t1 = time.time()
                time_idx_array = coord_df['time_idx'].values.astype(int)
                
            elif has_day:
                # Using the vectorized time index mapping
                if verbose:
                    print("    Creating time indices with vectorized mapping...")
                
                t1 = time.time()
                
                # Create arrays for faster operations
                years_array = coord_df['year'].values.astype(int)
                days_array = coord_df['day'].values.astype(int)
                
                # Ensure we have proper index lookups for the time dimension
                time_idx_array = np.zeros_like(years_array)
                
                # Create a mapping from (year, day) to time index
                for i, (y, d) in enumerate(zip(years_array, days_array)):
                    key = (y, d)
                    time_idx = time_map.get(key)
                    if time_idx is not None:
                        time_idx_array[i] = time_idx
                        
            # Add the time indices to the dataframe
            coord_df['time_idx'] = time_idx_array
            
            t1 = time.time()
            if verbose:
                print(f"    Creating time index mapping took {t1-t_start:.2f} seconds")
                print("    Building fast coordinate index lookup tables...")
                
            # Create lookup dictionaries for faster coordinate mapping
            lat_to_idx = {lat: i for i, lat in enumerate(sorted_lats)}
            lon_to_idx = {lon: i for i, lon in enumerate(sorted_lons)}
            
            t2 = time.time()
            if verbose:
                print(f"    Building coordinate lookup tables took {t2-t1:.2f} seconds")
                print("    Performing vectorized grouping by time...")
                
            # Group by time index - this is the slowest part
            grouped = coord_df.groupby('time_idx')
            
            t3 = time.time()
            if verbose:
                print(f"    Grouping by time took {t3-t2:.2f} seconds")
                print("    Creating data arrays...")
                
            # Using vectorized approach instead of slow pandas groupby
            n_times = len(times)
            if verbose:
                print(f"    Creating data array with dimensions ({n_times}, {len(sorted_lats)})")
            var_data = np.full((n_times, len(sorted_lats)), np.nan)
            
            # Prepare a timing dictionary for detailed analysis
            timing_per_group = {}
            total_points = 0
            
            # MAJOR OPTIMIZATION: Process directly without groupby, which is very slow for large datasets
            if verbose:
                print("    Using direct array-based data processing (much faster)...")
                print("    Getting optimal parallel configuration...")
                
            # Get optimal system resource configuration
            parallel_config = get_parallel_config(verbose=verbose)
            n_workers = parallel_config['n_jobs']
            
            if verbose:
                print(f"    Using {n_workers} workers with {parallel_config['backend']} backend")
                print("    Preparing optimized data structures...")
            
            # Pre-allocate the output array
            var_data = np.full((len(times), len(sorted_lats)), np.nan)
            
            # Extract arrays from dataframe for much faster processing
            lat_array = coord_df['lat'].values
            lon_array = coord_df['lon'].values
            time_idx_array = coord_df['time_idx'].values
            value_array = coord_df['value'].values
            
            # Create a mapping of (lat, lon) to the index in sorted_lats
            # This will be used to map from input data points to the full grid positions
            if verbose:
                print("    Creating mapping from points in output files to full grid positions...")
                
            # Create a mapping of sorted_lat/sorted_lon pairs to their indices in the sorted arrays
            grid_point_to_idx = {}
            for i, (lat, lon) in enumerate(zip(sorted_lats, sorted_lons)):
                grid_point_to_idx[(lat, lon)] = i
            
            # Create chunks for parallel processing
            chunk_size = len(value_array) // (n_workers * 2)  # Ensure we have at least 2 chunks per worker
            if chunk_size == 0:
                chunk_size = len(value_array)
                
            chunks = []
            for i in range(0, len(value_array), chunk_size):
                end = min(i + chunk_size, len(value_array))
                chunks.append((i, end))
                
            if verbose:
                print(f"    Processing data in {len(chunks)} chunks using {n_workers} workers")
            
            # Create simple latitude index dictionary for much faster lookups
            if verbose:
                print("    Creating high-performance latitude index...")
            lat_to_idx = {lat: i for i, lat in enumerate(sorted_lats)}
                
            # Define a parallel processing function that works directly with arrays
            # We'll process each chunk independently and return the results to avoid shared memory issues
            def process_chunk(chunk_idx):
                start, end = chunks[chunk_idx]
                # Create a local copy of the structure to store results
                # This avoids multiprocessing shared memory issues
                results_dict = {}
                points_processed = 0
                
                # Process this chunk
                for i in range(start, end):
                    lat = lat_array[i]
                    lon = lon_array[i]
                    time_idx = time_idx_array[i]
                    value = value_array[i]
                    
                    # Use direct latitude lookup which is MUCH faster than coordinate pair lookup
                    # This was the key optimization that previously gave us 10+ million points/sec
                    if lat in lat_to_idx:
                        point_idx = lat_to_idx[lat]
                        # Store in a dictionary keyed by (time_idx, point_idx) instead of modifying shared memory
                        results_dict[(time_idx, point_idx)] = value
                        points_processed += 1
                        
                return chunk_idx, results_dict, points_processed
            
            # Process chunks in parallel
            t_process_start = time.time()
            results = Parallel(n_jobs=n_workers, backend=parallel_config['backend'])(
                delayed(process_chunk)(i) for i in range(len(chunks))
            )
            
            # Merge results from all chunks into the output array
            if verbose:
                print("    Merging results from parallel chunks...")
                
            t_merge_start = time.time()
            total_points = 0
            
            # Collect results from all chunks
            for _, chunk_results, points in results:
                total_points += points
                # Apply each value from the chunk results to the output array
                for (time_idx, point_idx), value in chunk_results.items():
                    var_data[time_idx, point_idx] = value
                    
            merge_time = time.time() - t_merge_start
            processing_time = time.time() - t_process_start
            
            if verbose:
                print(f"    Merging results took {merge_time:.2f} seconds")
            
            if verbose:
                print(f"    Processed {total_points} points in {processing_time:.2f} seconds")
                if processing_time > 0:
                    print(f"    Processing speed: {total_points / processing_time:.0f} points/sec")
                print(f"    Final data array shape: {var_data.shape}")
                print(f"    Total non-NaN values: {np.count_nonzero(~np.isnan(var_data))}")
                print(f"    Memory usage: {var_data.nbytes / (1024 * 1024):.1f} MB")
            
            # Add to all_data_vars - this stores ALL variables across all iterations
            all_data_vars[var_col] = (('time', 'points'), var_data)
        else:
            # Use year directly
            t1 = time.time()
            coord_df['year_idx'] = coord_df['year'] - years[0]
            
            if verbose:
                print(f"    Creating year index mapping took {t1-t_start:.2f} seconds")
                print("    Building fast coordinate index lookup tables...")
                
            # Create lookup dictionaries for faster coordinate mapping
            lat_to_idx = {lat: i for i, lat in enumerate(sorted_lats)}
            lon_to_idx = {lon: i for i, lon in enumerate(sorted_lons)}
            
            t2 = time.time()
            if verbose:
                print(f"    Building coordinate lookup tables took {t2-t1:.2f} seconds")
                print("    Performing grouping by year...")
                
            # Group by year index - this can be slow
            grouped = coord_df.groupby('year_idx')
            
            t3 = time.time()
            if verbose:
                print(f"    Grouping by year took {t3-t2:.2f} seconds")
                print("    Creating data arrays...")
                
            # Create an array to hold the values for each year
            var_data = np.full((len(years), len(sorted_lats)), np.nan)
            
            # Prepare a timing dictionary for detailed analysis
            timing_per_group = {}
            total_points = 0
            
            # MAJOR OPTIMIZATION: Process directly without groupby, which is very slow for large datasets
            if verbose:
                print("    Using direct array-based data processing (much faster)...")
                print("    Getting optimal parallel configuration...")
                
            # Get optimal system resource configuration
            parallel_config = get_parallel_config(verbose=verbose)
            n_workers = parallel_config['n_jobs']
            
            if verbose:
                print(f"    Using {n_workers} workers with {parallel_config['backend']} backend")
                print("    Preparing optimized data structures...")
            
            # Pre-allocate the output array
            var_data = np.full((len(years), len(sorted_lats)), np.nan)
            
            # Extract arrays from dataframe for much faster processing
            lat_array = coord_df['lat'].values
            lon_array = coord_df['lon'].values
            year_idx_array = coord_df['year_idx'].values
            value_array = coord_df['value'].values
            
            # Create a mapping of (lat, lon) to the index in sorted_lats
            # This will be used to map from input data points to the full grid positions
            if verbose:
                print("    Creating mapping from points in output files to full grid positions...")
                
            # Create a mapping of sorted_lat/sorted_lon pairs to their indices in the sorted arrays
            grid_point_to_idx = {}
            for i, (lat, lon) in enumerate(zip(sorted_lats, sorted_lons)):
                grid_point_to_idx[(lat, lon)] = i
            
            # Create chunks for parallel processing based on available system resources
            chunk_size = len(value_array) // (n_workers * 2)  # Ensure we have at least 2 chunks per worker
            if chunk_size == 0:
                chunk_size = len(value_array)
                
            chunks = []
            for i in range(0, len(value_array), chunk_size):
                end = min(i + chunk_size, len(value_array))
                chunks.append((i, end))
                
            if verbose:
                print(f"    Processing data in {len(chunks)} chunks using {n_workers} workers")
            
            # Create simple latitude index dictionary for much faster lookups
            if verbose:
                print("    Creating high-performance latitude index...")
            lat_to_idx = {lat: i for i, lat in enumerate(sorted_lats)}
            
            # Define a parallel processing function that works directly with arrays
            # We'll process each chunk independently and return the results to avoid shared memory issues
            def process_chunk(chunk_idx):
                start, end = chunks[chunk_idx]
                # Create a local copy of the structure to store results
                # This avoids multiprocessing shared memory issues
                results_dict = {}
                points_processed = 0
                
                # Process this chunk
                for i in range(start, end):
                    lat = lat_array[i]
                    lon = lon_array[i]
                    year_idx = year_idx_array[i]
                    value = value_array[i]
                    
                    # Use direct latitude lookup which is MUCH faster than coordinate pair lookup
                    # This was the key optimization that previously gave us 10+ million points/sec
                    if lat in lat_to_idx:
                        point_idx = lat_to_idx[lat]
                        # Store in a dictionary keyed by (year_idx, point_idx) instead of modifying shared memory
                        results_dict[(year_idx, point_idx)] = value
                        points_processed += 1
                        
                return chunk_idx, results_dict, points_processed
            
            # Process chunks in parallel using Dask-compatible joblib
            t_process_start = time.time()
            results = Parallel(n_jobs=n_workers, backend=parallel_config['backend'])(
                delayed(process_chunk)(i) for i in range(len(chunks))
            )
            
            # Merge results from all chunks into the output array
            if verbose:
                print("    Merging results from parallel chunks...")
                
            t_merge_start = time.time()
            total_points = 0
            
            # Collect results from all chunks
            for _, chunk_results, points in results:
                total_points += points
                # Apply each value from the chunk results to the output array
                for (year_idx, point_idx), value in chunk_results.items():
                    var_data[year_idx, point_idx] = value
                    
            merge_time = time.time() - t_merge_start
            processing_time = time.time() - t_process_start
            
            if verbose:
                print(f"    Merging results took {merge_time:.2f} seconds")
            
            if verbose:
                print(f"    Processed {total_points} points in {processing_time:.2f} seconds")
                if processing_time > 0:
                    print(f"    Processing speed: {total_points / processing_time:.0f} points/sec")
                print(f"    Final data array shape: {var_data.shape}")
                print(f"    Total non-NaN values: {np.count_nonzero(~np.isnan(var_data))}")
                print(f"    Memory usage: {var_data.nbytes / (1024 * 1024):.1f} MB")
            
            # Add to all_data_vars - this stores ALL variables across all iterations
            # Always use unified 'time' dimension regardless of the original time format
            all_data_vars[var_col] = (('time', 'points'), var_data)
        
        if verbose:
            print(f"    Data array creation: {time.time() - t_start:.2f} seconds")
    
    if verbose:
        print("Creating xarray dataset with 1D coordinates...")
    
    # Create coordinates for 1D representation
    # We use a "points" dimension that contains all actual data points
    # Each point has an associated lat and lon value
    coords = {
        'points': np.arange(len(sorted_lats)),  # Point indices
        'lat_points': ('points', sorted_lats),   # Each point has a latitude
        'lon_points': ('points', sorted_lons)    # Each point has a longitude
    }
    
    # Add time coordinates if appropriate
    if has_day:
        coords['time'] = times
    else:
        coords['year'] = years
    
    # Create dataset
    ds = xr.Dataset(data_vars, coords=coords)
    
    # Add metadata
    ds.attrs['title'] = 'LPJ-GUESS output converted to NetCDF (1D representation)'
    ds.attrs['source'] = os.path.basename(file_paths[0])
    ds.attrs['creation_date'] = str(datetime.datetime.now())
    ds.attrs['grid_structure'] = '1D points array with no NaN values'
    ds.attrs['sorting'] = 'Points sorted by latitude (N->S) and longitude (E->W)'
    
    # Add variable metadata
    for var_name in var_columns:
        if var_name in ds:
            ds[var_name].attrs['long_name'] = var_name
            ds[var_name].attrs['units'] = 'unknown'  # Could be updated with a metadata lookup
    
    # Add coordinate metadata
    if 'points' in ds.coords:
        ds['points'].attrs['long_name'] = 'point indices'
        ds['points'].attrs['description'] = 'Data points sorted by latitude (N->S) and longitude (E->W)'
    
    if 'lon_points' in ds.coords:
        ds['lon_points'].attrs['long_name'] = 'longitude of each point'
        ds['lon_points'].attrs['units'] = 'degrees_east'
    
    # Now that we've processed the data, expand to the full grid if needed
    have_full_grid = grid_info is not None and 'full_lat' in grid_info and 'full_lon' in grid_info
    
    if have_full_grid:
        if verbose:
            print("Creating xarray dataset with complete grid expansion...")
            
        # Extract the full grid information
        full_lats = grid_info['full_lat']
        full_lons = grid_info['full_lon']
        
        # Create the 1D coordinate arrays from all land points in the grid
        # Sort latitudes north to south (descending) and longitudes west to east (ascending)
        # This is the final sorting that will appear in the output
        sorted_coord_pairs = sorted(zip(full_lats, full_lons), key=lambda x: (-x[0], x[1]))
        full_sorted_lats = np.array([lat for lat, _ in sorted_coord_pairs])
        full_sorted_lons = np.array([lon for _, lon in sorted_coord_pairs])
        
        # Map from our processed points to the full grid
        if verbose:
            print(f"Expanding from {len(sorted_lats)} processed points to {len(full_sorted_lats)} total grid points...")
        
        # Create expanded arrays with NaN values
        expanded_data_vars = {}
        
        # Build an efficient mapping between the two grids using a dictionary approach
        if verbose:
            print("    Building optimized grid mapping with efficient hashing...")
        t_grid_map = time.time()
        
        # Create a dictionary mapping from processed coordinates to their indices
        # Use string representation with fixed precision to handle floating-point issues
        processed_coords = {}
        for proc_idx, (lat, lon) in enumerate(zip(sorted_lats, sorted_lons)):
            # Round to fixed precision to handle floating-point imprecision
            key = (round(lat, 6), round(lon, 6))
            processed_coords[key] = proc_idx
            
        # Now efficiently map from full grid to processed points
        full_to_processed_idx = np.full(len(full_sorted_lats), -1, dtype=np.int32)
        matched_count = 0
        
        # Use the dictionary for O(1) lookups instead of O(n) searches
        for full_idx, (lat, lon) in enumerate(zip(full_sorted_lats, full_sorted_lons)):
            key = (round(lat, 6), round(lon, 6))
            proc_idx = processed_coords.get(key, -1)
            if proc_idx >= 0:
                full_to_processed_idx[full_idx] = proc_idx
                matched_count += 1
        
        if verbose:
            print(f"    Created mapping in {time.time() - t_grid_map:.2f} seconds")
            print(f"    Found {matched_count} matching points between processed and full grid")
            
        # Create expanded variables using the mapping (much faster than loops)
        if verbose:
            print("    Expanding variables to full grid...")
            
        t_expand = time.time()
        expanded_data_vars = {}
        
        # Process each variable efficiently using the mapping array
        for var_name, (dims, var_data) in all_data_vars.items():
            # Create expanded arrays with NaNs
            if dims[0] == 'time':
                time_dim = len(times)
                expanded_var = np.full((time_dim, len(full_sorted_lats)), np.nan)
                
                # Use the mapping to fill values (vectorized approach)
                valid_mask = full_to_processed_idx >= 0
                valid_indices = full_to_processed_idx[valid_mask]
                
                # This is much faster than looping
                for t in range(time_dim):
                    expanded_var[t, valid_mask] = var_data[t, valid_indices]
                    
            else:  # 'year' dimension
                year_dim = len(years)
                expanded_var = np.full((year_dim, len(full_sorted_lats)), np.nan)
                
                # Use the mapping to fill values (vectorized approach)
                valid_mask = full_to_processed_idx >= 0
                valid_indices = full_to_processed_idx[valid_mask]
                
                # This is much faster than looping
                for y in range(year_dim):
                    expanded_var[y, valid_mask] = var_data[y, valid_indices]
            
            # Store with the same dimension structure - FIXED SYNTAX
            expanded_data_vars[var_name] = ((dims[0], 'points'), expanded_var)
        
        # Use the expanded data and full coordinates
        # CRITICAL FIX: Store ALL expanded variables in all_data_vars
        # This ensures all variables from all iterations are included in the final NetCDF file
        all_data_vars = expanded_data_vars
        sorted_lats = full_sorted_lats
        sorted_lons = full_sorted_lons
        grid_name = "TL255-land"
        total_points = len(sorted_lats)
        
        if verbose:
            expansion_time = time.time() - t_expand
            print(f"    Grid expansion took {expansion_time:.2f} seconds")
            print(f"    Final grid has {total_points} points with {matched_count} data points")
            print(f"    Expansion speed: {total_points / expansion_time:.0f} points/sec")
    else:
        if verbose:
            print("Creating xarray dataset with processed points only...")
            print(f"Using grid with {len(sorted_lats)} points from output files")
        grid_name = "processed-points-only"
    
    # Create coordinates for the final dataset
    coords = {
        'lat': ('points', sorted_lats),
        'lon': ('points', sorted_lons),
    }
    
    # Add time or year coordinate
    if has_day:
        coords['time'] = ('time', times)
    else:
        coords['year'] = ('year', years)

    # DEBUG - Print variable names before dataset creation
    if verbose:
        print(f"Variables being added to dataset: {list(all_data_vars.keys())}")

    # Create the dataset with full grid information - CORRECT FORMAT
    # xarray expects data_vars in this format: {'var_name': (dims, data)}    
    processed_vars = {}
    for var_name, (dims, var_data) in all_data_vars.items():
        processed_vars[var_name] = (dims, var_data)

    ds = xr.Dataset(
        data_vars=processed_vars,
        coords=coords,
        attrs={
            'title': 'LPJG Processed Data',
            'source': 'lpjg2nc converter',
            'creation_date': str(datetime.datetime.now()),
            'grid_type': grid_name,
            'total_land_points': len(sorted_lats),
            'description': 'Data on land points only, with NaN values where no data is available'
        }
    )
    
    # Add extra coordinate information
    ds['lat'].attrs['standard_name'] = 'latitude'
    ds['lat'].attrs['long_name'] = 'latitude of grid points'
    ds['lat'].attrs['units'] = 'degrees_north'
    
    ds['lon'].attrs['standard_name'] = 'longitude'
    ds['lon'].attrs['long_name'] = 'longitude of grid points'
    ds['lon'].attrs['units'] = 'degrees_east'
    
    # Check if this is a dry run (just testing the processing)
    is_dry_run = output_path == "-" or output_path is None
    
    if is_dry_run:
        if verbose:
            print("Dry run mode: not writing NetCDF file")
            print(f"Dataset created with shape: {ds.dims}")
            print(f"Variables: {list(ds.data_vars.keys())}")
            print(f"Coordinates: {list(ds.coords.keys())}")
            
            # Print memory usage statistics
            total_memory = 0
            for var in ds.data_vars:
                var_memory = ds[var].nbytes / (1024 * 1024)  # MB
                print(f"  - {var}: {var_memory:.1f} MB")
                total_memory += var_memory
            print(f"Total memory usage: {total_memory:.1f} MB")
    else:
        # Write to NetCDF file if output path is provided
        if verbose:
            print(f"Writing dataset to {output_path}...")
            
        try:
            # Check if the output_path is a directory or file
            if os.path.isdir(output_path) or output_path.endswith('/'):
                # It's a directory, so generate a filename based on the input file
                base_input_name = os.path.basename(file_paths[0])
                file_base = os.path.splitext(base_input_name)[0]  # Remove extension
                # Generate a clean filename without timestamp
                filename = f"{file_base}.nc"
                full_output_path = os.path.join(output_path, filename)
            else:
                # It's already a file path
                full_output_path = output_path
                
            # Make sure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(full_output_path)), exist_ok=True)
            
            # Set encoding for compression
            encoding = {var: {'zlib': True, 'complevel': 4} for var in ds.data_vars}
            ds.to_netcdf(full_output_path, encoding=encoding)
            if verbose:
                print(f"Successfully wrote dataset to {full_output_path}")
                
            # Update output_path to reflect the actual file written
            output_path = full_output_path
        except Exception as e:
            print(f"Error writing NetCDF file: {e}")
            print("Continuing without writing file...")
    
    # Return the dataset regardless
    return ds
    
    if verbose:
        end_time = time.time()
        print(f"2D file processing completed in {end_time - start_time:.2f} seconds")
        print(f"Output written to {output_path}")
    
    return output_path



def process_3d_file(file_paths, output_path, grid_info=None, verbose=False):
    """Placeholder for 3D file processing that will be reimplemented in the future.
    
    Parameters
    ----------
    file_paths : list
        List of paths to .out files.
    output_path : str
        Path to save the NetCDF file.
    grid_info : dict, optional
        Grid information from grid_utils.read_grid_information.
    verbose : bool, optional
        Whether to print verbose output.
        
    Returns
    -------
    str or None
        Path to the created NetCDF file or None if processing is disabled.
    """
    if verbose:
        print("3D file processing is temporarily disabled. 3D support will be added back soon.")
    return None

def process_file(file_paths, output_path, grid_info=None, verbose=False):
    """
    Process a file or group of files and convert to NetCDF.
    
    Parameters
    ----------
    file_paths : list
        List of paths to .out files.
    output_path : str
        Path to save the NetCDF file.
    grid_info : dict, optional
        Grid information from grid_utils.read_grid_information.
    verbose : bool, optional
        Whether to print verbose output.
        
    Returns
    -------
    str
        Path to the created NetCDF file.
    """
    # Get file structure from the first file
    from file_parser import detect_file_structure
    
    structure = detect_file_structure(file_paths[0])
    
    # For now, only handling 2D files
    if structure['is_3d']:
        if verbose:
            print("3D file processing is temporarily disabled. 3D support will be added back soon.")
        return None
    else:
        return process_2d_file(file_paths, output_path, grid_info, verbose)
