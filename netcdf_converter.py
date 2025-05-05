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
from grid_utils import match_coordinates_to_grid

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
    
    # Create a time dimension based on years (and days if present)
    if has_day:
        days = combined_df['Day'].unique()
        # If days exist, create a time dimension for each year-day combination
        times = [pd.Timestamp(year=int(year), month=1, day=1) + pd.Timedelta(days=int(day)-1) 
                 for year in years for day in days]
        time_dim = 'time'
    else:
        # If no days, use years as the time dimension
        times = [pd.Timestamp(year=int(year), month=1, day=1) for year in years]
        time_dim = 'year'
    
    # Create a dictionary to store variables
    data_vars = {}
    
    # Process each variable column (everything after Lon, Lat, Year, [Day])
    start_idx = 3 if not has_day else 4
    var_columns = columns[start_idx:]
    
    if verbose:
        print(f"Processing {len(var_columns)} variables across {len(combined_df)} data points...")
    
    # Use tqdm for progress bar over variables
    for var_col in tqdm(var_columns, desc="Processing variables", disable=not verbose):
        # Create an empty array
        if has_day:
            var_data = np.full((len(times), len(grid_lats), len(grid_lons)), np.nan)
        else:
            var_data = np.full((len(years), len(grid_lats), len(grid_lons)), np.nan)
        
        # Fill the array with data using vectorized operations where possible
        if len(combined_df) > 10000 and verbose:
            print(f"  > Processing large dataset for {var_col} ({len(combined_df)} rows)")
            # Use progress bar for large datasets
            for i, row in tqdm(combined_df.iterrows(), total=len(combined_df), 
                              desc=f"  > Variable {var_col}", disable=not verbose):
                # Find the closest grid points
                lon_idx = np.abs(grid_lons - row['Lon']).argmin()
                lat_idx = np.abs(grid_lats - row['Lat']).argmin()
                
                if has_day:
                    year_idx = np.where(years == row['Year'])[0][0]
                    day_idx = np.where(days == row['Day'])[0][0]
                    time_idx = year_idx * len(days) + day_idx
                    var_data[time_idx, lat_idx, lon_idx] = row[var_col]
                else:
                    year_idx = np.where(years == row['Year'])[0][0]
                    var_data[year_idx, lat_idx, lon_idx] = row[var_col]
        else:
            # Process without inner progress bar for smaller datasets
            for i, row in combined_df.iterrows():
                # Find the closest grid points
                lon_idx = np.abs(grid_lons - row['Lon']).argmin()
                lat_idx = np.abs(grid_lats - row['Lat']).argmin()
                
                if has_day:
                    year_idx = np.where(years == row['Year'])[0][0]
                    day_idx = np.where(days == row['Day'])[0][0]
                    time_idx = year_idx * len(days) + day_idx
                    var_data[time_idx, lat_idx, lon_idx] = row[var_col]
                else:
                    year_idx = np.where(years == row['Year'])[0][0]
                    var_data[year_idx, lat_idx, lon_idx] = row[var_col]
        
        # Add to data variables
        if has_day:
            data_vars[var_col] = (('time', 'lat', 'lon'), var_data)
        else:
            data_vars[var_col] = (('year', 'lat', 'lon'), var_data)
    
    if verbose:
        print("Creating xarray dataset...")
    
    # Create the dataset
    if has_day:
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                'time': times,
                'lat': grid_lats,
                'lon': grid_lons
            }
        )
    else:
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                'year': years,
                'lat': grid_lats,
                'lon': grid_lons
            }
        )
    
    # Add global attributes
    ds.attrs['title'] = 'LPJ-GUESS output converted to NetCDF'
    ds.attrs['source_files'] = ', '.join([os.path.basename(f) for f in file_paths])
    ds.attrs['creation_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Add variable attributes
    for var in ds.data_vars:
        ds[var].attrs['long_name'] = var
    
    # Add coordinate attributes
    ds['lat'].attrs['long_name'] = 'latitude'
    ds['lat'].attrs['units'] = 'degrees_north'
    ds['lon'].attrs['long_name'] = 'longitude'
    ds['lon'].attrs['units'] = 'degrees_east'
    
    if has_day:
        ds['time'].attrs['long_name'] = 'time'
        ds['time'].attrs['standard_name'] = 'time'
    else:
        ds['year'].attrs['long_name'] = 'year'
        ds['year'].attrs['units'] = 'year'
    
    # Save to NetCDF
    if verbose:
        print("Writing to NetCDF file...")
    output_file = os.path.join(output_path, f"{os.path.basename(file_paths[0])}.nc")
    ds.to_netcdf(output_file)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if verbose:
        print(f"Saved NetCDF file: {output_file}")
        print(f"Processing completed in {elapsed_time:.2f} seconds")
    
    return output_file


def process_3d_file(file_paths, output_path, grid_info=None, verbose=False):
    """
    Process a 3D .out file (with depth levels) and convert it to NetCDF.
    
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
        print(f"Processing 3D file: {os.path.basename(file_paths[0])}")
    
    # Get the file structure from the first file
    from file_parser import detect_file_structure, extract_depths, read_and_combine_files
    
    structure = detect_file_structure(file_paths[0])
    columns = structure['columns']
    depth_cols = structure['depth_cols']
    
    # Extract depth values
    depths = extract_depths(depth_cols)
    
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
    
    # Create a time dimension based on years (and days if present)
    if has_day:
        days = combined_df['Day'].unique()
        # If days exist, create a time dimension for each year-day combination
        times = [pd.Timestamp(year=int(year), month=1, day=1) + pd.Timedelta(days=int(day)-1) 
                 for year in years for day in days]
        time_dim = 'time'
    else:
        # If no days, use years as the time dimension
        times = [pd.Timestamp(year=int(year), month=1, day=1) for year in years]
        time_dim = 'year'
    
    # Create a dictionary to store the variable
    data_vars = {}
    
    # Determine the variable name from the file name
    var_name = os.path.basename(file_paths[0]).replace('.out', '')
    
    if verbose:
        print(f"Creating 3D data array with shape: time={len(times) if has_day else len(years)}, " 
              f"depth={len(depths)}, lat={len(grid_lats)}, lon={len(grid_lons)}")
    
    # Create an empty 4D array (time, depth, lat, lon)
    if has_day:
        var_data = np.full((len(times), len(depths), len(grid_lats), len(grid_lons)), np.nan)
    else:
        var_data = np.full((len(years), len(depths), len(grid_lats), len(grid_lons)), np.nan)
    
    if verbose:
        print(f"Processing {len(combined_df)} data points across {len(depths)} depth levels...")
    
    # Fill the array with data
    if len(combined_df) > 10000 and verbose:
        # Use progress bar for large datasets
        for i, row in tqdm(combined_df.iterrows(), total=len(combined_df), 
                          desc="Processing data points", disable=not verbose):
            # Find the closest grid points
            lon_idx = np.abs(grid_lons - row['Lon']).argmin()
            lat_idx = np.abs(grid_lats - row['Lat']).argmin()
            
            if has_day:
                year_idx = np.where(years == row['Year'])[0][0]
                day_idx = np.where(days == row['Day'])[0][0]
                time_idx = year_idx * len(days) + day_idx
                
                for d_idx, depth_col in enumerate(depth_cols):
                    var_data[time_idx, d_idx, lat_idx, lon_idx] = row[depth_col]
            else:
                year_idx = np.where(years == row['Year'])[0][0]
                
                for d_idx, depth_col in enumerate(depth_cols):
                    var_data[year_idx, d_idx, lat_idx, lon_idx] = row[depth_col]
    else:
        # Process without progress bar for smaller datasets
        for i, row in combined_df.iterrows():
            # Find the closest grid points
            lon_idx = np.abs(grid_lons - row['Lon']).argmin()
            lat_idx = np.abs(grid_lats - row['Lat']).argmin()
            
            if has_day:
                year_idx = np.where(years == row['Year'])[0][0]
                day_idx = np.where(days == row['Day'])[0][0]
                time_idx = year_idx * len(days) + day_idx
                
                for d_idx, depth_col in enumerate(depth_cols):
                    var_data[time_idx, d_idx, lat_idx, lon_idx] = row[depth_col]
            else:
                year_idx = np.where(years == row['Year'])[0][0]
                
                for d_idx, depth_col in enumerate(depth_cols):
                    var_data[year_idx, d_idx, lat_idx, lon_idx] = row[depth_col]
    
    # Add to data variables
    if has_day:
        data_vars[var_name] = (('time', 'depth', 'lat', 'lon'), var_data)
    else:
        data_vars[var_name] = (('year', 'depth', 'lat', 'lon'), var_data)
    
    if verbose:
        print("Creating xarray dataset...")
    
    # Create the dataset
    if has_day:
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                'time': times,
                'depth': depths,
                'lat': grid_lats,
                'lon': grid_lons
            }
        )
    else:
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                'year': years,
                'depth': depths,
                'lat': grid_lats,
                'lon': grid_lons
            }
        )
    
    # Add global attributes
    ds.attrs['title'] = 'LPJ-GUESS output converted to NetCDF'
    ds.attrs['source_files'] = ', '.join([os.path.basename(f) for f in file_paths])
    ds.attrs['creation_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Add variable attributes
    for var in ds.data_vars:
        ds[var].attrs['long_name'] = var
    
    # Add coordinate attributes
    ds['lat'].attrs['long_name'] = 'latitude'
    ds['lat'].attrs['units'] = 'degrees_north'
    ds['lon'].attrs['long_name'] = 'longitude'
    ds['lon'].attrs['units'] = 'degrees_east'
    ds['depth'].attrs['long_name'] = 'depth'
    ds['depth'].attrs['units'] = 'm'
    
    if has_day:
        ds['time'].attrs['long_name'] = 'time'
        ds['time'].attrs['standard_name'] = 'time'
    else:
        ds['year'].attrs['long_name'] = 'year'
        ds['year'].attrs['units'] = 'year'
    
    # Save to NetCDF
    if verbose:
        print("Writing to NetCDF file...")
    output_file = os.path.join(output_path, f"{os.path.basename(file_paths[0])}.nc")
    ds.to_netcdf(output_file)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if verbose:
        print(f"Saved NetCDF file: {output_file}")
        print(f"Processing completed in {elapsed_time:.2f} seconds")
    
    return output_file


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
    
    if structure['is_3d']:
        return process_3d_file(file_paths, output_path, grid_info, verbose)
    else:
        return process_2d_file(file_paths, output_path, grid_info, verbose)
