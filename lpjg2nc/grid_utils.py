#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid utilities for LPJ-GUESS to NetCDF conversion.
"""

import os
import numpy as np
from netCDF4 import Dataset

def read_grid_information(base_path):
    """
    Read grid information from grids.nc file.
    Process it to create a reduced Gaussian grid structure.
    
    Parameters
    ----------
    base_path : str
        Path to the directory containing grids.nc.
        
    Returns
    -------
    dict or None
        Dictionary with grid information if successful, None otherwise.
    """
    grid_file = os.path.join(base_path, 'grids.nc')
    if not os.path.exists(grid_file):
        print(f"Warning: Grid file {grid_file} not found. Using coordinates from .out files.")
        return None
    
    try:
        grid_data = Dataset(grid_file)
        
        # Extract TL255-land.lat and TL255-land.lon
        # The dimensions in the file are (y_TL255-land, x_TL255-land)
        tl_lat = grid_data.variables['TL255-land.lat'][:].flatten()
        tl_lon = grid_data.variables['TL255-land.lon'][:].flatten()
        
        # Get unique values for latitude and longitude
        unique_lat = np.unique(tl_lat)
        unique_lon = np.unique(tl_lon)
        
        # Create a reduced Gaussian grid structure
        # For each latitude, find all associated longitudes
        reduced_grid = {}
        for lat in unique_lat:
            # Get indices where this latitude appears
            lat_indices = np.where(tl_lat == lat)[0]
            # Get the corresponding longitudes
            lons_at_lat = tl_lon[lat_indices]
            # Sort the longitudes
            lons_at_lat = np.sort(lons_at_lat)
            # Store in the reduced grid dictionary
            reduced_grid[lat] = lons_at_lat
        
        grid_data.close()
        
        return {
            'lat': unique_lat,  # All unique latitudes
            'lon': unique_lon,  # All unique longitudes (for reference)
            'full_lat': tl_lat,  # Full flattened array of latitudes
            'full_lon': tl_lon,  # Full flattened array of longitudes
            'reduced_grid': reduced_grid  # Dict mapping each latitude to its longitudes
        }
    except Exception as e:
        print(f"Warning: Error reading grid file: {e}. Using coordinates from .out files.")
        return None

def match_coordinates_to_grid(lons, lats, grid_info):
    """
    Match coordinates from .out files to the grid coordinates.
    
    Parameters
    ----------
    lons : array-like
        Longitude values from .out files.
    lats : array-like
        Latitude values from .out files.
    grid_info : dict
        Grid information from read_grid_information.
        
    Returns
    -------
    tuple
        (lon_indices, lat_indices) - Index arrays for the input coordinates in the grid.
    """
    if grid_info is None:
        return None, None
    
    grid_lons = grid_info['lon']
    grid_lats = grid_info['lat']
    
    # Find the indices of the closest grid points
    lon_indices = np.zeros(len(lons), dtype=int)
    lat_indices = np.zeros(len(lats), dtype=int)
    
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        lon_idx = np.abs(grid_lons - lon).argmin()
        lat_idx = np.abs(grid_lats - lat).argmin()
        
        lon_indices[i] = lon_idx
        lat_indices[i] = lat_idx
    
    return lon_indices, lat_indices
