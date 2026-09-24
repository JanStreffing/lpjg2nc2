#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid utilities for LPJ-GUESS to NetCDF conversion.
"""

import os
import numpy as np
from netCDF4 import Dataset

def read_grid_information(base_path, grid_file=None):
    """
    Read grid information from a grids.nc file.
    Process it to create a reduced Gaussian grid structure.
    
    Parameters
    ----------
    base_path : str
        Directory searched for grids.nc when grid_file is not given.
    grid_file : str, optional
        Explicit path to a grids.nc file (default: <base_path>/grids.nc).
        
    Returns
    -------
    dict or None
        Dictionary with grid information if successful, None otherwise.
    """
    if grid_file is None:
        grid_file = os.path.join(base_path, 'grids.nc')
    if not os.path.exists(grid_file):
        print(f"Warning: Grid file {grid_file} not found. Using coordinates from .out files.")
        return None
    
    try:
        grid_data = Dataset(grid_file)

        # Auto-detect the land grid variables: look for a matching
        # '<prefix>.lat' / '<prefix>.lon' pair (e.g. 'TL255-land', 'TCO95-land').
        var_names = list(grid_data.variables.keys())
        lat_vars = [v for v in var_names if v.endswith('.lat')]
        prefix = None
        for v in lat_vars:
            p = v[:-4]
            if f'{p}.lon' in grid_data.variables:
                # Prefer a '-land' grid if multiple pairs are present.
                if prefix is None or 'land' in p.lower():
                    prefix = p
        if prefix is None:
            raise KeyError(
                f"No matching '<name>.lat'/'<name>.lon' pair found in {grid_file}. "
                f"Available variables: {var_names}"
            )

        tl_lat = grid_data.variables[f'{prefix}.lat'][:].flatten()
        tl_lon = grid_data.variables[f'{prefix}.lon'][:].flatten()
        
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


def _infer_reduced_grid(lats, lons):
    """
    Infer reduced-Gaussian band structure from the points themselves.

    Returns (unique_lats ascending, dlon per band). The per-band longitude
    spacing is the smallest longitude gap within the band, snapped to
    360/n for integer n. Bands holding a single point (or only widely
    separated land points) can only overestimate the spacing.
    """
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    unique_lats = np.unique(lats)
    dlon = np.empty(len(unique_lats))
    for i, band in enumerate(unique_lats):
        band_lons = np.sort(np.unique(lons[lats == band]))
        if len(band_lons) < 2:
            dlon[i] = np.nan
            continue
        gaps = np.diff(np.append(band_lons, band_lons[0] + 360.0))
        gaps = gaps[gaps > 1e-6]
        dlon[i] = 360.0 / max(1, round(360.0 / gaps.min()))
    # Single-point bands: borrow the spacing of the nearest band that has one
    known = np.where(~np.isnan(dlon))[0]
    for i in np.where(np.isnan(dlon))[0]:
        dlon[i] = dlon[known[np.abs(known - i).argmin()]] if len(known) else 360.0
    return unique_lats, dlon


def compute_cell_bounds(lats, lons, grid_info=None):
    """
    Compute quadrilateral cell bounds (CF 'bounds' convention, 4 vertices
    per point) for points on a reduced-Gaussian grid.

    Unlike a naive midpoint-between-consecutive-array-entries approach
    (e.g. NCO's ncap2 make_bounds()), this uses the grid geometry:
    latitude band edges are the midpoints between neighboring Gaussian
    latitude bands (not evenly spaced), and the longitude width is the
    uniform per-band spacing (360 / n_lons_in_band) of a reduced-Gaussian
    grid.

    With grid_info (from read_grid_information(), i.e. a grids.nc) the band
    structure is exact. Without it, it is inferred from the points: the
    latitude bands present and the smallest longitude gap per band. The
    outermost bands are extended by half their neighbor gap (clipped at the
    poles).

    Parameters
    ----------
    lats, lons : array-like
        Coordinates of each point to compute bounds for.
    grid_info : dict, optional
        Must contain 'lat' (unique latitude bands) and 'reduced_grid'
        (band -> sorted lons) when given.

    Returns
    -------
    tuple of numpy.ndarray
        (lat_bnds, lon_bnds), each shape (len(lats), 4), vertex order
        SW, SE, NE, NW.
    """
    lats = np.asarray(lats)
    lons = np.asarray(lons)

    if grid_info is not None and 'reduced_grid' in grid_info:
        unique_lats = np.sort(np.unique(grid_info['lat']))  # south to north
        band_dlon = np.array([360.0 / max(1, len(grid_info['reduced_grid'][b]))
                              for b in unique_lats])
        exact_poles = True
    else:
        unique_lats, band_dlon = _infer_reduced_grid(lats, lons)
        exact_poles = False
    n_bands = len(unique_lats)

    mids = (unique_lats[:-1] + unique_lats[1:]) / 2 if n_bands > 1 else np.array([])
    if exact_poles or n_bands < 2:
        south, north = -90.0, 90.0
    else:
        south = unique_lats[0] - (mids[0] - unique_lats[0])
        north = unique_lats[-1] + (unique_lats[-1] - mids[-1])
    edges = np.concatenate(([south], mids, [north]))
    edges = np.clip(edges, -90.0, 90.0)

    # Nearest latitude band (.out-file values may differ from grids.nc by rounding)
    band_idx = np.abs(unique_lats[None, :] - lats[:, None]).argmin(axis=1) \
        if n_bands * len(lats) < 5e7 else \
        np.clip(np.searchsorted(unique_lats, lats), 0, n_bands - 1)
    lat_lo = edges[band_idx]
    lat_hi = edges[band_idx + 1]
    half = band_dlon[band_idx] / 2.0
    lon_lo = lons - half
    lon_hi = lons + half

    lat_bnds = np.stack([lat_lo, lat_lo, lat_hi, lat_hi], axis=1)
    lon_bnds = np.stack([lon_lo, lon_hi, lon_hi, lon_lo], axis=1)
    return lat_bnds, lon_bnds
