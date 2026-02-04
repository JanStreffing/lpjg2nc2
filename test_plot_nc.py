#!/usr/bin/env python3
"""
Test script to load and plot LPJ-GUESS NetCDF files.
"""

import sys
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

def plot_nc_file(nc_path):
    """Load and plot a NetCDF file."""
    print(f"Loading: {nc_path}")
    
    # Open the dataset
    ds = xr.open_dataset(nc_path)
    
    print("\n=== Dataset Info ===")
    print(ds)
    
    print("\n=== Dimensions ===")
    for dim, size in ds.dims.items():
        print(f"  {dim}: {size}")
    
    print("\n=== Coordinates ===")
    for coord in ds.coords:
        print(f"  {coord}: {ds.coords[coord].shape}")
    
    print("\n=== Variables ===")
    data_vars = list(ds.data_vars)
    for var in data_vars:
        print(f"  {var}: {ds[var].shape} - {ds[var].dims}")
    
    # Get lat/lon coordinates
    if 'lat_points' in ds.coords and 'lon_points' in ds.coords:
        lats = ds['lat_points'].values
        lons = ds['lon_points'].values
    elif 'lat' in ds.coords and 'lon' in ds.coords:
        lats = ds['lat'].values
        lons = ds['lon'].values
    else:
        print("\nNo recognizable lat/lon coordinates found")
        lats = None
        lons = None
    
    if lats is not None:
        print(f"\n=== Coordinate Ranges ===")
        print(f"  Latitude:  {lats.min():.2f} to {lats.max():.2f}")
        print(f"  Longitude: {lons.min():.2f} to {lons.max():.2f}")
    
    # Plot each variable
    n_vars = len(data_vars)
    if n_vars == 0:
        print("No data variables to plot")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, min(n_vars, 4), figsize=(5*min(n_vars, 4), 4))
    if n_vars == 1:
        axes = [axes]
    
    for i, var in enumerate(data_vars[:4]):  # Plot up to 4 variables
        ax = axes[i]
        data = ds[var].values
        
        # Handle time dimension - take first timestep
        if len(data.shape) > 1:
            data = data[0, :]  # First time step
        
        # Remove NaN for statistics
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            ax.text(0.5, 0.5, f'{var}\n(all NaN)', ha='center', va='center')
            continue
        
        print(f"\n=== {var} Statistics ===")
        print(f"  Min:  {np.nanmin(data):.4f}")
        print(f"  Max:  {np.nanmax(data):.4f}")
        print(f"  Mean: {np.nanmean(data):.4f}")
        print(f"  Valid points: {len(valid_data)} / {len(data)}")
        
        if lats is not None and lons is not None:
            # Scatter plot on map
            sc = ax.scatter(lons, lats, c=data, s=1, cmap='viridis')
            plt.colorbar(sc, ax=ax, shrink=0.8)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        else:
            # Simple line plot
            ax.plot(data)
            ax.set_xlabel('Point index')
            ax.set_ylabel('Value')
        
        ax.set_title(f'{var}')
    
    plt.tight_layout()
    
    # Save figure
    out_path = nc_path.replace('.nc', '_plot.png')
    plt.savefig(out_path, dpi=150)
    print(f"\n=== Plot saved to: {out_path} ===")
    
    plt.show()
    
    ds.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_plot_nc.py <netcdf_file>")
        print("Example: python test_plot_nc.py /path/to/file.nc")
        sys.exit(1)
    
    plot_nc_file(sys.argv[1])
