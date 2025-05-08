#!/usr/bin/env python3
import numpy as np
import xarray as xr
import sys

def analyze_netcdf(file_path, verbose=True, return_stats=False):
    """
    Analyze a NetCDF file to count NaN vs. non-NaN values.
    
    Parameters
    ----------
    file_path : str
        Path to the NetCDF file to analyze
    verbose : bool, default=True
        Whether to print analysis results
    return_stats : bool, default=False
        Whether to return statistics dictionary
        
    Returns
    -------
    dict or None
        Dictionary with NaN statistics if return_stats is True
    """
    if verbose:
        print(f"Analyzing file: {file_path}")
    
    ds = xr.open_dataset(file_path)
    
    if verbose:
        print("\nDataset dimensions:")
        for dim_name, dim_size in ds.dims.items():
            print(f"  {dim_name}: {dim_size}")
    
    if verbose:
        print("\nVariable statistics:")
    
    total_values = 0
    total_nans = 0
    var_stats = {}
    
    for var_name in ds.data_vars:
        # Skip coordinate variables
        if var_name in ds.coords:
            continue
        
        var_data = ds[var_name].values
        n_values = var_data.size
        n_nans = np.isnan(var_data).sum()
        n_valid = n_values - n_nans
        nan_percentage = (n_nans / n_values) * 100 if n_values > 0 else 0
        
        var_stats[var_name] = {
            "total": int(n_values),
            "valid": int(n_valid),
            "nans": int(n_nans),
            "valid_pct": float(100 - nan_percentage),
            "nan_pct": float(nan_percentage)
        }
        
        if verbose:
            print(f"  {var_name}:")
            print(f"    Total values: {n_values}")
            print(f"    Valid values: {n_valid} ({100 - nan_percentage:.2f}%)")
            print(f"    NaN values: {n_nans} ({nan_percentage:.2f}%)")
        
        total_values += n_values
        total_nans += n_nans
    
    total_valid = total_values - total_nans
    total_nan_percentage = (total_nans / total_values) * 100 if total_values > 0 else 0
    
    if verbose:
        print("\nOverall statistics:")
        print(f"  Total values across all variables: {total_values}")
        print(f"  Total valid values: {total_valid} ({100 - total_nan_percentage:.2f}%)")
        print(f"  Total NaN values: {total_nans} ({total_nan_percentage:.2f}%)")
        
        if total_nan_percentage > 95:
            print("\n  ⚠️ Warning: Dataset is very sparse (>95% NaN values)")
            print("     Consider using a spatial subset or different grid resolution.")
    
    # Close the dataset
    ds.close()
    
    if return_stats:
        return {
            "total_values": int(total_values),
            "total_valid": int(total_valid),
            "total_nans": int(total_nans),
            "total_valid_pct": float(100 - total_nan_percentage),
            "total_nan_pct": float(total_nan_percentage),
            "variables": var_stats
        }

def print_short_summary(stats):
    """
    Print a concise summary of NaN statistics.
    
    Parameters
    ----------
    stats : dict
        Dictionary of NaN statistics from analyze_netcdf()
    """
    valid_pct = stats['total_valid_pct']
    nan_pct = stats['total_nan_pct']
    
    print(f"📊 Data coverage: {valid_pct:.2f}% valid data, {nan_pct:.2f}% NaN values")
    
    # Add warning for very sparse datasets
    if nan_pct > 95:
        print(f"⚠️  Warning: Dataset is very sparse (>95% NaN values)")
        print(f"   Consider using a spatial subset or different grid resolution.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_nans.py <netcdf_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    analyze_netcdf(file_path)
