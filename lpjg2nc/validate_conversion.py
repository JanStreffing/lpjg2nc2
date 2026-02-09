#!/usr/bin/env python
"""
Validation script to compare .out files with their converted .nc files.
Checks if the data for year 1365 is identical between source and converted files.
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime


def get_file_pairs(out_dir, nc_dir, year=1365):
    """Find matching .out and .nc file pairs for a specific year."""
    pairs = []
    
    # Find all .out files for the specified year
    out_pattern = os.path.join(out_dir, f"*_{year}.out")
    out_files = glob.glob(out_pattern)
    
    for out_file in out_files:
        basename = os.path.basename(out_file)
        nc_basename = basename.replace('.out', '.nc')
        nc_file = os.path.join(nc_dir, nc_basename)
        
        if os.path.exists(nc_file):
            pairs.append((out_file, nc_file))
    
    return pairs


def read_out_file(out_file, max_rows=None):
    """Read .out file and return as DataFrame."""
    try:
        df = pd.read_csv(out_file, delim_whitespace=True, nrows=max_rows)
        return df
    except Exception as e:
        return None, str(e)


def read_nc_file(nc_file):
    """Read .nc file and return as xarray Dataset."""
    try:
        ds = xr.open_dataset(nc_file)
        return ds
    except Exception as e:
        return None, str(e)


def validate_file_pair(out_file, nc_file, verbose=False):
    """
    Validate that data in .out file matches data in .nc file.
    Returns a dict with validation results.
    """
    result = {
        'out_file': os.path.basename(out_file),
        'nc_file': os.path.basename(nc_file),
        'status': 'UNKNOWN',
        'details': [],
        'out_points': 0,
        'nc_points': 0,
        'out_times': 0,
        'nc_times': 0,
        'data_match': False,
        'max_diff': None,
        'mean_diff': None
    }
    
    # Read .out file
    out_df = read_out_file(out_file)
    if out_df is None or isinstance(out_df, tuple):
        result['status'] = 'ERROR'
        result['details'].append(f"Failed to read .out file: {out_df[1] if isinstance(out_df, tuple) else 'Unknown error'}")
        return result
    
    # Read .nc file
    nc_ds = read_nc_file(nc_file)
    if nc_ds is None or isinstance(nc_ds, tuple):
        result['status'] = 'ERROR'
        result['details'].append(f"Failed to read .nc file: {nc_ds[1] if isinstance(nc_ds, tuple) else 'Unknown error'}")
        return result
    
    # Get basic stats from .out file
    result['out_points'] = len(out_df['Lat'].unique()) if 'Lat' in out_df.columns else 0
    result['out_times'] = len(out_df['Day'].unique()) if 'Day' in out_df.columns else (
        len(out_df['Month'].unique()) if 'Month' in out_df.columns else 1
    )
    
    # Get basic stats from .nc file
    if 'points' in nc_ds.dims:
        result['nc_points'] = nc_ds.dims['points']
    if 'time' in nc_ds.dims:
        result['nc_times'] = nc_ds.dims['time']
    elif 'year' in nc_ds.dims:
        result['nc_times'] = nc_ds.dims['year']
    
    # Get variable names from .nc file
    nc_data_vars = [v for v in nc_ds.data_vars]
    if not nc_data_vars:
        result['status'] = 'ERROR'
        result['details'].append("No data variables in .nc file")
        return result
    
    # Identify coordinate columns in .out file
    coord_cols = ['Lon', 'Lat', 'Year', 'Day', 'Month', 'Mth']
    out_var_cols = [c for c in out_df.columns if c not in coord_cols]
    
    result['details'].append(f"NC vars: {len(nc_data_vars)}, OUT cols: {len(out_var_cols)}")
    
    # Get NC coordinates
    nc_lats = nc_ds['lat'].values
    nc_lons = nc_ds['lon'].values
    
    # Build lookup from nc data
    nc_lookup = {}
    for i, (lat, lon) in enumerate(zip(nc_lats, nc_lons)):
        nc_lookup[(round(lat, 4), round(lon, 4))] = i
    
    # Compare data values - find common variables
    common_vars = [v for v in nc_data_vars if v in out_var_cols]
    
    # If no common vars, try matching first NC var to first OUT col (for single-var files)
    if not common_vars and len(nc_data_vars) == 1 and len(out_var_cols) >= 1:
        # Single variable case - use first data column
        common_vars = [(nc_data_vars[0], out_var_cols[0])]
    elif common_vars:
        common_vars = [(v, v) for v in common_vars]
    else:
        result['status'] = 'WARN'
        result['details'].append(f"No common variables found between NC and OUT files")
        nc_ds.close()
        return result
    
    try:
        matches = 0
        mismatches = 0
        max_diff = 0
        diffs = []
        
        # Sample rows to compare
        sample_out = out_df.head(500)
        
        for nc_var, out_col in common_vars[:3]:  # Check up to 3 variables
            nc_data = nc_ds[nc_var].values
            
            for _, row in sample_out.iterrows():
                lat = round(row['Lat'], 4)
                lon = round(row['Lon'], 4)
                out_val = row[out_col]
                
                if (lat, lon) in nc_lookup:
                    point_idx = nc_lookup[(lat, lon)]
                    
                    # Get time index
                    if 'Day' in out_df.columns:
                        time_idx = int(row['Day']) - 1
                    elif 'Month' in out_df.columns:
                        time_idx = int(row['Month']) - 1
                    elif 'Mth' in out_df.columns:
                        time_idx = int(row['Mth']) - 1
                    else:
                        time_idx = 0
                    
                    # Handle different array shapes
                    if len(nc_data.shape) == 2:
                        if time_idx < nc_data.shape[0] and point_idx < nc_data.shape[1]:
                            nc_val = nc_data[time_idx, point_idx]
                        else:
                            continue
                    elif len(nc_data.shape) == 1:
                        if point_idx < nc_data.shape[0]:
                            nc_val = nc_data[point_idx]
                        else:
                            continue
                    else:
                        continue
                    
                    if np.isnan(out_val) and np.isnan(nc_val):
                        matches += 1
                    elif np.isclose(out_val, nc_val, rtol=1e-4, atol=1e-6):
                        matches += 1
                    else:
                        mismatches += 1
                        diff = abs(out_val - nc_val) if not np.isnan(nc_val) else float('inf')
                        if diff != float('inf'):
                            diffs.append(diff)
                            max_diff = max(max_diff, diff)
        
        total_compared = matches + mismatches
        if total_compared > 0:
            match_rate = matches / total_compared * 100
            result['data_match'] = match_rate > 99.0
            result['max_diff'] = max_diff if diffs else 0
            result['mean_diff'] = np.mean(diffs) if diffs else 0
            result['details'].append(f"Compared {total_compared} values: {matches} match, {mismatches} differ ({match_rate:.2f}% match)")
            
            if match_rate >= 99.0:
                result['status'] = 'PASS'
            elif match_rate >= 90.0:
                result['status'] = 'WARN'
            else:
                result['status'] = 'FAIL'
        else:
            result['status'] = 'WARN'
            result['details'].append("Could not compare any values (coordinate mismatch)")
            
    except Exception as e:
        result['status'] = 'ERROR'
        result['details'].append(f"Comparison error: {str(e)}")
    
    nc_ds.close()
    return result


def main():
    parser = argparse.ArgumentParser(description='Validate .out to .nc conversion')
    parser.add_argument('--out-dir', '-i', required=True, help='Directory with .out files')
    parser.add_argument('--nc-dir', '-n', required=True, help='Directory with .nc files')
    parser.add_argument('--year', '-y', type=int, default=1365, help='Year to validate (default: 1365)')
    parser.add_argument('--output', '-o', default='validation_results.txt', help='Output file for results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    print(f"Validating conversion for year {args.year}")
    print(f"  .out directory: {args.out_dir}")
    print(f"  .nc directory: {args.nc_dir}")
    
    # Find file pairs
    pairs = get_file_pairs(args.out_dir, args.nc_dir, args.year)
    print(f"Found {len(pairs)} file pairs to validate")
    
    if not pairs:
        print("No matching file pairs found!")
        return 1
    
    # Validate each pair
    results = []
    for out_file, nc_file in pairs:
        print(f"  Validating: {os.path.basename(out_file)} ...", end=' ')
        result = validate_file_pair(out_file, nc_file, args.verbose)
        results.append(result)
        print(result['status'])
    
    # Write results to file
    with open(args.output, 'w') as f:
        f.write(f"LPJG2NC Conversion Validation Report\n")
        f.write(f"=====================================\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Year validated: {args.year}\n")
        f.write(f"Source directory: {args.out_dir}\n")
        f.write(f"Output directory: {args.nc_dir}\n")
        f.write(f"\n")
        
        # Summary
        pass_count = sum(1 for r in results if r['status'] == 'PASS')
        warn_count = sum(1 for r in results if r['status'] == 'WARN')
        fail_count = sum(1 for r in results if r['status'] == 'FAIL')
        error_count = sum(1 for r in results if r['status'] == 'ERROR')
        
        f.write(f"SUMMARY\n")
        f.write(f"-------\n")
        f.write(f"Total files validated: {len(results)}\n")
        f.write(f"  PASS:  {pass_count}\n")
        f.write(f"  WARN:  {warn_count}\n")
        f.write(f"  FAIL:  {fail_count}\n")
        f.write(f"  ERROR: {error_count}\n")
        f.write(f"\n")
        
        # Detailed results
        f.write(f"DETAILED RESULTS\n")
        f.write(f"----------------\n")
        for r in results:
            f.write(f"\n{r['out_file']}\n")
            f.write(f"  Status: {r['status']}\n")
            f.write(f"  .out points: {r['out_points']}, times: {r['out_times']}\n")
            f.write(f"  .nc points: {r['nc_points']}, times: {r['nc_times']}\n")
            if r['max_diff'] is not None:
                f.write(f"  Max diff: {r['max_diff']:.6e}, Mean diff: {r['mean_diff']:.6e}\n")
            for detail in r['details']:
                f.write(f"  - {detail}\n")
        
        f.write(f"\n--- End of Report ---\n")
    
    print(f"\nResults written to: {args.output}")
    print(f"Summary: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL, {error_count} ERROR")
    
    return 0 if fail_count == 0 and error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
