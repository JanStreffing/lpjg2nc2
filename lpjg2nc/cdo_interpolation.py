import os
import subprocess
import re

def is_cdo_available():
    """Check if CDO command-line tool is available."""
    try:
        result = subprocess.run(['cdo', '--version'],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              universal_newlines=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def create_grid_file(resolution, xsize=None, ysize=None, output_path=None):
    """Create a grid description file for the specified resolution or grid dimensions.

    Parameters
    ----------
    resolution : float
        Grid resolution in degrees. Can be None if xsize and ysize are provided directly.
    xsize : int, optional
        Number of longitude points in the grid
    ysize : int, optional
        Number of latitude points in the grid
    output_path : str, optional
        Path to save the grid file. If None, creates in script directory.

    Returns
    -------
    str
        Path to the created grid file
    """
    
    # Calculate grid parameters if not provided directly
    if xsize is None or ysize is None:
        if resolution is None:
            raise ValueError("Either resolution or both xsize and ysize must be provided")
        xsize = int(360 / resolution)
        ysize = int(180 / resolution)
    
    # Calculate resolution if not provided
    if resolution is None:
        resolution = 360 / xsize
    
    # Create default output filename if not specified
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, f"grid_{xsize}x{ysize}_global.txt")
    
    # Calculate grid increments and starting points
    xinc = 360 / xsize
    yinc = 180 / ysize
    xfirst = -180 + (xinc / 2)
    yfirst = -90 + (yinc / 2)

    # Create grid file
    with open(output_path, 'w') as f:
        f.write("gridtype = lonlat\n")
        f.write(f"xsize    = {xsize}\n")
        f.write(f"ysize    = {ysize}\n")
        f.write(f"xfirst   = {xfirst}\n")
        f.write(f"xinc     = {xinc}\n")
        f.write(f"yfirst   = {yfirst}\n")
        f.write(f"yinc     = {yinc}\n")

    return output_path


def remap_to_regular_grid(input_file, grid_param, output_file=None, verbose=False):
    """Remap a NetCDF file to a regular global grid using CDO.

    Parameters
    ----------
    input_file : str
        Path to input NetCDF file
    grid_param : str or float
        Either a resolution in degrees (e.g., 0.5, 1, 2) or
        a grid dimension specification in the format 'XxY' (e.g., '360x180')
    output_file : str, optional
        Path to output remapped file. If None, will create an appropriate filename
    verbose : bool, optional
        Whether to print verbose output

    Returns
    -------
    str or None
        Path to remapped file if successful, None otherwise
    """
    if not is_cdo_available():
        print("Error: CDO not available. Cannot perform remapping.")
        return None
    
    # Parse grid parameter
    resolution = None
    xsize = None
    ysize = None
    grid_desc = ""
    
    # Check if the format is dimensions (e.g., 360x180)
    if isinstance(grid_param, str) and 'x' in grid_param.lower():
        match = re.match(r'(\d+)x(\d+)', grid_param.lower())
        if match:
            xsize = int(match.group(1))
            ysize = int(match.group(2))
            # Calculate approximate resolution
            resolution = 360 / xsize
            grid_desc = f"{xsize}x{ysize}"
        else:
            print(f"Invalid grid format: {grid_param}. Expected format: XxY (e.g., 360x180)")
            return None
    else:
        # Assume it's a resolution in degrees
        try:
            resolution = float(grid_param)
            if resolution <= 0:
                print(f"Invalid resolution: {grid_param}. Must be a positive number.")
                return None
            # Calculate grid dimensions based on resolution
            xsize = int(360 / resolution)
            ysize = int(180 / resolution)
            grid_desc = f"{resolution}deg"
        except ValueError:
            print(f"Invalid grid parameter: {grid_param}. Must be a number or in format XxY.")
            return None
    
    # Create default output filename if not specified
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_{grid_desc}{ext}"

    # Create grid file with explicit dimensions if available
    grid_file = create_grid_file(resolution, xsize, ysize)

    # Run CDO to remap (using nearest neighbor interpolation which works with unstructured grids)
    cmd = f"cdo remapnn,{grid_file} {input_file} {output_file}"
    if verbose:
        print(f"Remapping with CDO: {cmd}")

    try:
        result = subprocess.run(cmd,
                              stdout=subprocess.PIPE if not verbose else None,
                              stderr=subprocess.PIPE if not verbose else None,
                              shell=True,
                              universal_newlines=True)

        if result.returncode == 0:
            if verbose:
                print(f"Successfully remapped to {resolution}° grid: {output_file}")
            return output_file
        else:
            print("Error during CDO remapping")
            if verbose and result.stderr:
                print(result.stderr)
            return None
    except Exception as e:
        print(f"Error running CDO: {e}")
        return None
