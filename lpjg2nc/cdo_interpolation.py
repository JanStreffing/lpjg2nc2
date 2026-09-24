import os
import subprocess

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


REMAP_OPERATORS = ('remapdis', 'remaplaf', 'remapnn', 'remapcon')


def remap_to_regular_grid(input_file, remap_spec, output_file=None, verbose=False):
    """Remap a NetCDF file using CDO, from a raw CDO remap spec.

    Parameters
    ----------
    input_file : str
        Path to input NetCDF file
    remap_spec : str
        The CDO operator and grid argument exactly as CDO itself expects
        them, comma-separated, e.g. 'remapcon,r360x180' or
        'remapnn,global_1'. The grid part can be anything CDO understands:
        a built-in grid name, or a path to a grid description file. Used
        verbatim in both the CDO command and the output filename.
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

    if ',' not in remap_spec:
        print(f"Invalid --remap value: {remap_spec}. Expected '<operator>,<grid>', "
              f"e.g. 'remapcon,r360x180' or 'remapnn,global_1'.")
        return None

    operator, grid = remap_spec.split(',', 1)
    if operator not in REMAP_OPERATORS:
        print(f"Invalid remap operator: {operator}. Must be one of {', '.join(REMAP_OPERATORS)}.")
        return None
    if not grid:
        print(f"Invalid --remap value: {remap_spec}. Missing grid after '{operator},'.")
        return None

    # Output filename suffix names both the operator and the grid, exactly
    # as given (only '/' is swapped out, in case grid is a file path).
    grid_desc = f"{operator}_{grid}".replace('/', '_')

    # Create default output filename if not specified
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_{grid_desc}{ext}"

    # Run CDO to remap, passing the grid argument through to CDO as-is
    cmd = f"cdo {operator},{grid} {input_file} {output_file}"
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
                print(f"Successfully remapped with {remap_spec}: {output_file}")
            return output_file
        else:
            print("Error during CDO remapping")
            if verbose and result.stderr:
                print(result.stderr)
            if os.path.exists(output_file):
                os.remove(output_file)  # CDO leaves a partial file behind
            return None
    except Exception as e:
        print(f"Error running CDO: {e}")
        return None
