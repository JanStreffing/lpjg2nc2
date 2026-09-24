#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variable metadata (long_name, units) for LPJ-GUESS output files.

The metadata come from the output file descriptions in LPJ-GUESS'
modules/{CMIPoutput,commonoutput,miscoutput}.cpp, e.g.

    declare_parameter("file_baresoilFrac_yearly", ..., "Gridcell fraction covered by bare soils [-]");
    declare_parameter("file_cmass", ..., "C biomass output file");

which give long_name "Gridcell fraction covered by bare soils" and units
"-" for baresoilFrac_yearly.out, and long_name "C biomass" for cmass.out.
The .out file names are set in the .ins files (e.g.
file_runoff "tot_runoff.out"), so these are read as well to map each
parameter to its .out file(s). The result is stored in output_metadata.json
next to this module. Regenerate it after the sources or .ins files changed
with

    python -m lpjg2nc.metadata \\
        /path/to/lpj_guess/modules/{CMIPoutput,commonoutput,miscoutput}.cpp \\
        --ins /path/to/esm_tools/namelists/lpj_guess/*.ins*
"""

import os
import re
import glob
import json
import argparse

METADATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_metadata.json')

_metadata = None


def parse_output_source(cpp_file):
    """
    Parse the file_* parameter descriptions of an LPJ-GUESS output module.

    Descriptions of the form "Long name [units]" (CMIPoutput.cpp) give
    long_name and units. Otherwise the description is the long_name, without
    "output file" and a trailing "output" (commonoutput.cpp, miscoutput.cpp).
    Descriptions that are empty after that (e.g. "Daily output.") are skipped.

    Parameters
    ----------
    cpp_file : str
        Path to e.g. LPJ-GUESS' modules/CMIPoutput.cpp.

    Returns
    -------
    dict
        Parameter name without "file_" (e.g. 'baresoilFrac_yearly') ->
        {'long_name': ..., 'units': ...}; 'units' is missing if the
        description has no trailing [units].
    """
    with open(cpp_file, encoding='utf-8', errors='replace') as f:
        src = f.read()
    metadata = {}
    for name, desc in re.findall(
            r'declare_parameter\(\s*"file_(\w+)"\s*,\s*&\w+\s*,\s*\d+\s*,\s*"((?:[^"\\]|\\.)*)"\s*\)', src):
        # "Long name [units]"; some descriptions repeat the units bracket
        m = re.fullmatch(r'(.*?)\s*((?:\[[^\]]*\]\s*)+)', desc)
        if m:
            units = re.findall(r'\[([^\]]*)\]', m.group(2))[-1].strip()
            metadata[name] = {'long_name': m.group(1).strip(), 'units': units}
            continue
        # "Soil temperature output file (5cm depth)" -> "Soil temperature (5cm depth)"
        long_name = re.sub(r'\s*\boutput\s+file\b', '', desc.strip(), flags=re.I)
        long_name = re.sub(r'\s*\boutput\s*\.?\s*$', '', long_name, flags=re.I)
        if long_name and long_name.lower() not in ('daily', 'monthly', 'annual'):
            metadata[name] = {'long_name': long_name}
    return metadata


def parse_ins_file_names(ins_files):
    """
    Read the .out file names that .ins files assign to file_* parameters.

    Parameters
    ----------
    ins_files : list of str
        .ins files (or .ins.j2 templates), e.g. lpjg_output.ins.

    Returns
    -------
    dict
        Parameter name without "file_" (e.g. 'runoff') -> set of .out
        basenames without extension (e.g. {'tot_runoff'}). Commented
        ('!file_...') assignments count too, since they may be enabled.
    """
    names = {}
    for ins_file in ins_files:
        with open(ins_file, encoding='utf-8', errors='replace') as f:
            src = f.read()
        for name, out_file in re.findall(r'^\s*!?\s*file_(\w+)\s+"([^"]+)"', src, re.M):
            stem = os.path.splitext(os.path.basename(out_file))[0]
            if stem:
                names.setdefault(name, set()).add(stem)
    return names


def build_metadata(cpp_files, ins_files=()):
    """
    Build the .out basename -> {'long_name', 'units'} table.

    Every parameter is stored under its own name (the .out basename unless
    an .ins file says otherwise) and under every .out basename the .ins
    files give it. Conflicting entries for one basename are reported and
    the first one is kept.
    """
    params = {}
    for cpp_file in cpp_files:
        for name, attrs in parse_output_source(cpp_file).items():
            if name in params and params[name] != attrs:
                print(f"Warning: file_{name} described in several sources; keeping the first: {params[name]}")
                continue
            params.setdefault(name, attrs)
    ins_names = parse_ins_file_names(ins_files)
    metadata = {}
    for name, attrs in params.items():
        for stem in [name] + sorted(ins_names.get(name, set()) - {name}):
            if stem in metadata and metadata[stem] != attrs:
                print(f"Warning: {stem}.out matches several parameters; keeping the first: {metadata[stem]}")
                continue
            metadata[stem] = attrs
    return metadata


def get_file_metadata(out_file):
    """
    Look up long_name and units for an LPJ-GUESS .out file.

    Parameters
    ----------
    out_file : str
        Path or basename of the .out file, e.g. 'baresoilFrac_yearly.out'.

    Returns
    -------
    dict
        {'long_name': ..., 'units': ...} (either may be missing), or an
        empty dict if the file is not known.
    """
    global _metadata
    if _metadata is None:
        try:
            with open(METADATA_FILE) as f:
                _metadata = json.load(f)
        except (OSError, ValueError) as e:
            print(f"Warning: could not read variable metadata from {METADATA_FILE}: {e}")
            _metadata = {}
    name = os.path.splitext(os.path.basename(out_file))[0]
    return dict(_metadata.get(name, {}))


def add_variable_metadata(ds, var_names, out_file):
    """
    Set long_name and units of var_names in ds from the metadata of out_file.

    All variables of a file (e.g. one per PFT column) get the same
    attributes, since LPJ-GUESS describes files, not columns. Variables of
    unknown files are left unchanged.
    """
    attrs = get_file_metadata(out_file)
    if not attrs:
        return
    for var_name in var_names:
        if var_name in ds:
            ds[var_name].attrs.update(attrs)


def main():
    parser = argparse.ArgumentParser(
        prog='python -m lpjg2nc.metadata',
        description=f'Regenerate {os.path.basename(METADATA_FILE)} from LPJ-GUESS output modules and .ins files.')
    parser.add_argument('cpp_files', nargs='+',
                        help='LPJ-GUESS output modules, e.g. modules/{CMIPoutput,commonoutput,miscoutput}.cpp')
    parser.add_argument('--ins', nargs='+', default=[],
                        help='.ins files or directories of them that set the .out file names')
    args = parser.parse_args()
    ins_files = []
    for path in args.ins:
        ins_files += sorted(glob.glob(os.path.join(path, '*.ins*'))) if os.path.isdir(path) else [path]
    metadata = build_metadata(args.cpp_files, ins_files)
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=1, sort_keys=True)
        f.write('\n')
    n_units = sum('units' in v for v in metadata.values())
    print(f"Wrote {len(metadata)} entries ({n_units} with units) from {len(args.cpp_files)} sources "
          f"and {len(ins_files)} .ins files to {METADATA_FILE}")


if __name__ == '__main__':
    main()
