#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Minimal configuration for Sphinx documentation builder to avoid LaTeX issues

# -- Project information -----------------------------------------------------

project = 'lpjg2nc2'
copyright = '2025, Jan Streffing'
author = 'Jan Streffing'
version = '1.0'
release = '1.0.0'

# -- General configuration ---------------------------------------------------

# Completely disable all extensions
extensions = []

# Basic settings
source_suffix = '.rst'
master_doc = 'index'
language = 'en'
exclude_patterns = []

# -- Options for HTML output only ---------------------------------------------

# Use the ReadTheDocs theme
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
}

# Disable static path to avoid file not found errors
html_static_path = []
