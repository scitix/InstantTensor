# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'InstantTensor'
copyright = '2026, Yitao Yuan'
author = 'Yitao Yuan'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',       # generate documentation from docstrings
    'sphinx.ext.napoleon',      # support Google/NumPy style docstrings
    'sphinx.ext.viewcode',      # add source code links
    'sphinx.ext.intersphinx',   # link to other project documents
    'sphinx_autodoc_typehints', # automatically handle type hints
]

autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
}

# Control the order of members in autodoc output
# Options: 'alphabetical', 'bysource', 'groupwise'
autodoc_member_order = 'bysource'  # in the order of source code

# Mock C++ extension modules that are not available during documentation build
# This allows Sphinx to generate docs without needing to compile the C++ code
autodoc_mock_imports = [
    'instanttensor._C',  # C++ extension module
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# Alabaster theme options
html_theme_options = {
    'logo': None,
    'logo_name': False,
    'description': '',
    'github_user': None,
    'github_repo': None,
    'github_button': False,
    'github_banner': False,
    'travis_button': False,
    'codecov_button': False,
    'analytics_id': None,
    'note_bg': '#FFF59C',
    'note_border': '#FFE082',
    'sidebar_width': '300px',  # add sidebar width
}

# use short title to avoid line break
html_short_title = 'InstantTensor'
