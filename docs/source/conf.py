# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

# Ensure qrl is importable
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "qrl-qai"
copyright = "2025, Jay Shah"
author = "Jay Shah"
release = "0.3.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "nbsphinx",
]

autosummary_generate = True

nbsphinx_execute = "never"


# Mock heavy deps to avoid crashes on RTD
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "torchaudio",
    "pennylane",
    "gymnasium",
    "pygame",
    "cv2",
    "moviepy",
    "imageio_ffmpeg",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    'collapse_navigation': False,  # Keep navigation expanded
    'navigation_depth': -1,  # Show all levels (or set to specific depth like 4)
    'titles_only': False,  # Show all entries, not just titles
    # "sticky_navigation": True,
}

html_static_path = ["_static"]
