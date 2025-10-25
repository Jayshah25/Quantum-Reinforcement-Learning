# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os, sys

# Ensure qrl is importable
sys.path.insert(0, os.path.abspath("../.."))

# Project info
project = 'qrl-qai'
copyright = '2025, Jay Shah'
author = 'Jay Shah'
release = '0.2.0'

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True

# Mock heavy deps to avoid crashes on RTD
autodoc_mock_imports = [
    "torch", "torchvision", "torchaudio",
    "pennylane",
    "gymnasium", "pygame",
    "cv2", "moviepy", "imageio_ffmpeg",
]


# HTML theme
html_theme = "furo"  # or "sphinx_rtd_theme"



# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
