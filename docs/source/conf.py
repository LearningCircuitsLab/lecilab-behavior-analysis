# Configuration file for the Sphinx documentation builder.
# Build locally with:  sphinx-build docs/source docs/build
import os
import sys

import setuptools_scm

# the package is installed in the docs build (pip install -e .), and also
# importable from the repository when building locally
sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("."))

project = "lecilab-behavior-analysis"
copyright = "2022, Learning Circuits Lab"
author = "Hernando M Vergara"
try:
    release = setuptools_scm.get_version(root="../..", relative_to=__file__)
except LookupError:
    release = "0.0.0"

# draw the example figures with the current code (from tests/data)
import make_examples  # noqa: E402

make_examples.main()

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_sitemap",
    "myst_parser",
]
myst_enable_extensions = ["colon_fence", "deflist", "linkify", "html_image"]
myst_heading_anchors = 3
templates_path = ["_templates"]
autosummary_generate = True
autodoc_member_order = "bysource"
add_module_names = False
exclude_patterns = ["**.ipynb_checkpoints", "**/*.ipynb", "make_examples.py"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# same look as the Training Village documentation
html_theme = "shibuya"
html_title = "lecilab-behavior-analysis"
html_static_path = ["_static"]
html_theme_options = {
    "color_mode": "light",
    "github_url": "https://github.com/LearningCircuitsLab/lecilab-behavior-analysis",
    "nav_links": [
        {"title": "Getting started", "url": "getting_started"},
        {"title": "Figures", "url": "figures"},
        {"title": "API", "url": "api_index"},
        {"title": "Daily reports", "url": "https://learningcircuitslab.github.io/village-report-generator/documentation/"},
    ],
}
github_user = "LearningCircuitsLab"
html_baseurl = f"https://{github_user}.github.io/{project}/"
sitemap_url_scheme = "{link}"
