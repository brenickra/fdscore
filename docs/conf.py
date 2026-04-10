import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "fdscore"
author = "Brenick Resende"
copyright = "2026, Brenick Resende"
release = "0.3.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

myst_enable_extensions = ["dollarmath"]

html_theme = "furo"
html_show_sphinx = False

html_theme_options = {
    "sidebar_hide_name": False,
}

autodoc_default_options = {}

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "ndarray": "numpy.ndarray",
    "sequence": "collections.abc.Sequence",
    "Sequence": "collections.abc.Sequence",
    "ERSResult": "fdscore.types.ERSResult",
    "FDSResult": "fdscore.types.FDSResult",
    "FDSTimePlan": "fdscore.types.FDSTimePlan",
    "IterativeInversionParams": "fdscore.types.IterativeInversionParams",
    "Metric": "fdscore.types.Metric",
    "PSDMetricsResult": "fdscore.types.PSDMetricsResult",
    "PSDParams": "fdscore.types.PSDParams",
    "PSDResult": "fdscore.types.PSDResult",
    "SDOFParams": "fdscore.types.SDOFParams",
    "SNParams": "fdscore.types.SNParams",
    "SineDwellSegment": "fdscore.types.SineDwellSegment",
}
napoleon_custom_sections = [
    "Algorithm",
    "Derivation",
    "Inherited assumptions",
    "Metric restriction",
    "Overview",
    "Pipeline",
    "Requirements",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
