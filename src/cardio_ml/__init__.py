"""
Package cardio_ml — machine learning engineering for CVD classification.

Importing this package automatically applies the resource usage policy
(BLAS thread limits, process priority) defined in `config.py`. This
ensures that any entry point (CLI, API, notebook, tests) respects the
CPU budget without repeating the setup.
"""

from cardio_ml import config as _config

# Apply the resource policy once, on the first import of the package.
_config.apply_resource_policy()

__version__ = "0.1.0"
__all__ = ["__version__"]
