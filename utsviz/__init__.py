# The version is written by setuptools-scm at build/install time and read back
# from the installed package metadata, replacing the generated _version.py that
# versioneer used to keep in the source tree.
from importlib.metadata import version as _version, \
    PackageNotFoundError as _PackageNotFoundError

try:
    __version__ = _version('utsviz')
except _PackageNotFoundError:  # running from a source tree, never installed
    __version__ = 'unknown'
del _version, _PackageNotFoundError

from .plots.core import *
from .colormaps import *
