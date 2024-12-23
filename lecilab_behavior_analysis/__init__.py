from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lecilab-behavior-analysis")
except PackageNotFoundError:
    # package is not installed
    pass
