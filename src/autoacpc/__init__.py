"""autoacpc - Automatically set brain image origin to AC-PC alignment."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("autoacpc")
except PackageNotFoundError:
    __version__ = "0.6.0"
