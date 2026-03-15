from importlib.metadata import version, PackageNotFoundError
import importlib

DIST_NAME = "qrl-qai"
try:
    __version__ = version(DIST_NAME)
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

__all__ = ["__version__", "env", "agents", "algorithms"]

def __getattr__(name):
    if name in ("env", "agents", "algorithms"):
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(name)
