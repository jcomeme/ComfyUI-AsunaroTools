"""Top-level package for asunarotools."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """AsunaroTools"""
__email__ = "omemeankohji@gmail.com"
__version__ = "0.0.1"

from .src.asunarotools.nodes import NODE_CLASS_MAPPINGS
from .src.asunarotools.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
