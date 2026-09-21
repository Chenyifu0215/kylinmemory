"""Compatibility imports backed by the authoritative source implementation."""
from ._vendor.tools.interrupt import *
from ._vendor.tools.interrupt import __dict__ as _source

def __getattr__(name):
    try:
        return _source[name]
    except KeyError:
        raise AttributeError(name) from None
