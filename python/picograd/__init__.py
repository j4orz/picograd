from importlib import import_module as _import_module
print("initializing picograd")
print("importing picograd._pgrs")
_pgrs = _import_module("picograd._pgrs")

from . import nn
from .tensor import Tensor # language
from .device import Device # runtime
__all__ = ["nn", "Tensor", "Device"]
