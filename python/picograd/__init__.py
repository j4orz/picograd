from importlib import import_module as _import_module
print("initializing picograd")
print("importing picograd._pgrs")
_pgrs = _import_module("picograd._pgrs")

from .tensor import StridedTensor
from . import nn
from .device import Device
__all__ = ["nn", "StridedTensor", "Device"]
