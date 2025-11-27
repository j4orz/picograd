from __future__ import annotations
from typing import Any, Generic, Sequence, Iterator, TypeVar, cast
import functools, ctypes
from dataclasses import dataclass
from collections import defaultdict
from picograd.dtype import DType, PtrDType
from picograd.engine.compiler import Renderer
from picograd.helpers import DEBUG, LRU, getenv, select_first_inited

ALL_DEVICES = ["CPU", "HIP", "CUDA"]
DeviceType = TypeVar('DeviceType', bound='Runtime')

# picograd to tinygrad bridge
# - for memory: removed lruallocator, bufferspec
# - for compute: removed llvmcompiler (requires llvmlite or ffi-llvmctypes)

# **************** Runtime: Host Allocators + Device Compilers ****************
class Runtime:
  """
  the Runtime base class is the heterogenous runtime dual to the domain specific ndarray language of the Tensor object
  which wires up a Buffer Allocator, Kernels, and Compilers for picograd's interpreter and compiler pipelines
  """
  # TODO (picograd profiling): profile_events:list[ProfileEvent] = [ProfileDeviceEvent("CPU")] # NOTE: CPU is the default device.
  def __init__(self, device:str, allocator:Allocator, compilers:Sequence[CompilerPairT]|None, kernel, graph=None, group_id=None):
    self.device, self.allocator, self.kernel, self.graph, self.group_id = device, allocator, kernel, graph, group_id
    self.renderer, self.compiler = compilers
    if DEBUG >= 1: print(f"{self.device}: using {self.compiler.__class__.__name__}")
  def synchronize(self): raise NotImplementedError("need synchronize")

# **************** Host Memory Allocation ****************
class Buffer:
  """
  ...
  """
  def __init__(self, device:str, size:int, dtype:DType, opaque:Any=None, options:BufferSpec|None=None, initial_value:bytes|None=None,
               uop_refcount=0, base:Buffer|None=None, offset:int=0, preallocate=False):
    # if isinstance(dtype, ImageDType): options = BufferSpec(image=dtype) # TODO: image hack shouldn't be here. where should it be?
    # else:
    assert isinstance(dtype, DType) and not isinstance(dtype, PtrDType)
    self.device, self.size, self.dtype, self.options, self.offset, self.allocated_views = device, size, dtype, options, offset, 0
    if base is None:
      assert offset == 0, "base buffers can't have offset"
      self._base = None
      self._uop_refcount = uop_refcount
      if opaque is not None: self.allocate(opaque)
      if initial_value is not None:
        self.allocate()
        self.copyin(memoryview(initial_value))
    else:
      assert base._base is None, "base can't have a base"
      assert device == base.device, "base must have the same device"
      self._base = base
    if preallocate: self.allocate()

# TODO: size, dest, src are the same type. can we enforce this?
class Allocator(Generic[DeviceType]):
  """
  ...
  """
  def __init__(self, dev:DeviceType):
    self.dev: DeviceType = dev
    self.default_buffer_spec: BufferSpec = BufferSpec()
    self.supports_copy_from_disk: bool = True
  def alloc(self, size:int, options:BufferSpec|None=None):
    assert size > 0, f"alloc size must be positive, getting {size}"
    return self._alloc(size, options if options is not None else self.default_buffer_spec)
  def free(self, opaque, size:int, options:BufferSpec|None=None):
    self._free(opaque, options if options is not None else self.default_buffer_spec)

  # implemented by the runtime
  def _alloc(self, size:int, options:BufferSpec): raise NotImplementedError("need alloc")
  def _free(self, opaque, options:BufferSpec): pass  # if opaque is a Python object, you don't need a free
  def _copyin(self, dest, src:memoryview): raise NotImplementedError("need copyin")
  def _copyout(self, dest:memoryview, src): raise NotImplementedError("need copyout")

# **************** Device Compute Compilation ****************

class Compiler:
  def __init__(self): None # TODO (picograd jit compile cache): cachekey:str|None=None): # self.cachekey = None if DISABLE_COMPILER_CACHE else cachekey
  def compile(self, src:str) -> bytes: return src.encode()   # NOTE: empty compiler is the default
  def disassemble(self, lib:bytes): pass

class CompileError(Exception): pass
CompilerPairT = tuple[functools.partial|type[Renderer], functools.partial|type[Compiler]]