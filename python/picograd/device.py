from __future__ import annotations
from typing import Any, Generic, Sequence, Iterator, TypeVar, cast
import functools
from dataclasses import dataclass
from collections import defaultdict

from picograd.dtype import DType, PtrDType
from picograd.engine.compiler import Renderer
from picograd.helpers import DEBUG, LRU, getenv, select_first_inited, unwrap_class_type

ALL_DEVICES = ["CPU", "HIP", "CUDA"]
DeviceType = TypeVar('DeviceType', bound='Runtime')

# **************** Runtime: Host Allocators + Device Compilers ****************
class Runtime:
  """
  the Runtime base class is the heterogenous runtime dual to the domain specific ndarray language of the Tensor object
  picograd's runtime implementations will subclass Runtime to provide memory and compute management used by both interpreter(pt1) and compiler(pt2) pipelines
  - Buffer Allocator
  - Kernel Compiler
  """
  # TODO (picograd profiling): profile_events:list[ProfileEvent] = [ProfileDeviceEvent("CPU")] # NOTE: CPU is the default device.

  def __init__(self, device:str, allocator:Allocator, compilers:Sequence[CompilerPairT]|None, kernel, graph=None, group_id=None):
    self.device, self.allocator, self.kernel, self.graph, self.group_id = device, allocator, kernel, graph, group_id
    self.compilers = cast(list[CompilerPairT], compilers or [(Renderer, Compiler)])

    envnames = [self._get_compiler_envvar(c) for r,c in self.compilers]
    enable_comps = set((en, comp_pair) for en, comp_pair in zip(envnames, self.compilers) if en is not None and getenv(en, -1) == 1)
    disable_comps = set((en, comp_pair) for en, comp_pair in zip(envnames, self.compilers) if en is not None and getenv(en, -1) == 0)

    if len(enable_comps) > 1: raise RuntimeError(f"{self.device}: multiple compilers set in env {enable_comps}")
    for _, comp_pair in disable_comps: self.compilers.remove(comp_pair)

    self.renderer, self.compiler = select_first_inited([list(enable_comps)[0][1]] if len(enable_comps) == 1 else self.compilers,
                                                       f"No compiler for {self.device} is available")

    if DEBUG >= 1: print(f"{self.device}: using {self.compiler.__class__.__name__}")

  def _get_compiler_envvar(self, c):
    compiler_name = f"{unwrap_class_type(c).__name__.upper().removesuffix('COMPILER').removeprefix(devname:=self.device.split(':')[0].upper())}"
    return f"{devname}_{compiler_name if len(compiler_name) > 0 else unwrap_class_type(c).__name__.upper()}"

  def synchronize(self):
    """
    Synchronize all pending operations on the device.

    This method ensures that all previously queued operations on the device have been completed before proceeding.
    """
    # override this in your device implementation
  def _at_profile_finalize(self):
    """
    Called at the end of profiling to allow the device to finalize any profiling.
    """
    # override this in your device implementation
  def finalize(self):
    """
    Called at the end of process lifetime to allow the device to finalize.
    """
    # override this in your device implementation

# **************** Host Memory Allocation ****************
@dataclass(frozen=True, eq=True)
class BufferSpec:
  # TODO: move device, size, dtype here?
  uncached: bool = False
  cpu_access: bool = False
  host: bool = False
  nolru: bool = False
  external_ptr: int|None = None

class Buffer:
  """
  moose
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
  moose
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

# TODO: picograd lru buffer cache
# class LRUAllocator(Allocator, Generic[DeviceType]):

# **************** Device Compute Compilation ****************

class Compiler:
  def __init__(self): None # TODO (picograd jit compile cache): cachekey:str|None=None): # self.cachekey = None if DISABLE_COMPILER_CACHE else cachekey
  def compile(self, src:str) -> bytes: return src.encode()   # NOTE: empty compiler is the default
  def disassemble(self, lib:bytes): pass

# TODO (picograd amdllvmcompiler)
# class LLVMCompiler(Compiler):

class CompileError(Exception): pass
CompilerPairT = tuple[functools.partial|type[Renderer], functools.partial|type[Compiler]]