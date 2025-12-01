from __future__ import annotations
from typing import Any, Generic, Self, Sequence, Iterator, TypeVar, cast
import functools,  atexit, re, pathlib, contextlib, importlib, inspect, os, ctypes
from picograd.dtype import DType, PtrDType
from picograd.engine.compiler import Renderer
from picograd.helpers import ALLOW_DEVICE_USAGE, DEBUG, LRU, getenv

# picograd to tinygrad bridge
# - removed Device.Default
# - removed lruallocator and bufferspec (no need to support advanced allocation options for now)
# - removed llvmcompiler (requires llvmlite or ffi-llvmctypes)
# - removed imagedtype (no need to support imagedtypes for now)

ALL_DEVICES = ["HIP", "CUDA"] # "CPU", "CL", "MOJO"
DeviceType = TypeVar('DeviceType', bound='Runtime')

class _Device:
  """
  device registry which maps device strings to device Runtimes
  """
  def __init__(self) -> None:
    self._devices = [x.stem[len("ops_"):].upper() for x in (pathlib.Path(__file__).parent).iterdir() if x.stem.endswith("runtime.py")]
    self._opened_devices:set[str] = set()
  def __getitem__(self, ix:str) -> Runtime: return self.canonicalized_runtime(self.canonicalize_device(ix))
  def canonicalize_device(self, device:str) -> str: return re.sub(r":0$", "", (d:=device.split(":", 1)[0].upper()) + device[len(d):])
  def canonicalized_runtime(self, ix:str) -> Runtime:
    assert ALLOW_DEVICE_USAGE or ix.split(":")[0] in ["DISK", "TINYFS", "NPY", "PYTHON"], f"usage of device {ix} disallowed"
    base = (__package__ or __name__).split('.')[0]  # tinygrad
    x = ix.split(":")[0].lower()
    output = [cls for cname, cls in inspect.getmembers(importlib.import_module(f'{base}.runtime.ops_{x}')) if (cname.lower() == x + "device")][0](ix)
    if DEBUG >= 1: print(f"opened device {ix} from pid:{os.getpid()}")
    self._opened_devices.add(ix)
    return output

Device = _Device()
atexit.register(lambda: [Device[dn].finalize() for dn in Device._opened_devices])

# **************** Runtime: Host Allocators + Device Compilers ****************
class Runtime:
  """
  Runtime is a base class which wires up a Buffer Allocator, Kernels, and Compilers for picograd's interpreter and compiler pipelines
  """
  # TODO (picograd profiling): profile_events:list[ProfileEvent] = [ProfileDeviceEvent("CPU")] # NOTE: CPU is the default device.
  def __init__(self, device:str, allocator:Allocator, compilers:Sequence[CompilerPairT]|None, kernel, graph=None, group_id=None):
    self.device, self.allocator, self.kernel, self.graph, self.group_id = device, allocator, kernel, graph, group_id
    self.renderer, self.compiler = compilers
    if DEBUG >= 1: print(f"{self.device}: using {self.compiler.__class__.__name__}")
  def synchronize(self): raise NotImplementedError("need synchronize")

# **************** A Buffer from an Allocator ****************
def from_mv(mv:memoryview, to_type:type[ctypes._SimpleCData]=ctypes.c_char) -> ctypes.Array:
  return ctypes.cast(ctypes.addressof(to_type.from_buffer(mv)), ctypes.POINTER(to_type * len(mv))).contents
def to_mv(ptr:int, sz:int) -> memoryview: return memoryview((ctypes.c_uint8 * sz).from_address(ptr)).cast("B")
def mv_address(mv): return ctypes.addressof(ctypes.c_char.from_buffer(mv))
def flat_mv(mv:memoryview): return mv if len(mv) == 0 else mv.cast("B", shape=(mv.nbytes,))

class Buffer:
  """
  Buffer provides an on-device handle of an OpNode's backing storage with a Runtime's Allocator
  picograd follows tinygrad's bent towards object-oriented organization where
  the Allocator lives *on* the Buffer, rather than an freestanding pure Allocator.allocate() returning a Buffer,
  similar to how the interpreter's evaluator lives *on* the OpNode, rather than a freestanding .eval() returning an OpNode
  """
  def is_initialized(self) -> bool: return self.is_allocated() and hasattr(self, '_buf') # check if the underlying buffer is allocated and the current buffer/view is initialized
  def is_allocated(self) -> bool: return self.base.is_allocated() if self._basebuf is not None else hasattr(self, '_buf') # check if the underlying buffer is allocated, possibly from the base object
  def ensure_allocated(self) -> Self: return self.allocate() if not self.is_initialized() else self
  def copyin(self, mv:memoryview):
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    assert self.is_initialized(), "can't copyin to unallocated buffer"
    self.allocator._copyin(self._buf, mv)
    return self
  def copyout(self, mv:memoryview) -> memoryview:
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    assert self.is_initialized(), "can't copyout unallocated buffer"
    self.allocator._copyout(mv, self._buf)
    return mv

  def __init__(self, device:str, size:int, dtype:DType, buf_opaque:Any=None, initial_value: bytes|None=None,
               options:BufferSpec|None=None, Op_refcount=0, base:Buffer|None=None, offset:int=0, preallocate=False):
    assert isinstance(dtype, DType) and not isinstance(dtype, PtrDType)
    self.device, self.size, self.dtype, = device, size, dtype
    self.options, self.offset, self.allocated_views = options, offset, 0

    if base is None:
      assert offset == 0, "base buffers can't have offset"
      self._basebuf = None
      self._Op_refcount = Op_refcount

      if buf_opaque is not None: self.allocate(buf_opaque)
      if initial_value is not None:
        self.allocate()
        self.copyin(memoryview(initial_value))
    else:
      assert base._basebuf is None, "base can't have a base"
      assert device == base.device, "base must have the same device"
      self._basebuf = base
    if preallocate: self.allocate()

  """
  
  """
  def allocate(self, opaque=None, external_ptr=None) -> Self:
    assert not self.is_initialized(), "can't allocate already allocated buffer"
    if DEBUG >= 7: print(f"buffer: allocate {self.nbytes} bytes on {self.device}")
    if not self.device.startswith("NULL") and self.size > MAX_BUFFER_SIZE > 0: raise RuntimeError(f"buffer of size {self.size/1e6:.2f}M is too large")
    self.allocator: Allocator = Device[self.device].allocator

    if external_ptr is not None:
      self.options = replace(self.options, external_ptr=external_ptr) if self.options else BufferSpec(external_ptr=external_ptr)

    if self._basebuf is None:
      self._buf = opaque if opaque is not None else self.allocator.alloc(self.nbytes, self.options)
    else:
      self._basebuf.ensure_allocated()
      self._basebuf.allocated_views += 1
      assert hasattr(self.allocator, "_offset"), "offset function required for view"
      self._buf: Any = self.allocator._offset(self.base._buf, self.nbytes, self.offset)
      
    return self

# TODO: size, dest, src are the same type. can we enforce this?
class Allocator(Generic[DeviceType]):
  """
  
  """
  def __init__(self, dev:DeviceType):
    self.dev: DeviceType = dev
    self.supports_copy_from_disk: bool = True # self.default_buffer_spec: BufferSpec = BufferSpec()
  def alloc(self, size:int): #, options:BufferSpec|None=None):
    assert size > 0, f"alloc size must be positive, getting {size}"
    return self._alloc(size)#, options if options is not None else self.default_buffer_spec)
  def free(self, opaque, size:int): # , options:BufferSpec|None=None):
    self._free(opaque) #, options if options is not None else self.default_buffer_spec)

  # implemented by the runtime
  def _alloc(self, size:int): raise NotImplementedError("need alloc") # options:BufferSpec): raise NotImplementedError("need alloc")
  def _free(self, opaque): pass # options:BufferSpec): pass  # if opaque is a Python object, you don't need a free
  def _copyin(self, dest, src:memoryview): raise NotImplementedError("need copyin")
  def _copyout(self, dest:memoryview, src): raise NotImplementedError("need copyout")

# **************** Kernels from a Compiler ****************
class Compiler:
  def __init__(self): None # TODO (picograd jit compile cache): cachekey:str|None=None): # self.cachekey = None if DISABLE_COMPILER_CACHE else cachekey
  def compile(self, src:str) -> bytes: return src.encode()   # NOTE: empty compiler is the default
  def disassemble(self, lib:bytes): pass

class CompileError(Exception): pass
CompilerPairT = tuple[functools.partial|type[Renderer], functools.partial|type[Compiler]]