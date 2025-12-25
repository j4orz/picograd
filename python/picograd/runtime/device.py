from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Generic, Self, Sequence, Iterator, TypeVar, cast
import functools,  atexit, re, pathlib, contextlib, importlib, inspect, os, ctypes
from picograd.dtype import DType, PtrDType
from picograd.engine.compiler import Renderer
from picograd import helpers
from picograd.helpers import ALLOW_DEVICE_USAGE, DEBUG, LRU, MAX_BUFFER_SIZE, T, getenv

# picograd to tinygrad bridge
# - removed Device.Default
# - removed lruallocator and bufferspec (no need to support advanced allocation options for now)
# - removed llvmcompiler (requires llvmlite or ffi-llvmctypes)
# - removed imagedtype (no need to support imagedtypes for now)

# **************** Runtime: Memory (Buffer Allocators) + Compute (Kernel Compilers) ****************
class Runtime:
  """
  Runtime is a base class which wires up
  1. memory (Buffer Allocator) and
  2. compute (Kernels, and Kernel Compilers)
  """
  # profile_events:list[ProfileEvent] = [ProfileDeviceEvent("CPU")] # NOTE: CPU is the default device.

  def __init__(self, device: str, allocator: Allocator, compilers: CompilerSet|None, kernel): # graph=None, group_id=None):
    self.device, self.allocator, self.kernel, self.graph, self.group_id = device, allocator, kernel # , graph, group_id

    self.comps_ctrl_var = compilers.ctrl_var if compilers is not None else None
    self.comp_sets: dict[Any, tuple[ContextVar|None, tuple[type[Renderer]|functools.partial, type[Compiler]|functools.partial]]] = {}
    self.cached_pair: dict[Any, tuple[Renderer, Compiler]] = {}
    for cpair in (compilers.cset if compilers is not None else [CompilerPair(Renderer, Compiler)]):
      self.comp_sets[self._compiler_name(cpair.compiler)] = (cpair.ctrl_var, (cpair.renderer, cpair.compiler))

  @property
  def renderer(self) -> Renderer: return self._select_compiler_pair()[0]

  @property
  def compiler(self) -> Compiler: return self._select_compiler_pair()[1]

  def _compiler_name(self, c:type[Compiler]|functools.partial) -> str:
    return unwrap_class_type(c).__name__.upper().removesuffix("COMPILER").removeprefix(devname:=self.device.split(':')[0].upper()) or devname

  def _select_compiler_pair(self) -> tuple[Renderer, Compiler]:
    # select forced compiler from global env var.
    forced_comps = set([self.comp_sets[val][1]] if self.comps_ctrl_var is not None and (val:=self.comps_ctrl_var.value) else [])

    # add forced compilers from individual env vars (only if global env var is not set, as it takes precedence).
    if not forced_comps: forced_comps |= set(rc for en, rc in self.comp_sets.values() if en is not None and en.value == 1)
    if len(forced_comps) > 1: raise RuntimeError(f"{self.device}: multiple compilers set in env {forced_comps}")

    # select remaining compilers (all or forced only)
    comps = list(rc for en, rc in self.comp_sets.values())

    # remove disabled compilers
    for en, rc in self.comp_sets.values():
      if en is not None and en.value == 0 and rc in comps: comps.remove(rc)

    return select_first_inited(list(forced_comps) if len(forced_comps)>0 else comps, f"No compiler for {self.device} is available", self.cached_pair)

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

ALL_DEVICES = ["CUDA", "HIP"] # "CPU", "CL"/"MOJO"
DeviceType = TypeVar('DeviceType', bound='Runtime')
class _RuntimeRegistry:
  """
  device registry which maps device strings to device Runtimes
  """
  def __init__(self) -> None:
    self._devices = [path.stem[len("ops_"):].upper() for path in (pathlib.Path(__file__).parent).iterdir() if path.stem.endswith("runtime.py")]
    self._opened_devices:set[str] = set()
  def __getitem__(self, device:str) -> Runtime: return self.canonicalize_runtime(self.canonicalize_device(device))

  def canonicalize_device(self, device:str) -> str: return re.sub(r":0$", "", (d:=device.split(":", 1)[0].upper()) + device[len(d):])
  def canonicalize_runtime(self, canonicalized_device:str) -> Runtime:
    assert ALLOW_DEVICE_USAGE or canonicalized_device.split(":")[0] in ["DISK", "TINYFS", "NPY", "PYTHON"], f"usage of device {canonicalized_device} disallowed"
    package_name = (__package__ or __name__).split('.')[0]
    canonicalized_device_normalized = canonicalized_device.split(":")[0].lower()
    foo = inspect.getmembers(importlib.import_module(f'{package_name}.runtime.{canonicalized_device_normalized}_runtime'))
    output = [cls for cname, cls in foo if (cname.lower() == canonicalized_device_normalized + "device")][0](canonicalized_device)
    if DEBUG >= 1: print(f"opened device {canonicalized_device} from pid:{os.getpid()}")
    self._opened_devices.add(canonicalized_device)
    return output

Device = _RuntimeRegistry()
atexit.register(lambda: [Device[dn].finalize() for dn in Device._opened_devices])

# **************** MEMORY: (Buffer Allocators) ****************
class Buffer:
  """
  Buffer provides an on-device handle of an OpNode's backing storage
  picograd follows tinygrad's bent towards object-oriented organization where containers are lazily initialied
  i.e Tensor, OpNode, and Buffer are all non-allocating/evalauting/materializing, and only do so on Tensor.evaluate(), OpNode.evaluate(), and a Buffer.allocate()
  """
  def __init__(self, device:str, dtype:DType, size:int,
               buf_opaque:Any=None, initial_value: bytes|None=None,
               base:Buffer|None=None, offset:int=0, preallocate=False,
               opnode_refcount=0): #options:BufferSpec|None=None,):
    assert isinstance(dtype, DType) and not isinstance(dtype, PtrDType)
    self.device, self.size, self.dtype, = device, size, dtype
    self.offset, self.allocated_views = offset, 0
    if DEBUG >=1: print("initializing Buffer on device", device)

    if base is not None:
      assert base._basebuf is None, "base can't have a base"
      assert device == base.device, "base must have the same device"
      self._basebuf = base
    else:
      assert offset == 0, "base buffers can't have offset"
      self._basebuf, self._opnode_refcount = None, opnode_refcount
      if buf_opaque is not None: self.allocate(buf_opaque) # if DEBUG >=1: print("allocated the Buffer with an opaque buf", buf_opaque)
      if initial_value is not None: self.allocate(), self.copyin(memoryview(initial_value))
    if preallocate: self.allocate()
    if DEBUG >=1: print("DONE initializing Buffer on device", device)

  @property
  def base(self) -> Buffer: return self._basebuf if self._basebuf is not None else self
  def ref(self, count):
    self.base._opnode_refcount += count
    return self

  def allocate(self, opaque=None, external_ptr=None) -> Self:
    assert not self.is_initialized(), "can't allocate already allocated buffer"
    if DEBUG >= 7: print(f"buffer: allocate {self.nbytes} bytes on {self.device}")
    if not self.device.startswith("NULL") and self.size > MAX_BUFFER_SIZE > 0: raise RuntimeError(f"buffer of size {self.size/1e6:.2f}M is too large")

    print("allocating the initialized Buffer on device", self.device)
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

# TODO: size, dest, src are the same type. can we enforce this?
class Allocator(Generic[DeviceType]):
  """
  
  """
  def __init__(self, dev:DeviceType):
    self.dev: DeviceType = dev
    self.supports_copy_from_disk: bool = True # self.default_buffer_spec: BufferSpec = BufferSpec()

  # required
  def _alloc(self, size:int): raise NotImplementedError("need alloc") # options:BufferSpec): raise NotImplementedError("need alloc")
  def _free(self, opaque): pass # options:BufferSpec): pass  # if opaque is a Python object, you don't need a free
  def _copyin(self, dest, src:memoryview): raise NotImplementedError("need copyin")
  def _copyout(self, dest:memoryview, src): raise NotImplementedError("need copyout")

  # provided
  def alloc(self, size:int): #, options:BufferSpec|None=None):
    assert size > 0, f"alloc size must be positive, getting {size}"
    return self._alloc(size)#, options if options is not None else self.default_buffer_spec)
  def free(self, opaque, size:int): # , options:BufferSpec|None=None):
    self._free(opaque) #, options if options is not None else self.default_buffer_spec)

def from_mv(mv:memoryview, to_type:type[ctypes._SimpleCData]=ctypes.c_char) -> ctypes.Array:
  return ctypes.cast(ctypes.addressof(to_type.from_buffer(mv)), ctypes.POINTER(to_type * len(mv))).contents
def to_mv(ptr:int, sz:int) -> memoryview: return memoryview((ctypes.c_uint8 * sz).from_address(ptr)).cast("B")
def mv_address(mv): return ctypes.addressof(ctypes.c_char.from_buffer(mv))
def flat_mv(mv:memoryview): return mv if len(mv) == 0 else mv.cast("B", shape=(mv.nbytes,))

# **************** COMPUTE: (Kernel Compilers) ****************
class Compiler:
  def __init__(self): None # TODO (picograd jit compile cache): cachekey:str|None=None): # self.cachekey = None if DISABLE_COMPILER_CACHE else cachekey
  def compile(self, src:str) -> bytes: return src.encode()   # NOTE: empty compiler is the default
  def disassemble(self, lib:bytes): pass

class CompileError(Exception): pass
@dataclass(frozen=True)
class CompilerPair: renderer:type[Renderer]|functools.partial; compiler:type[Compiler]|functools.partial; ctrl_var:ContextVar|None = None # noqa: E702

@dataclass(frozen=True)
class CompilerSet: cset:list[CompilerPair]; ctrl_var:ContextVar|None = None # noqa: E702
