from __future__ import annotations
from typing import Generic, Sequence, Iterator, TypeVar, cast
import functools, re, pathlib, contextlib, inspect, os, atexit
from picograd.engine.compiler import Renderer
from picograd.helpers import ALLOW_DEVICE_USAGE, DEBUG, LRU, getenv

# picograd to tinygrad bridge
# - removed deviceregistry (picograd's M:N support is limited to cuda and hip)
# - removed lruallocator and bufferspec (no need to support advanced allocation options for now)
# - removed llvmcompiler (requires llvmlite or ffi-llvmctypes)
# - removed imagedtype (no need to support imagedtypes for now)

ALL_DEVICES = ["CPU", "CL", "HIP", "CUDA"]
DeviceType = TypeVar('DeviceType', bound='Runtime')

class _Device:
  """
  device registry which maps device strings to device Runtimes
  """
  def __init__(self) -> None:
    self._devices = [x.stem[len("ops_"):].upper() for x in (pathlib.Path(__file__).parent/"runtime").iterdir() if x.stem.startswith("ops_")]
    self._opened_devices:set[str] = set()

  def __getitem__(self, ix:str) -> Runtime: return self.__get_canonicalized_item(self.canonicalize(ix))
  @functools.cache  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def __get_canonicalized_item(self, ix:str) -> Runtime:
    assert ALLOW_DEVICE_USAGE or ix.split(":")[0] in ["DISK", "TINYFS", "NPY", "PYTHON"], f"usage of device {ix} disallowed"
    base = (__package__ or __name__).split('.')[0]  # tinygrad
    x = ix.split(":")[0].lower()
    ret = [cls for cname, cls in inspect.getmembers(importlib.import_module(f'{base}.runtime.ops_{x}')) \
           if (cname.lower() == x + "device")][0](ix)
    if DEBUG >= 1: print(f"opened device {ix} from pid:{os.getpid()}")
    self._opened_devices.add(ix)
    return ret
  
  def canonicalize(self, device:str|None) -> str: return self._canonicalize(device if device is not None else Device.DEFAULT) # NOTE: you can't cache canonicalize in case Device.DEFAULT changes
  @functools.cache  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def _canonicalize(self, device:str) -> str: return re.sub(r":0$", "", (d:=device.split(":", 1)[0].upper()) + device[len(d):])

  @property
  def default(self) -> Runtime: return self[self.DEFAULT]
  def get_available_devices(self) -> Iterator[str]:
    for device in ALL_DEVICES:
      with contextlib.suppress(Exception): yield self[device].device

  @functools.cached_property
  def DEFAULT(self) -> str:
    dev = [dev] if (dev:=getenv("DEV", "").upper()) else []
    from_env = dedup(dev + [d for d in self._devices if d not in ["DISK", "TINYFS", "NPY"] and getenv(d) == 1])
    assert len(from_env) < 2, f"multiple devices set in env: {from_env}"
    if len(from_env) == 1: return from_env[0]
    try:
      device = next(self.get_available_devices())
      os.environ[device] = "1"   # we set this in environment for spawned children
      return device
    except StopIteration as exc: raise RuntimeError("no usable devices") from exc

Device: _Device = _Device()
atexit.register(lambda: [Device[dn].finalize() for dn in Device._opened_devices])

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

# TODO: size, dest, src are the same type. can we enforce this?
class Allocator(Generic[DeviceType]):
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

class Compiler:
  def __init__(self): None # TODO (picograd jit compile cache): cachekey:str|None=None): # self.cachekey = None if DISABLE_COMPILER_CACHE else cachekey
  def compile(self, src:str) -> bytes: return src.encode()   # NOTE: empty compiler is the default
  def disassemble(self, lib:bytes): pass

class CompileError(Exception): pass
CompilerPairT = tuple[functools.partial|type[Renderer], functools.partial|type[Compiler]]