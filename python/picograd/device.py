from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, Iterator, Sequence

ALL_DEVICES = ["CPU", "CL", "HIP", "CUDA"]

class Device:
  # profile_events:list[ProfileEvent] = [ProfileDeviceEvent("CPU")] # NOTE: CPU is the default device.

  def __init__(self, device:str, allocator:Allocator, compilers:Sequence[CompilerPairT]|None,
               runtime, graph=None, group_id=None):
    self.device, self.allocator, self.runtime = device, allocator, runtime
    self.graph, self.group_id = graph, group_id
    self.compilers = cast(list[CompilerPairT], compilers or [(Renderer, Compiler)])

    # envnames = [self._get_compiler_envvar(c) for r,c in self.compilers]
    # enable_comps = set((en, comp_pair) for en, comp_pair in zip(envnames, self.compilers) if en is not None and getenv(en, -1) == 1)
    # disable_comps = set((en, comp_pair) for en, comp_pair in zip(envnames, self.compilers) if en is not None and getenv(en, -1) == 0)

    # if len(enable_comps) > 1: raise RuntimeError(f"{self.device}: multiple compilers set in env {enable_comps}")
    # for _, comp_pair in disable_comps: self.compilers.remove(comp_pair)

    # try: self.renderer, self.compiler = next(self._get_available_compilers([list(enable_comps)[0][1]] if len(enable_comps) == 1 else self.compilers))
    # except StopIteration as exc: raise RuntimeError(f"no usable compilers for {self.device}") from exc

    # if DEBUG >= 1: print(f"{self.device}: using {self.compiler.__class__.__name__}")

  def _get_compiler_envvar(self, c):
    compiler_name = f"{unwrap_class_type(c).__name__.upper().removesuffix('COMPILER').removeprefix(devname:=self.device.split(':')[0].upper())}"
    return f"{devname}_{compiler_name if len(compiler_name) > 0 else unwrap_class_type(c).__name__.upper()}"

  def _get_available_compilers(self, compilers) -> Iterator[tuple[Renderer, Compiler]]:
    for renderer, compiler in compilers:
      with contextlib.suppress(Exception): yield renderer(), compiler()

  def synchronize(self):
    raise NotImplementedError("todo") # override this in your runtime implementation
    # def _at_profile_finalize(self): # override this in your device implementation
    # def finalize(self): # override this in your device implementation

# **************** Compilers ****************

class Compiler:
  def __init__(self, cachekey:str|None=None): self.cachekey = None if DISABLE_COMPILER_CACHE else cachekey
  def compile(self, src:str) -> bytes: return src.encode()   # NOTE: empty compiler is the default
  def compile_cached(self, src:str) -> bytes:
    if self.cachekey is None or (lib := diskcache_get(self.cachekey, src)) is None:
      assert not getenv("ASSERT_COMPILE"), f"tried to compile with ASSERT_COMPILE set\n{src}"
      lib = self.compile(src)
      if self.cachekey is not None: diskcache_put(self.cachekey, src, lib)
    return lib
  def disassemble(self, lib:bytes): pass
  def __init__(self): raise NotImplementedError("todo")

# **************** Host Allocator, Device Buffers ****************

# TODO: size, dest, src are the same type. can we enforce this?
class Allocator(Generic[DeviceType]):
  def __init__(self, dev:DeviceType):
    self.dev: DeviceType = dev
    self.default_buffer_spec: BufferSpec = BufferSpec()
    self.supports_copy_from_disk: bool = True
  # overridden in LRUAllocator
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

@dataclass(frozen=True, eq=True)
class BufferSpec:
  # TODO: move device, size, dtype here?
  image: ImageDType|None = None
  uncached: bool = False
  cpu_access: bool = False
  host: bool = False
  nolru: bool = False
  external_ptr: int|None = None

class Buffer:
  profile_events:list[ProfileEvent] = []
  def __init__(self, device:str, size:int, dtype:DType, opaque:Any=None, options:BufferSpec|None=None, initial_value:bytes|None=None,
               uop_refcount=0, base:Buffer|None=None, offset:int=0, preallocate=False):
    if isinstance(dtype, ImageDType): options = BufferSpec(image=dtype) # TODO: image hack shouldn't be here. where should it be?
    else: assert isinstance(dtype, DType) and not isinstance(dtype, PtrDType)
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
  @property
  def base(self) -> Buffer: return self._base if self._base is not None else self
  @property
  def uop_refcount(self): return self.base._uop_refcount
  def ref(self, cnt):
    self.base._uop_refcount += cnt
    return self
  # check if the underlying buffer is allocated and the current buffer/view is initialized
  def is_initialized(self) -> bool: return self.is_allocated() and hasattr(self, '_buf')
  # check if the underlying buffer is allocated, possibly from the base object
  def is_allocated(self) -> bool: return self.base.is_allocated() if self._base is not None else hasattr(self, '_buf')
  def ensure_allocated(self) -> Buffer: return self.allocate() if not self.is_initialized() else self
  def allocate(self, opaque=None, external_ptr=None) -> Buffer:
    assert not self.is_initialized(), "can't allocate already allocated buffer"
    if DEBUG >= 7: print(f"buffer: allocate {self.nbytes} bytes on {self.device}")
    if not self.device.startswith("NULL") and self.size > MAX_BUFFER_SIZE > 0: raise RuntimeError(f"buffer of size {self.size/1e6:.2f}M is too large")
    self.allocator:Allocator = Device[self.device].allocator
    if external_ptr is not None:
      self.options = replace(self.options, external_ptr=external_ptr) if self.options else BufferSpec(external_ptr=external_ptr)
    if self._base is not None:
      self._base.ensure_allocated()
      self._base.allocated_views += 1
      assert hasattr(self.allocator, "_offset"), "offset function required for view"
      self._buf: Any = self.allocator._offset(self.base._buf, self.nbytes, self.offset)
    else:
      self._buf = opaque if opaque is not None else self.allocator.alloc(self.nbytes, self.options)
      if not self.device.startswith("DISK"): GlobalCounters.mem_used += self.nbytes
      if PROFILE: Buffer.profile_events.append(ProfilePointEvent(self.device, "alloc", self.trace_num, {"dtype":self.dtype, "sz":self.size}))
    return self
  def deallocate(self):
    assert hasattr(self, '_buf'), "buffer must be allocated to deallocate"
    if DEBUG is not None and DEBUG >= 7: print(f"buffer: deallocate {self.nbytes} bytes on {self.device}")
    if self._base is None and (self.options is None or self.options.external_ptr is None):
      if GlobalCounters is not None and not self.device.startswith("DISK"): GlobalCounters.mem_used -= self.nbytes
      if PROFILE: Buffer.profile_events.append(ProfilePointEvent(self.device, "free", self.trace_num))
      self.allocator.free(self._buf, self.nbytes, self.options)
    elif self._base is not None: self._base.allocated_views -= 1
    del self._buf
  def __reduce__(self):
    buf = None
    if self._base is not None:
      return self.__class__, (self.device, self.size, self.dtype, None, None, None, 0, self.base, self.offset, self.is_allocated())
    if self.device == "NPY": return self.__class__, (self.device, self.size, self.dtype, self._buf, self.options, None, self.uop_refcount)
    if self.is_allocated():
      buf = bytearray(self.nbytes)
      self.copyout(memoryview(buf))
    return self.__class__, (self.device, self.size, self.dtype, None, self.options, buf, self.uop_refcount)
  @property
  def trace_num(self) -> int:
    if not hasattr(self, '_trace_num'): self._trace_num = len(Buffer.profile_events)
    return self._trace_num
  @property
  def nbytes(self): return self.size*self.dtype.itemsize
  @suppress_finalizing
  def __del__(self): (not hasattr(self, '_buf')) or self.deallocate()
  def __repr__(self):
    return f"<buf real:{self.is_allocated()} device:{self.device} size:{self.size} dtype:{self.dtype}" + \
           (f" offset:{self.offset}" if self._base is not None else "") + (f" {self.options=}" if self.options is not None else "") + ">"
  def as_dmaref(self) -> DMARef:
    assert hasattr(self.allocator, "_as_dmaref"), f"Device {self.device} doesn't support DMA"
    return self.allocator._as_dmaref(self._buf)
  def as_buffer(self, allow_zero_copy=False, force_zero_copy=False) -> memoryview:
    # zero copy with as_buffer (disabled by default due to use after free)
    if (force_zero_copy or allow_zero_copy) and hasattr(self.allocator, '_as_buffer') and (self.options is None or self.options.image is None):
      return self.allocator._as_buffer(self._buf)
    assert not force_zero_copy, "force zero copy was passed, but copy is required"
    return self.copyout(memoryview(bytearray(self.nbytes)))
  def as_typed_buffer(self, shape=None, allow_zero_copy=False, force_zero_copy=False) -> memoryview:
    assert self.dtype.base.fmt is not None, f"no fmt dtype for {self.dtype.base}"
    assert self.dtype.base.fmt != "e" or sys.version_info >= (3, 12)
    return self.as_buffer(allow_zero_copy, force_zero_copy).cast(self.dtype.base.fmt, shape if shape is not None else (self.size,))
  def numpy(self) -> 'np.ndarray': # type: ignore [name-defined] # noqa: F821
    import numpy as np
    assert _to_np_dtype(self.dtype.base) is not None, f"no np dtype for {self.dtype.base}"
    return np.frombuffer(self.as_buffer(), dtype=_to_np_dtype(self.dtype.base))
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
  def view(self, size:int, dtype:DType, offset:int) -> Buffer:
    assert offset < self.nbytes, "offset must be less than nbytes"
    if self._base is not None: return Buffer(self.device, size, dtype, base=self._base, offset=self.offset+offset)
    return Buffer(self.device, size, dtype, base=self, offset=offset)