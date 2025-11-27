from __future__ import annotations
from typing import Callable
import ctypes
from dataclasses import dataclass
from enum import auto, IntEnum, Enum
from picograd.device import Allocator
from picograd.dtype import DType, dtypes
from picograd.helpers import DEBUG, MAX_BUFFER_SIZE

sint = int # |Op MOOSE

class PatternMatcher:
  def __init__(): raise NotImplementedError

class Pattern:
  def __init__(): raise NotImplementedError

class FastEnum(IntEnum): # wrapper around IntEnum that preserves Enum.__str__ and makes auto() unique across all FastEnum subclasses
  def __str__(self): return Enum.__str__(
        self)
  @staticmethod
  def _generate_next_value_(_, __, ___, last_values): return 1 + max([0, *last_values, *[max(c) for c in FastEnum.__subclasses__()]])

# the order of these OpCode controls the order of the toposort
class OpCode(FastEnum):
  # ops that aren't rendered: noop, sink, unique, device, kernel, precast, rewrite_error, sentinel, after, group
  # buffer ops: copy, buffer, buffer_view, mselect, mstack, bufferize, contiguous, contiguous_backward
  # movement ops: these only exist in the tensor graph reshape, permute, expand, pad, shrink, flip, multi
  # def_global, def_local, def_reg, def_var, bind, special(range)
  # reduce_axis, reduce, allreduce
  # unroll, contract, gep, vectorize, cat, ptrcat
  CAST = auto(); BITCAST = auto(); EXP2 = auto(); LOG2 = auto(); SIN = auto(); SQRT = auto(); RECIPROCAL = auto(); NEG = auto(); TRUNC = auto() # unaryops
  # laod, store, assign, wmma, index

  # binaryops
  MM = auto(); FA = auto() # TODO: order??
  ADD = auto(); MUL = auto(); SHL = auto(); SHR = auto(); IDIV = auto(); MAX = auto(); MOD = auto()
  CMPLT = auto(); CMPNE = auto(); CMPEQ = auto()
  XOR = auto(); OR = auto(); AND = auto()
  THREEFRY = auto(); SUB = auto(); FDIV = auto(); POW = auto()

  # where, mulacc
  # barrier, range, if, end, endif
  # vconst, const, custom, customi

@dataclass(eq=False, slots=True)
class Op: # (ComputeMixin): # MovementMixin, metaclass=OpMetaClass
  """
  ...
  """
  code: OpCode; src:tuple[Op, ...] = tuple(); dtype:DType = dtypes.void
  # arg:Any = None; tag:Any = None
  @property
  def device(self) -> str|tuple[str, ...]: raise NotImplementedError("todo")
  @property
  def shape(self) -> tuple[sint, ...]: raise NotImplementedError("todo")

  def toposort(self, gate:Callable|None=None) -> dict[Op, None]:
    visited: dict[Op, None] = {}
    stack: list[tuple[Op, bool]] = [(self, False)] # each stack entry is (node, visited_flag)

    while stack:
      node, visited = stack.pop()
      if node in visited: continue
      if not visited:
        if gate is None or gate(node): # MOOSE gate?
          stack.append((node, True))  # push node back on stack to process after its srcs
          for s in reversed(node.src): stack.append((s, False)) # push srcs on the stack
      else: visited[node] = None # second time i'm seeing this node, add it to returned toposort
    return visited
  
  @staticmethod
  def new_buffer(device:str|tuple[str, ...], size:int, dtype:DType, num=None):
    return Op(OpCode.BUFFER, dtype, (Op.unique(num), Op(OpCode.DEVICE, arg=device)), size)
  
  @property
  def device(self) -> str|tuple[str, ...]: return unwrap(self._device)
  @recursive_property
  def _device(self) -> str|tuple[str, ...]|None:
    if self.op is OpCode.DEVICE: return self.arg
    if self.op is OpCode.BUFFERIZE: return self.arg.device
    if self.op is OpCode.AFTER: return self.src[0]._device
    if self.op is OpCode.MSELECT:
      assert isinstance(self.src[0].device, tuple), "mselect must be on tuple device"
      return self.src[0].device[self.arg]
    if self.op is OpCode.MSTACK: return tuple(cast(str, x.device) for x in self.src)
    if self.op in {OpCode.COPY, OpCode.BUFFER, OpCode.ALLREDUCE}: return self.src[1].device
    for x in self.src:
      if x._device is not None: return x._device
    return None
  @property
  def buf_Op(self) -> Op:
    if self.op is OpCode.BUFFER: return self
    if self.op is OpCode.MSELECT: return self.src[0].buf_Op.mselect(self.arg)
    if self.op is OpCode.MSTACK: return Op(OpCode.MSTACK, self.dtype, src=tuple(x.buf_Op for x in self.src))
    assert self.base.op is OpCode.AFTER, f"must be AFTER {self.base.op}"
    return self.base.src[0].buf_Op.base

  def as_buf(self) -> Op:
    if self.op is OpCode.MSELECT: return self.src[0].as_buf().mselect(self.arg)
    if self.op is OpCode.MSTACK: return Op(OpCode.MSTACK, self.dtype, src=tuple(x.as_buf() for x in self.src))
    # TODO: this should be the only one of these. this is the one RANGEIFY uses
    s = self
    while len(s.src) and s.op not in {OpCode.BUFFER, OpCode.BUFFERIZE, OpCode.MSTACK}: s = s.src[0]
    return s

  def buf_target(self) -> Op:
    # the buffer that's being loaded from or store to
    match self.op:
      case OpCode.DEFINE_GLOBAL | OpCode.DEFINE_LOCAL | OpCode.DEFINE_REG: return self
      case OpCode.AFTER | OpCode.INDEX | OpCode.STORE | OpCode.LOAD: return self.src[0].buf_target()
      case OpCode.VECTORIZE:
        assert all_same(self.src)
        return self.src[0].buf_target()
      case _: raise RuntimeError(f"buf_target called on non load/index/store {self.op}")

  @property
  def buffer(self) -> Buffer|MultiBuffer:
    from tinygrad.device import Buffer, MultiBuffer
    if self is not self.base:
      assert self.op is OpCode.RESHAPE, f"can only be RESHAPE {self}"
      return self.src[0].buffer
    if self.op is OpCode.MSELECT:
      ret = self.src[0].buffer
      assert isinstance(ret, MultiBuffer)
      return ret.bufs[self.arg]
    if self.op is OpCode.MSTACK:
      ret = MultiBuffer.__new__(MultiBuffer)
      ret.bufs = [cast(Buffer, x.buffer) for x in self.src]
      assert all_same([x.size for x in ret.bufs]) and all_same([x.dtype for x in ret.bufs]), "multibuffers mismatch buffers"
      return ret
    assert self.op is OpCode.BUFFER, f"must be BUFFER {self.op}"
    if (cret:=buffers.get(self)) is not None: return cret
    rdtype = self.dtype if isinstance(self.dtype, ImageDType) else self.dtype.base
    if isinstance(self.device, tuple): ret = MultiBuffer(self.device, self.size, rdtype).ref(1)
    else: ret = Buffer(self.device, self.size, rdtype).ref(1)
    buffers[self] = ret
    return ret
  @property
  def realized(self) -> Buffer|MultiBuffer|None:
    # NOTE: this is used by the JIT to determine which inputs we capture
    return self.buffer if self.op in {OpCode.BUFFER, OpCode.MSTACK} and self.buffer.is_allocated() else None
  @property
  def is_realized(self) -> bool:
    return all(x.base.realized is not None for x in self.base.src) if self.base.op is OpCode.MULTI else self.base.realized is not None

# **************** Python/C Foreign Function Helpers  ****************
def from_mv(mv:memoryview, to_type:type[ctypes._SimpleCData]=ctypes.c_char) -> ctypes.Array:
  return ctypes.cast(ctypes.addressof(to_type.from_buffer(mv)), ctypes.POINTER(to_type * len(mv))).contents
def to_mv(ptr:int, sz:int) -> memoryview: return memoryview((ctypes.c_uint8 * sz).from_address(ptr)).cast("B")
def mv_address(mv): return ctypes.addressof(ctypes.c_char.from_buffer(mv))
def flat_mv(mv:memoryview): return mv if len(mv) == 0 else mv.cast("B", shape=(mv.nbytes,))

class Buffer:
  """
  Buffer provides an on-device handle of a Tensor's backing storage with a Runtime's Allocator
  """
  def __init__(self,
               device:str, size:int, dtype:DType,
               opaque:Any=None, options:BufferSpec|None=None, initial_value:bytes|None=None,
               Op_refcount=0, base:Buffer|None=None, offset:int=0, preallocate=False):
    assert isinstance(dtype, DType) and not isinstance(dtype, PtrDType)
    self.device, self.size, self.dtype, self.options, self.offset, self.allocated_views = device, size, dtype, options, offset, 0

    if base is None:
      assert offset == 0, "base buffers can't have offset"
      self._base = None
      self._Op_refcount = Op_refcount

      if opaque is not None: self.allocate(opaque)
      if initial_value is not None:
        self.allocate()
        self.copyin(memoryview(initial_value))
    else:
      assert base._base is None, "base can't have a base"
      assert device == base.device, "base must have the same device"
      self._base = base
    if preallocate: self.allocate()
  
  def is_initialized(self) -> bool: return self.is_allocated() and hasattr(self, '_buf') # check if the underlying buffer is allocated and the current buffer/view is initialized
  def is_allocated(self) -> bool: return self.base.is_allocated() if self._base is not None else hasattr(self, '_buf') # check if the underlying buffer is allocated, possibly from the base object
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
      self._buf: Any = self.allocator._offset(self.base._buf, self.nbytes, self.offset) # <-------------------------- ...
    else:
      self._buf = opaque if opaque is not None else self.allocator.alloc(self.nbytes, self.options)
      #if not self.device.startswith("DISK"): GlobalCounters.mem_used += self.nbytes
      #if PROFILE: Buffer.profile_events.append(ProfilePointEvent(self.device, "alloc", self.trace_num, {"dtype":self.dtype, "sz":self.size}))
    return self
  
  def copyin(self, mv:memoryview):
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    assert self.is_initialized(), "can't copyin to unallocated buffer"
    self.allocator._copyin(self._buf, mv) # <---------------------------------------------- ...
    return self
  def copyout(self, mv:memoryview) -> memoryview:
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    assert self.is_initialized(), "can't copyout unallocated buffer"
    self.allocator._copyout(mv, self._buf) # <------------------------------------ ...
    return mv