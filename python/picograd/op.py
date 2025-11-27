from __future__ import annotations
from typing import Callable, Sequence
import math, ctypes
from dataclasses import dataclass
from enum import auto, IntEnum, Enum
from picograd.device import Allocator
from picograd.dtype import DType, dtypes
from picograd.helpers import DEBUG, MAX_BUFFER_SIZE

sint = int # |Op MOOSE

class FastEnum(IntEnum): # wrapper around IntEnum that preserves Enum.__str__ and makes auto() unique across all FastEnum subclasses
  def __str__(self): return Enum.__str__(
        self)
  @staticmethod
  def _generate_next_value_(_, __, ___, last_values): return 1 + max([0, *last_values, *[max(c) for c in FastEnum.__subclasses__()]])

# the order of these OpCode controls the order of the toposort
class OpCode(FastEnum):
  # ops that aren't rendered
  # NOOP = auto(); SINK = auto(); UNIQUE = auto(); DEVICE = auto(); KERNEL = auto(); PRECAST = auto(); REWRITE_ERROR = auto() 
  # SENTINEL = auto()
  # AFTER = auto() # AFTER passes src[0] through and promises in the toposort that any consumers of the AFTER run after src[1:]
  # GROUP = auto() # GROUP is a NOOP that just merges things together

  # buffer ops
  # COPY = auto(); BUFFER = auto(); BUFFER_VIEW = auto(); MSELECT = auto(); MSTACK = auto()
  # BUFFERIZE = auto() # create buffer
  # CONTIGUOUS = auto(); CONTIGUOUS_BACKWARD = auto(); DETACH = auto() # ops that adjust the behavior of the scheduler

  # movement ops! these only exist in the tensor graph
  # RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); FLIP = auto()
  # MULTI = auto()  # MULTI is really a movement op

  # DEFINE_GLOBAL = auto(); DEFINE_LOCAL = auto(); DEFINE_REG = auto() # TODO: unify these ops into the levels of the memory hierarchy. depends on ASSIGN is STORE
  # DEFINE_VAR = auto(); BIND = auto() # this is for symbolic shapes
  # SPECIAL = auto()   # this is a RANGE for GPU dimensions, similar to symbolic shapes but not exactly

  # REDUCE_AXIS = auto(); REDUCE = auto(); ALLREDUCE = auto() # reduce
  # UNROLL = auto(); CONTRACT = auto(); GEP = auto(); VECTORIZE = auto(); CAT = auto(); PTRCAT = auto() # optimization helper ops
  CAST = auto(); BITCAST = auto(); EXP2 = auto(); LOG2 = auto(); SIN = auto(); SQRT = auto(); RECIPROCAL = auto(); NEG = auto(); TRUNC = auto() # unaryops
  # LOAD = auto(); STORE = auto() # load/store before math
  # ASSIGN = auto()  # TODO: ASSIGN is STORE, remove ASSIGN
  # WMMA = auto()   # tensor core math op, not elementwise
  # INDEX = auto() # INDEX is a BinaryOp similar to ADD, but it operates on pointers

  # binaryops
  MM = auto(); FA = auto() # TODO: order??
  ADD = auto(); MUL = auto(); SHL = auto(); SHR = auto(); IDIV = auto(); MAX = auto(); MOD = auto()
  CMPLT = auto(); CMPNE = auto(); CMPEQ = auto()
  XOR = auto(); OR = auto(); AND = auto()
  THREEFRY = auto(); SUB = auto(); FDIV = auto(); POW = auto()

  # WHERE = auto(); MULACC = auto() # ternaryops
  # BARRIER = auto(); RANGE = auto(); IF = auto(); END = auto(); ENDIF = auto() # control flow ops
  # VCONST = auto(); CONST = auto() # consts. VCONST is a vectorized const
  # CUSTOM = auto(); CUSTOMI = auto() # CUSTOM/CUSTOMI are used to output strings into codegen. the I makes the string inline

@dataclass(eq=False, slots=True)
class Op: # (ComputeMixin): # MovementMixin, metaclass=UOpMetaClass
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

class PatternMatcher:
  def __init__(): raise NotImplementedError

class Pattern:
  def __init__(): raise NotImplementedError

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
               uop_refcount=0, base:Buffer|None=None, offset:int=0, preallocate=False):
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