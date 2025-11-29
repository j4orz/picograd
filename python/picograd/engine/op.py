from __future__ import annotations
from typing import Optional, Self
import ctypes
from dataclasses import dataclass
from enum import auto, IntEnum, Enum

from picograd.helpers import DEBUG, MAX_BUFFER_SIZE
from picograd.engine.opcode import OpCode, OpMixin
from picograd.runtime.device import Allocator
from picograd.dtype import ConstLike, DType, dtypes

# picograd to tinygrad bridge
# - removed buf_op and as_buf used by haldie/tvm schedule/rangify to map high level ops back to buffers
# - removed buf_target
# - rename OpMixin.alu() -> OpMixin.eval()
# - retrofit an eager interpreter in OpMixin.eval()

# **************** Expression Graph ****************
@dataclass(eq=False, slots=True) # NOTE: this should be frozen, but frozen is slower
class Op(OpMixin):
  """
  Op structs (which Tensor's deusugar into) are vertices which form an expression graph G=(V,E) where V is a Set<Op> and E is a Set<(Op,Op)>
  the name of the struct "Op" is somewhat of a misnomer because the structs store
  *state* for both the a. specified compute (OpCode) and b. the allocated memory (Buffer)
  so it's more accurate to conceptualize the struct of Op as both the function type f: _ -> _ and the evaluation of said function f(.)
  produced by the *functionality* of the dynamically "eager" interpreter and the just-in-time lazy "graph" compiler.

  the derivative f'(x) is the sum of path products on the expression graph, where factors in the product are local derivatives.
  selecting the optimal order to evaluate such path products is NP-hard.
  since the functions f(x) that need to be differentiated in the field of machine learning are loss functions of the form f: R^n -> R which fan-out,
  the reverse direction is heuristically used with a reverse topological sort.
  """
  inputs: tuple[Op, ...] = tuple()
  ftype: OpCode
  dtype: DType = dtypes.void
  storage: Buffer # make this Optional when adding the compiler pipeline

  def const_like(self, b:ConstLike):
    return Op.const(self.dtype, b, device=self._device, shape=self._shape) # constants can optionally have a DEVICE source
  
  def eval(self, ftype: OpCode, *inputs:Op) -> Self: # required method by OpMixin
    """
    the eager evaluator is an embedded interpreter which override the semantics of the host language
    """
    out_dtype = (self, *inputs)[-1].dtype
    # if op in {OpCode.CMPLT, OpCode.CMPNE, OpCode.CMPEQ}: out_dtype = dtypes.bool.vec(out_dtype.count) if out_dtype.count > 1 else dtypes.bool
    # return Op((self,)+inputs, ftype, out_dtype,)
    match ftype:
      case OpCode.NEG: launch_neg(*inputs)
      case OpCode.ADD: launch_add(*inputs)
      case OpCode.MUL: launch_mul(*inputs)
      case OpCode.MM: launch_mm(*inputs)
      case OpCode.RECIP: launch_recip(*inputs)
      case OpCode.EXP2: launch_exp2(*inputs)
      case OpCode.LOG2: launch_log2(*inputs)
      case OpCode.SIN: launch_sin(*inputs)
      case _: raise NotImplementedError(f"unsupported opcode {ftype!r}")
  
  @staticmethod
  def new_buffer(device:str|tuple[str, ...], size:int, dtype:DType, num=None):
    return Op(OpCode.BUFFER, dtype, (Op.unique(num), Op(OpCode.DEVICE, arg=device)), size)
  
  @property
  def device(self) -> str|tuple[str, ...]: return unwrap(self._device)
  def _device(self) -> str|tuple[str, ...]|None: # @recursive_property
    if self.op is OpCode.DEVICE: return self.arg
    if self.op is OpCode.BUFFERIZE: return self.arg.device
    if self.op is OpCode.AFTER: return self.inputs[0]._device
    if self.op is OpCode.MSELECT:
      assert isinstance(self.inputs[0].device, tuple), "mselect must be on tuple device"
      return self.inputs[0].device[self.arg]
    if self.op is OpCode.MSTACK: return tuple(cast(str, x.device) for x in self.inputs)
    if self.op in {OpCode.COPY, OpCode.BUFFER, OpCode.ALLREDUCE}: return self.inputs[1].device
    for x in self.inputs:
      if x._device is not None: return x._device
    return None

  @property
  def storage(self) -> Buffer|MultiBuffer:
    from tinygrad.device import Buffer, MultiBuffer
    if self is not self.base:
      assert self.op is OpCode.RESHAPE, f"can only be RESHAPE {self}"
      return self.inputs[0].storage
    if self.op is OpCode.MSELECT:
      ret = self.inputs[0].storage
      assert isinstance(ret, MultiBuffer)
      return ret.bufs[self.arg]
    if self.op is OpCode.MSTACK:
      ret = MultiBuffer.__new__(MultiBuffer)
      ret.bufs = [cast(Buffer, x.storage) for x in self.inputs]
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
  def size(self) -> int: return prod([int(x.vmax) if isinstance(x, Op) else x for x in self.shape])
  
# **************** Python/C Foreign Function Helpers  ****************
def from_mv(mv:memoryview, to_type:type[ctypes._SimpleCData]=ctypes.c_char) -> ctypes.Array:
  return ctypes.cast(ctypes.addressof(to_type.from_buffer(mv)), ctypes.POINTER(to_type * len(mv))).contents
def to_mv(ptr:int, sz:int) -> memoryview: return memoryview((ctypes.c_uint8 * sz).from_address(ptr)).cast("B")
def mv_address(mv): return ctypes.addressof(ctypes.c_char.from_buffer(mv))
def flat_mv(mv:memoryview): return mv if len(mv) == 0 else mv.cast("B", shape=(mv.nbytes,))

# **************** Memory  ****************
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