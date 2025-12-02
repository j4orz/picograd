from __future__ import annotations
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass
import itertools

from picograd.helpers import DEBUG, MAX_BUFFER_SIZE
from picograd.engine.irparser import OpCode, GraphBuilder
if TYPE_CHECKING:
  from picograd.runtime.device import Buffer, Device
from picograd.dtype import ConstLike, DType, dtypes

# picograd to tinygrad bridge
# - removed buf_op and as_buf used by haldie/tvm schedule/rangify to map high level ops back to buffers
# - removed buf_target
# - rename OpMixin.alu() -> OpMixin.eval()
# - retrofit an eager interpreter in OpMixin.eval()

# **************** Expression Graph ****************
@dataclass(eq=False, slots=True) # NOTE: this should be frozen, but frozen is slower
class OpNode(GraphBuilder):
  """
  GraphOp structs (which Tensor's deusugar into) are vertices which form an expression graph G=(V,E) where V is a Set<Op> and E is a Set<(Op,Op)>
  the name of the struct "Op" is somewhat of a misnomer because the structs store
  *state* for both the a. specified compute (OpCode) and b. the allocated memory (Buffer)
  so it's more accurate to conceptualize the struct of Op as both the function type f: _ -> _ and the evaluation of said function f(.)
  produced by the *functionality* of the dynamically "eager" interpreter and the just-in-time lazy "graph" compiler.

  the derivative f'(x) is the sum of path products on the expression graph, where factors in the product are local derivatives.
  selecting the optimal order to evaluate such path products (given that the operations represented by each vertex is *associative*) is NP-hard.
  since the functions f(x) that need to be differentiated in the field of machine learning are loss functions of the form f: R^n -> R which fan-out,
  the reverse direction is heuristically used with a reverse topological sort given that the time complexity is proportional to the number of outputs m which in this case is 1
  for many deeplearning workloads that are a series of matrix-matrix multiplications with a final matrix-vector multiplication, multiplying in the reverse direction results in [(v,e)->(v,e)^2]
  """
  ftype: OpCode
  inputs: tuple[OpNode, ...]
  dtype: DType
  # storage: Buffer
  payload: Any

  @property
  def device(self) -> str|tuple[str, ...]: return unwrap(self._device)
  @property
  def size(self) -> int: return prod([int(x.vmax) if isinstance(x, OpNode) else x for x in self.shape])

  unique_num = itertools.count(0)
  @staticmethod
  def unique(paylaod:int|None=None):
    return OpNode(OpCode.UNIQUE, tuple(), dtypes.void, next(OpNode.unique_num) if paylaod is None else paylaod)

  @staticmethod
  def new_buffer(device:str|tuple[str, ...], size:int, dtype:DType, num=None):
    # for now, picograd's retrofitted eager semantics mean all opnodes have their buffers eagerly materialized
    return OpNode(OpCode.BUFFER, (OpNode.unique(num), OpNode(OpCode.DEVICE, tuple(), dtypes.void, payload=device)), dtype, size)

  def const_like(self, b:ConstLike):
    return OpNode.const(self.dtype, b, device=self._device, shape=self._shape) # constants can optionally have a DEVICE source
  
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
  
  # **************** Shape ****************
  @recursive_property
  def _shape(self) -> tuple[sint, ...]|None:
    match self.op:
      # late ops don't have shape
      case OpCode.UNIQUE | OpCode.DEVICE | OpCode.RANGE | OpCode.LOAD | OpCode.IF | OpCode.BARRIER | OpCode.CUSTOM | OpCode.CUSTOMI | \
           OpCode.VECTORIZE | OpCode.VCONST | OpCode.GEP | OpCode.SPECIAL | OpCode.UNROLL | OpCode.CONTRACT:
        return None

      case OpCode.INDEX:
        # non pointer index doesn't have a shape
        if not isinstance(self.dtype, PtrDType): return None
        # fully indexed doesn't have a shape. TODO: remove this
        if self.src[0]._shape is None or len(self.src[1:]) == len(self.src[0].shape): return None
        # pointer index
        return self.src[0].shape[len(self.src[1:]):]

      # some ops init the shape
      case OpCode.CONST | OpCode.DEFINE_VAR | OpCode.BIND: return () if self._device is not None else None
      case OpCode.BUFFER: return (self.arg,)
      case OpCode.BUFFER_VIEW: return (self.arg[0],)
      case OpCode.BUFFERIZE: return tuple([int(r.vmax+1) for r in self.src[1:]])
      case OpCode.DEFINE_GLOBAL | OpCode.DEFINE_LOCAL | OpCode.DEFINE_REG: return (self.ptrdtype.size,)

      # passthrough ops
      case OpCode.REDUCE | OpCode.MSTACK | OpCode.MSELECT | OpCode.DETACH | OpCode.CONTIGUOUS | OpCode.CONTIGUOUS_BACKWARD | OpCode.AFTER | OpCode.END:
        return self.src[0]._shape

      # ops with custom handling
      case OpCode.KERNEL: return self.arg.ast._shape

      # TODO: disallow shape changing bitcast
      case OpCode.BITCAST:
        ps = self.src[0]._shape
        if ps is None: return None
        if (output_sz:=self.dtype.itemsize) != (input_sz:=self.src[0].dtype.itemsize): return ps[:-1]+(ssimplify((ps[-1]*input_sz) // output_sz),)
        return ps

      # TODO: disallow reshape from nothing. tested by TestOpenClip.test_multigpu_clip_score
      case OpCode.RESHAPE:
        if self.src[0]._shape is None: return self.marg

    # movement ops change the shape. this is the logic from the old ShapeTracker
    # NOTE: ssimplify is required because the shape needs to be canonical for broadcasting and same shape checking
    if self.op in GroupOp.Movement.union({OpCode.MULTI, OpCode.REDUCE_AXIS, OpCode.WMMA}):
      ps = self.src[0]._shape
      # TODO: WMMA is used for both axis WMMA and op WMMA. fix this and remove this hack. tested by BERT on AMD LLVM
      if ps is None and self.op is OpCode.WMMA: return None
      if ps is None: raise RuntimeError(f"movement op {self.op} requires shape")
      match self.op:
        case OpCode.RESHAPE:
          if not all(x >= 0 for x in self.marg): raise ValueError(f"shape can't contain negative numbers {self.marg}")
          if prod(ps) != prod(self.marg): raise ValueError(f"bad reshape: {ps} -> {self.marg}")
          return self.marg
        case OpCode.EXPAND:
          if len(ps) != len(self.marg) or not all(s==ns or (s==1 and ns>=0) for s,ns in zip(ps, self.marg)):
            raise ValueError(f"bad expand: {ps} -> {self.marg}")
          return self.marg
        case OpCode.PERMUTE:
          if sorted(self.marg) != list(range(len(ps))): raise ValueError(f"invalid permutation {self.marg} of len {len(ps)}")
          return tuple(ps[i] for i in self.marg)
        case OpCode.PAD:
          # TODO: why do i need resolve here?
          if len(ps) != len(self.marg) or not all(resolve(b>=0) and resolve(e>=0) for b,e in self.marg): raise ValueError(f"invalid pad {self.marg}")
          return tuple(ssimplify(s+b+e) for s,(b,e) in zip(ps, self.marg))
        case OpCode.SHRINK:
          # TODO: why do i need resolve here?
          if len(ps) != len(self.marg) or not all(resolve(0<=b) and resolve(b<=e) and resolve(e<=s) for s,(b,e) in zip(ps, self.marg)):
            raise ValueError(f"invalid shrink {self.marg} for {ps}")
          return tuple(ssimplify(e-s) for s,e in self.marg)
        case OpCode.FLIP:
          if len(ps) != len(self.marg) or not all(isinstance(x, bool) for x in self.marg): raise ValueError(f"bad flip on {ps}, {self.marg}")
          return ps
        case OpCode.MULTI: return tuple(s*len(self.device) if a == self.axis else s for a,s in enumerate(ps))
        case OpCode.REDUCE_AXIS | OpCode.WMMA:
          axis_arg = self.arg[1] if self.op is OpCode.REDUCE_AXIS else self.arg[7]
          if not isinstance(axis_arg, tuple) or not all(isinstance(x, int) and x>=0 and x<len(ps) for x in axis_arg):
            raise ValueError(f"invalid type for axis: {axis_arg}")
          return tuple(1 if i in axis_arg else s for i,s in enumerate(ps))

    # elementwise ops keep the shape the same. all inputs with shape must match
    if self.op in GroupOp.ALU.union({OpCode.CAST, OpCode.COPY, OpCode.ASSIGN, OpCode.NOOP, OpCode.GROUP, OpCode.SINK, OpCode.ALLREDUCE, OpCode.STORE}):
      # TODO: remove this hack for 3 op assign
      input_shapes = [x._shape for x in (self.src[:2] if self.op is OpCode.ASSIGN else self.src) if x._shape is not None]
      if len(input_shapes) == 0: return None
      if not all_same(input_shapes): raise RuntimeError(f"shape mismatch at {self.op}: {input_shapes}")
      return input_shapes[0]

    # all Ops must be explicitly handled
    raise NotImplementedError(f"no shape handling for {self.op} with {self.dtype}")

  @property
  def shape(self) -> tuple[sint, ...]:
    if (ret:=self._shape) is None: raise RuntimeError(f"shape requested, but {self.op} doesn't have a shape")
    return ret

  @property
  def size(self) -> int: return prod([int(x.vmax) if isinstance(x, OpNode) else x for x in self.shape])

  @functools.cached_property
  def ended_ranges(self):
    if self.op in range_start: return self.src[range_start[self.op]:]
    return ()

  # determine what ranges this is in
  @recursive_property
  def _ranges(self) -> dict[OpNode, None]:
    ret: dict[OpNode, None] = {}
    for s in self.src: ret.update(s.ranges)
    for er in self.ended_ranges:
      if er.op is Ops.RANGE:
        # if it's a single RANGE, we don't flow through it.
        if er in ret: del ret[er]
      else:
        # if it's not a RANGE, we include all ranges in srcs.
        # technically we shouldn't flow through these ranges either, but this is pre pm_add_control_flow so it's the same.
        for s in er.ranges:
          if s in ret: del ret[s]
    return ret
  
  # **************** Storage ****************
  @staticmethod
  def new_buffer(device:str|tuple[str, ...], size:int, dtype:DType, num=None):
    return OpNode(OpCode.BUFFER, dtype, (OpNode.unique(num), OpNode(OpCode.DEVICE, arg=device)), size)
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
  def buf_uop(self) -> OpNode:
    if self.op is OpCode.BUFFER: return self
    if self.op is OpCode.MSELECT: return self.src[0].buf_uop.mselect(self.arg)
    if self.op is OpCode.MSTACK: return OpNode(OpCode.MSTACK, self.dtype, src=tuple(x.buf_uop for x in self.src))
    assert self.base.op is OpCode.AFTER, f"must be AFTER {self.base.op}"
    return self.base.src[0].buf_uop.base

  def as_buf(self) -> OpNode:
    if self.op is OpCode.MSELECT: return self.src[0].as_buf().mselect(self.arg)
    if self.op is OpCode.MSTACK: return OpNode(OpCode.MSTACK, self.dtype, src=tuple(x.as_buf() for x in self.src))
    # TODO: this should be the only one of these. this is the one RANGEIFY uses
    s = self
    while len(s.src) and s.op not in {OpCode.BUFFER, OpCode.BUFFERIZE, OpCode.MSTACK}: s = s.src[0]
    return s

  def buf_target(self) -> OpNode:
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
  
  # **************** Compute ****************
  def _apply_compute_opcode(self, ftype: OpCode, *inputs:OpNode) -> Self: # required method by OpMixin
    """
    the evaluator overrides* the semantics of the host language with a nonstandard interpretation (device acceleration of f(x), automatic differentiation of f'(x))
    called by OpMixin.eval() which acts as the embedded DSL's "parser", by coupling python dunder builtins to be aware of the corresponding OpCode intermediate representation
    *: note that .eval is not a static method, that self is the Op that the OpCode ftype is operating on, to produce a new Self
    the interpreter's evaluator lives *on* the OpNode, rather than a freestanding .eval() returning an OpNode,
    similar to how the Allocator lives *on* the Buffer, rather than an freestanding pure Allocator.allocate() returning a Buffer,
    """
    out_dtype = (self, *inputs)[-1].dtype
    # if op in {OpCode.CMPLT, OpCode.CMPNE, OpCode.CMPEQ}: out_dtype = dtypes.bool.vec(out_dtype.count) if out_dtype.count > 1 else dtypes.bool
    # return Op((self,)+inputs, ftype, out_dtype,)
    match ftype:
      case OpCode.NEG: launch_neg(*inputs)
      case OpCode.ADD:
        # 1. memory: allocate and memcpy on device
        device = HIPDevice()
        a, b, c = [device.allocator.alloc(4), device.allocator.alloc(4), device.allocator.alloc(4)]
        device.allocator._copyin(a, memoryview(bytearray([2,0,0,0])))
        device.allocator._copyin(b, memoryview(bytearray([3,0,0,0])))
        # 2. compute: compile a kernel to a binary
        kernel = HIPCCCompiler().compile("__global__ void add(int *a, int *b, int *c) { int id = blockDim.x * blockIdx.x + threadIdx.x; if(id < N) c[id] = a[id] + b[id]; }")
        # 3. launch
        f = device.kernel("add", kernel)
        f(a, b, c) # HIPKernel

        print(val := device.allocator._as_buffer(c).cast("I").tolist()[0])
        assert val == 5 # check the data out
      case OpCode.MUL: raise NotImplementedError("todo")
      case OpCode.MM: raise NotImplementedError("todo")
      case OpCode.RECIPROCAL: raise NotImplementedError("todo")
      case OpCode.EXP2: raise NotImplementedError("todo")
      case OpCode.LOG2: raise NotImplementedError("todo")
      case OpCode.SIN: raise NotImplementedError("todo")
      case _: raise NotImplementedError(f"unsupported opcode {ftype!r}")

  def _apply_movement_opcode(self, op:Ops, arg, same_shape_noop:bool=False) -> OpNode:
    match op:
      case OpCode.RESHAPE | OpCode.EXPAND: src_args = [arg]
      case OpCode.PAD | OpCode.SHRINK: src_args = list(zip(*arg))
      case OpCode.PERMUTE | OpCode.FLIP: src_args = []
      case _: raise RuntimeError(f"{op} is not a MovementOp")
    usrcs = []
    for arg in src_args:
      if len(arg) == 0: usrcs.append(OpNode(OpCode.VECTORIZE, dtypes.index.vec(0)))
      elif all(isinstance(x, int) for x in arg): usrcs.append(OpNode.const(dtypes.index.vec(len(arg)), arg))
      else: usrcs.append(OpNode(OpCode.VECTORIZE, dtypes.index.vec(len(arg)), tuple(OpNode.const(dtypes.index, x) if isinstance(x, int) else x for x in arg)))
    if len(usrcs) == 0: ret = OpNode(op, self.dtype, (self,), arg)
    else: ret = OpNode(op, self.dtype, (self,)+OpNode.sink(*usrcs).simplify().src)
    # for all movement ops, we check shape property
    if ret.shape == self.shape and same_shape_noop: return self
    return ret
