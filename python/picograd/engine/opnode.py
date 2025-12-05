from __future__ import annotations
import math
from typing import TYPE_CHECKING, Any, Self
from dataclasses import dataclass
import itertools

from picograd import helpers
from picograd.helpers import DEBUG, MAX_BUFFER_SIZE
from picograd.engine.irparser import GroupedOpCodes, OpCode, GraphBuilder
if TYPE_CHECKING:
  from picograd.runtime.device import Buffer, Device
from picograd.dtype import Const, ConstLike, DType, PtrDType, dtypes

# picograd to tinygrad bridge
# - removed buf_op and as_buf used by haldie/tvm schedule/rangify to map high level ops back to buffers
# - removed buf_target
# - rename OpMixin.alu() -> OpMixin.eval()
# - retrofit an eager interpreter in OpMixin.eval()

# out_dtype = (self, *inputs)[-1].dtype
# # if op in {OpCode.CMPLT, OpCode.CMPNE, OpCode.CMPEQ}: out_dtype = dtypes.bool.vec(out_dtype.count) if out_dtype.count > 1 else dtypes.bool
# # return Op((self,)+inputs, ftype, out_dtype,)
# match opcode:
#   case OpCode.NEG: launch_neg(*inputs)
#   case OpCode.ADD:
#     # 1. memory: allocate and memcpy on device
#     device = HIPDevice()
#     a, b, c = [device.allocator.alloc(4), device.allocator.alloc(4), device.allocator.alloc(4)]
#     device.allocator._copyin(a, memoryview(bytearray([2,0,0,0])))
#     device.allocator._copyin(b, memoryview(bytearray([3,0,0,0])))
#     # 2. compute: compile a kernel to a binary
#     kernel = HIPCCCompiler().compile("__global__ void add(int *a, int *b, int *c) { int id = blockDim.x * blockIdx.x + threadIdx.x; if(id < N) c[id] = a[id] + b[id]; }")
#     # 3. launch
#     f = device.kernel("add", kernel)
#     f(a, b, c) # HIPKernel

#     print(val := device.allocator._as_buffer(c).cast("I").tolist()[0])
#     assert val == 5 # check the data out
#   case _: raise NotImplementedError(f"unsupported opcode {opcode!r}")

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
  opcode: OpCode
  inputs: tuple[OpNode, ...]
  dtype: DType
  # storage: Buffer
  payload: Any=None

  @property
  def device(self) -> str|tuple[str, ...]: return unwrap(self._device)
  def _device(self) -> str|tuple[str, ...]|None: 
    if self.opcode is OpCode.DEVICE: return self.payload
    if self.opcode is OpCode.BUFFERIZE: return self.payload.device
    if self.opcode is OpCode.AFTER: return self.inputs[0]._device
    if self.opcode is OpCode.MSELECT:
      assert isinstance(self.inputs[0].device, tuple), "mselect must be on tuple device"
      return self.inputs[0].device[self.payload]
    if self.opcode is OpCode.MSTACK: return tuple(cast(str, x.device) for x in self.inputs)
    if self.opcode in {OpCode.COPY, OpCode.BUFFER, OpCode.ALLREDUCE}: return self.inputs[1].device
    for x in self.inputs:
      if x._device is not None: return x._device
    return None
  @property
  def size(self) -> int: return helpers.prod([int(x.vmax) if isinstance(x, OpNode) else x for x in self.shape])

  # **************** GraphBuilder Required Methods ****************
  """
  the graphbuilder overrides the semantics of the host language with a nonstandard interpretation (device acceleration of f(x), automatic differentiation of f'(x))
  with ComputeOpCodeBuilder._apply_compute_opcode() and MovementOpCodeBuilder._apply_movement_opcode()
  which act as the embedded DSL's "parser", by coupling python dunder builtins to be aware of the corresponding IR OpCode

  *:  keep in mind that the semantics of these two methods are applying *ir op code*
      that is, to maintain parity in semantics with tinygrad (and a smooth pedagogical progression),
      the returned OpNode's are still un-{materialized/realized/evaluated}, and caller's (namely tensor.py)
      need to invoke .eval() on the OpNode for eager semantics.

  **: if you're coming from a functional mindset, note the pythonic/object-oriented .apply_opcode is not a static method
      that self is the OpNode that the OpCode ftype is operating on, to produce a new Self(OpNode)
      i.e the interpreter's evaluator lives *on* the OpNode, rather than a freestanding .apply_opcode() returning an OpNode,
      i.e similar to how the Allocator lives *on* the Buffer, rather than an freestanding pure Allocator.allocate() returning a Buffer,
  """
  def _apply_compute_opcode(self, opcode: OpCode, *inputs:OpNode) -> Self:
    output_dtype = (self, *inputs)[-1].dtype # use the last input's dtype 
    if opcode in {OpCode.CMPLT, OpCode.CMPNE, OpCode.CMPEQ}: output_dtype = dtypes.bool.vec(output_dtype.count) if output_dtype.count > 1 else dtypes.bool
    return OpNode(opcode, (self,)+inputs, output_dtype,)

  def _apply_movement_opcode(self, opcode: OpCode, payload, same_shape_noop: bool=False) -> Self:
    """
    _apply_movement_opcode is a lot more involved compare to _apply_compute_opcode.
    this is largely because movement opcode's (i.e OpCode.{RESHAPE/EXPAND/PAD/PERMUTE/FLIP/etc...})
    modify the *shape*, which is logical/virtual and needs to be mapped to physical memory.

    with the application of movement opcode's, there's a design decision to be made.
      1. following the numpy/torch model (like c++'s std::iterator/std::container), view operations are non-allocating and share the same underlying storage
         tinygrad followed this design decision with their ShapeTracker/LazyBuffer abstractions, which tracked logical nd-indices to physical 1d-indices with a stack of views
        
          option 1 conflates, confuses, and couples the *algorithm* with it's *layout/organization*
          (see kelley's halide disertation: https://dspace.mit.edu/handle/1721.1/89996),
          and becomes problematic when you want to *vertically split* the shape for _____ optimizations.

      2. the alternative design decision is to *encode* and embed all movement semantics around a Tensor's shape *in* the dsl's IR itself,
          to enable __________ about the shapes with the RANGIFY and POSTOPT abstractions,
          inspired by halide and tvm paper (https://arxiv.org/abs/1802.04799).

        see: https://x.com/__tinygrad__/status/1964037572503752910
    
          so _apply_movement_opcode converts the *payload* (i.e python tuple) for the given movement *opcode* (i.e OpCode.{RESHAPE/EXPAND/PAD/PERMUTE/FLIP/etc...})
          *into* the embedded dsl's IR with OpNode's that have OpCode.{VECTORIZE/VCONST} which are subsequently used as input OpNode's to the originally specified movement OpNode
    """
    output_opnodes_inputs = OpNode._convert_movementopcode_payload_to_opnodeir_input(opcode, payload)
    if len(output_opnodes_inputs) == 0:                         output_opnode = OpNode(opcode, (self,), self.dtype, payload)
    else:                                                       output_opnode = OpNode(opcode, (self,) + helpers.normalize_shape(output_opnodes_inputs), self.dtype) # no .simplify() peephole on inputs
                                                                                                                                            # i.e constant folding 2*3->6
                                                                                                                                            # i.e VECTORIZE -> VCONST

    if output_opnode.shape == self.shape and same_shape_noop:   return self # for all movement ops, we check if the movement op results in an identiy no-op
    return                                                      output_opnode
  
  @staticmethod
  def _convert_movementopcode_payload_to_opnodeir_input(opcode: OpCode, payload):
    if DEBUG >= 1: print("converting movementopcode payload to opnode inputs...")
    match opcode:
      case OpCode.RESHAPE | OpCode.EXPAND:                      decoded_payload = [payload]
      case OpCode.PAD | OpCode.SHRINK:                          decoded_payload = list(zip(*payload))
      case OpCode.PERMUTE | OpCode.FLIP:                        decoded_payload = []
      case _: raise RuntimeError(f"{opcode} is not a MovementOp")

    if DEBUG >= 1: print("decoded movementopcode payload is", decoded_payload)
    output_opnodes_inputs = []
    for payload in decoded_payload:
      if len(payload) == 0:                                     output_opnodes_inputs.append(OpNode(OpCode.VECTORIZE, tuple(), dtypes.index.vec(0)))       # empty payload => empty index vector
      elif all(isinstance(x, int) for x in payload):            output_opnodes_inputs.append(OpNode.const(payload, dtypes.index.vec(len(payload))))        # all int payload => constant index vector
      else:                                                     output_opnodes_inputs.append(OpNode(OpCode.VECTORIZE, tuple(OpNode.const(dtypes.index, x) if isinstance(x, int) else x for x in payload))), dtypes.index.vec(len(payload)), # mized int/OpNode payload => 
                                                                                                    
    if DEBUG >= 1: print("output opnodes inputs are:", output_opnodes_inputs)
    return output_opnodes_inputs

  # **************** Evaluation ****************
  def simplify(self, tracked=False):
    # late import!
    from tinygrad.uop.symbolic import symbolic
    with Context(TRACK_MATCH_STATS=0 if not tracked else TRACK_MATCH_STATS.value):
      return graph_rewrite(self, symbolic, name="simplify")
  def ssimplify(self) -> OpNode|Const: return ret.arg if (ret:=self.simplify()).op is OpCode.CONST else ret
  def sintify(self) -> int: return self.arg if self.op is OpCode.CONST else self

  # **************** Sugar ****************
  def sink(*srcs:OpNode|None, **kwargs):  # pylint: disable=no-self-argument
    return OpNode(OpCode.SINK, dtypes.void, tuple([x for x in srcs if x is not None]), **kwargs)

  def const_like(self, b:ConstLike): return OpNode.const(self.dtype, b, device=self._device, shape=self._shape) # constants can optionally have a DEVICE source

  @staticmethod
  def const(c: ConstLike, dtype: DType,
            device: str | tuple[str, ...] | None = None,
            shape: tuple[int, ...] | None=None,
            inputs=None,
            unique: bool | int=False):
    if isinstance(c, OpNode):                                   return c.unbind()[0] if c.op is OpCode.BIND else c
    if isinstance(c, tuple) and helpers.all_same(c):            c = c[0]     # doesn't have to be a VCONST if they are all the same
    if isinstance(c, float) and math.isnan(c):                  c = math.nan # NOTE: float('nan') != float('nan'), so we canonicalize here

    opcode = OpCode.VCONST if isinstance(c, tuple) else OpCode.CONST
    output_opnode = OpNode(opcode, () if inputs is None else (inputs,), dtype, payload=dtypes.as_const(c, dtype),)
    if device is not None:
      if unique or not isinstance(unique, bool):                output_opnode = output_opnode.replace(src=(OpNode(OpCode.DEVICE, arg=device), OpNode.unique(None if unique is True else unique)))
      else:                                                     output_opnode = output_opnode.replace(src=(OpNode(OpCode.DEVICE, arg=device),))
    elif unique or not isinstance(unique, bool):                raise RuntimeError("unique consts only with DEVICE")

    if shape is not None: output_opnode = output_opnode.reshape((1,)*len(shape)).expand(shape)
    return output_opnode
  
  # **************** Shape ****************
  @property
  def size(self) -> int: return helpers.prod([int(x.vmax) if isinstance(x, OpNode) else x for x in self.shape])
  
  @property
  def shape(self) -> tuple[int, ...]:
    if (output:=self._shape) is None: raise RuntimeError(f"shape requested, but {self.opcode} doesn't have a shape")
    return output
  
  @property
  def _shape(self) -> tuple[int, ...] | None:
    match self.opcode:
      # late ops don't have shape
      case OpCode.UNIQUE | OpCode.DEVICE | OpCode.RANGE | OpCode.LOAD | OpCode.IF | OpCode.BARRIER | OpCode.CUSTOM | OpCode.CUSTOMI | \
           OpCode.VECTORIZE | OpCode.VCONST | OpCode.GEP | OpCode.SPECIAL | OpCode.UNROLL | OpCode.CONTRACT:
        return None

      case OpCode.INDEX:
        if not isinstance(self.dtype, PtrDType):                                        return None # non pointer index doesn't have a shape
        elif self.inputs[0]._shape is None or len(self.inputs[1:]) == len(self.inputs[0].shape): return None # fully indexed doesn't have a shape. TODO: remove this
        else:                                                                           return self.inputs[0].shape[len(self.inputs[1:]):] # pointer index

      # constructor ops (which init the shape)
      case OpCode.CONST | OpCode.DEFINE_VAR | OpCode.BIND:                              return () if self._device is not None else None
      case OpCode.BUFFER:                                                               return (self.payload,)
      case OpCode.BUFFER_VIEW:                                                          return (self.payload[0],)
      case OpCode.BUFFERIZE:                                                            return tuple([int(r.vmax+1) for r in self.inputs[1:]])
      case OpCode.DEFINE_GLOBAL | OpCode.DEFINE_LOCAL | OpCode.DEFINE_REG:              return (self.ptrdtype.size,)
      # passthrough ops
      case OpCode.REDUCE | OpCode.MSTACK | OpCode.MSELECT | OpCode.DETACH | OpCode.CONTIGUOUS | OpCode.CONTIGUOUS_BACKWARD | OpCode.AFTER | OpCode.END: return self.inputs[0]._shape
      # ops with custom handling
      case OpCode.KERNEL: return self.payload.ast._shape
      # TODO: disallow shape changing bitcast
      case OpCode.BITCAST:
        ps = self.inputs[0]._shape
        if ps is None: return None
        if (output_sz:=self.dtype.itemsize) != (input_sz:=self.inputs[0].dtype.itemsize): return ps[:-1]+(ssimplify((ps[-1]*input_sz) // output_sz),)
        return ps
      case OpCode.RESHAPE: # TODO: disallow reshape from nothing. tested by TestOpenClip.test_multigpu_clip_score
        if self.inputs[0]._shape is None: return self.marg

    # elementwise ops keep the shape the same. all inputs with shape must match
    if self.opcode in GroupedOpCodes.Compute.union({OpCode.CAST, OpCode.COPY, OpCode.ASSIGN, OpCode.NOOP, OpCode.GROUP, OpCode.SINK, OpCode.ALLREDUCE, OpCode.STORE}):
      # TODO: remove this hack for 3 op assign
      input_shapes = [x._shape for x in (self.inputs[:2] if self.opcode is OpCode.ASSIGN else self.inputs) if x._shape is not None]
      if len(input_shapes) == 0: return None
      if not all_same(input_shapes): raise RuntimeError(f"shape mismatch at {self.opcode}: {input_shapes}")
      return input_shapes[0]

    # movement ops change the shape. this is the logic from the old ShapeTracker
    # NOTE: ssimplify is required because the shape needs to be canonical for broadcasting and same shape checking
    if self.opcode in GroupedOpCodes.Movement.union({OpCode.MULTI, OpCode.REDUCE_AXIS, OpCode.WMMA}):
      ps = self.inputs[0]._shape
      # TODO: WMMA is used for both axis WMMA and op WMMA. fix this and remove this hack. tested by BERT on AMD LLVM
      if ps is None and self.opcode is OpCode.WMMA: return None
      if ps is None: raise RuntimeError(f"movement op {self.opcode} requires shape")
      match self.opcode:
        case OpCode.RESHAPE:
          if not all(x >= 0 for x in self.marg): raise ValueError(f"shape can't contain negative numbers {self.marg}")
          if helpers.prod(ps) != helpers.prod(self.marg): raise ValueError(f"bad reshape: {ps} -> {self.marg}")
          return self.marg
        case OpCode.EXPAND:
          foo = len(ps) != len(self.marg) or not all(s==ns or (s==1 and ns>=0) for s,ns in zip(ps, self.marg))
          if foo: raise ValueError(f"bad expand: {ps} -> {self.marg}")
          return self.marg
        case OpCode.PERMUTE:
          foo = sorted(self.marg) != list(range(len(ps)))
          if foo: raise ValueError(f"invalid permutation {self.marg} of len {len(ps)}")
          return tuple(ps[i] for i in self.marg)
        case OpCode.PAD:
          # TODO: why do i need resolve here?
          foo = len(ps) != len(self.marg) or not all(resolve(b>=0) and resolve(e>=0) for b,e in self.marg)
          if foo: raise ValueError(f"invalid pad {self.marg}")
          return tuple(ssimplify(s+b+e) for s,(b,e) in zip(ps, self.marg))
        case OpCode.SHRINK:
          # TODO: why do i need resolve here?
          foo = len(ps) != len(self.marg) or not all(resolve(0<=b) and resolve(b<=e) and resolve(e<=s) for s,(b,e) in zip(ps, self.marg))
          if foo: raise ValueError(f"invalid shrink {self.marg} for {ps}")
          return tuple(ssimplify(e-s) for s,e in self.marg)
        case OpCode.FLIP:
          foo = len(ps) != len(self.marg) or not all(isinstance(x, bool) for x in self.marg)
          if foo: raise ValueError(f"bad flip on {ps}, {self.marg}")
          return ps
        case OpCode.MULTI: return tuple(s*len(self.device) if a == self.axis else s for a,s in enumerate(ps))
        case OpCode.REDUCE_AXIS | OpCode.WMMA:
          axis_arg = self.payload[1] if self.opcode is OpCode.REDUCE_AXIS else self.payload[7]
          foo = not isinstance(axis_arg, tuple) or not all(isinstance(x, int) and x>=0 and x<len(ps) for x in axis_arg)
          if foo: raise ValueError(f"invalid type for axis: {axis_arg}")
          return tuple(1 if i in axis_arg else s for i,s in enumerate(ps))

    raise NotImplementedError(f"no shape handling for {self.opcode} with {self.dtype}") # all OpCodes must be explicitly handled

  def ended_ranges(self):
    if self.opcode in range_start: return self.inputs[range_start[self.opcode]:]
    return ()

  # determine what ranges this is in
  def _ranges(self) -> dict[OpNode, None]:
    ret: dict[OpNode, None] = {}
    for s in self.inputs: ret.update(s.ranges)
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
    return OpNode(OpCode.BUFFER, (OpNode.unique(num), OpNode(OpCode.DEVICE, tuple(), dtypes.void, payload=device)), dtype, size)

  unique_num = itertools.count(0)
  @staticmethod
  def unique(paylaod:int|None=None):
    return OpNode(OpCode.UNIQUE, tuple(), dtypes.void, next(OpNode.unique_num) if paylaod is None else paylaod)

  @property
  def device(self) -> str|tuple[str, ...]: return unwrap(self._device)
  def _device(self) -> str|tuple[str, ...]|None:
    if self.opcode is OpCode.DEVICE: return self.payload
    if self.opcode is OpCode.BUFFERIZE: return self.payload.device
    if self.opcode is OpCode.AFTER: return self.inputs[0]._device
    if self.opcode is OpCode.MSELECT:
      assert isinstance(self.inputs[0].device, tuple), "mselect must be on tuple device"
      return self.inputs[0].device[self.payload]
    if self.opcode is OpCode.MSTACK: return tuple(cast(str, x.device) for x in self.inputs)
    if self.opcode in {OpCode.COPY, OpCode.BUFFER, OpCode.ALLREDUCE}: return self.inputs[1].device
    for x in self.inputs:
      if x._device is not None: return x._device
    return None
  @property
  def buf_uop(self) -> OpNode:
    if self.opcode is OpCode.BUFFER: return self
    if self.opcode is OpCode.MSELECT: return self.inputs[0].buf_uop.mselect(self.payload)
    if self.opcode is OpCode.MSTACK: return OpNode(OpCode.MSTACK, self.dtype, src=tuple(x.buf_uop for x in self.inputs))
    assert self.base.op is OpCode.AFTER, f"must be AFTER {self.base.op}"
    return self.base.inputs[0].buf_uop.base

  def as_buf(self) -> OpNode:
    if self.opcode is OpCode.MSELECT: return self.inputs[0].as_buf().mselect(self.payload)
    if self.opcode is OpCode.MSTACK: return OpNode(OpCode.MSTACK, self.dtype, src=tuple(x.as_buf() for x in self.inputs))
    # TODO: this should be the only one of these. this is the one RANGEIFY uses
    s = self
    while len(s.inputs) and s.op not in {OpCode.BUFFER, OpCode.BUFFERIZE, OpCode.MSTACK}: s = s.inputs[0]
    return s

  def buf_target(self) -> OpNode:
    # the buffer that's being loaded from or store to
    match self.opcode:
      case OpCode.DEFINE_GLOBAL | OpCode.DEFINE_LOCAL | OpCode.DEFINE_REG: return self
      case OpCode.AFTER | OpCode.INDEX | OpCode.STORE | OpCode.LOAD: return self.inputs[0].buf_target()
      case OpCode.VECTORIZE:
        assert all_same(self.inputs)
        return self.inputs[0].buf_target()
      case _: raise RuntimeError(f"buf_target called on non load/index/store {self.opcode}")

  @property
  def buffer(self) -> Buffer|MultiBuffer:
    from tinygrad.device import Buffer, MultiBuffer
    if self is not self.base:
      assert self.opcode is OpCode.RESHAPE, f"can only be RESHAPE {self}"
      return self.inputs[0].buffer
    if self.opcode is OpCode.MSELECT:
      ret = self.inputs[0].buffer
      assert isinstance(ret, MultiBuffer)
      return ret.bufs[self.payload]
    if self.opcode is OpCode.MSTACK:
      ret = MultiBuffer.__new__(MultiBuffer)
      ret.bufs = [cast(Buffer, x.buffer) for x in self.inputs]
      assert all_same([x.size for x in ret.bufs]) and all_same([x.dtype for x in ret.bufs]), "multibuffers mismatch buffers"
      return ret
    assert self.opcode is OpCode.BUFFER, f"must be BUFFER {self.opcode}"
    if (cret:=buffers.get(self)) is not None: return cret
    rdtype = self.dtype if isinstance(self.dtype, ImageDType) else self.dtype.base
    if isinstance(self.device, tuple): ret = MultiBuffer(self.device, self.size, rdtype).ref(1)
    else: ret = Buffer(self.device, self.size, rdtype).ref(1)
    buffers[self] = ret
    return ret
  @property
  def realized(self) -> Buffer|MultiBuffer|None:
    # NOTE: this is used by the JIT to determine which inputs we capture
    return self.buffer if self.opcode in {OpCode.BUFFER, OpCode.MSTACK} and self.buffer.is_allocated() else None
  @property
  def is_realized(self) -> bool:
    return all(x.base.realized is not None for x in self.base.inputs) if self.base.op is OpCode.MULTI else self.base.realized is not None