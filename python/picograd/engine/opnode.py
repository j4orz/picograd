from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Self
from dataclasses import dataclass
import math, itertools, weakref

from picograd import helpers
from picograd.helpers import DEBUG, MAX_BUFFER_SIZE
from picograd.engine.irparser import GroupedOpCode, OpCode, TensorDSL
if TYPE_CHECKING:
  from picograd.runtime.device import Buffer, Device
from picograd.dtype import Const, ConstLike, DType, ImageDType, PtrDType, dtypes

# picograd to tinygrad bridge
# - removed buf_op and as_buf used by haldie/tvm schedule/rangify to map high level ops back to buffers
# - removed buf_target
# - rename OpMixin.alu() -> OpMixin.eval()
# - retrofit an eager interpreter in OpMixin.eval()

def pretty_print(opnode:OpNode, cache=None, d=0)->str:
  def dfs(opnode:OpNode, cache:dict):
    for s in opnode.inputs:
      cache.setdefault(s, [len(cache), 0, False])[1] += 1
      if cache[s][1] == 1: dfs(s, cache)
  if cache is None: dfs(opnode, cache:={})
  if (cx:=cache.setdefault(opnode, [0,0,False]))[2]: return f"{' '*d} x{cx[0]}"
  cx[2], inputs = True, (''.join(f'\n{pretty_print(s, cache, d+2)},' for s in opnode.inputs))
  return f"{' '*d}{f'x{cx[0]}:=' * (cx[1]>1)}{type(opnode).__name__}({opnode.opcode}, {opnode.dtype}, payload={opnode.payloadstr()}, inputs=({inputs}))"

# recursive_property replaces functools.cached_property in recursive UOp functions to prevent RecursionError
class recursive_property(property):
  def __init__(self, fxn):
    self.fxn = fxn
    self.nm = "_RECURSIVE_PROPERTY_"+fxn.__name__
    self.__doc__ = fxn.__doc__
  def __get__(self, x:OpNode|None, owner=None):
    if x is None: return self
    # this is very similar to toposort/topovisit
    stack: list[tuple[OpNode, bool]] = [(x, False)]
    while stack:
      opnode, visited = stack.pop()
      if self.nm in opnode.__dict__: continue
      if not visited:
        stack.append((opnode, True))
        for s in reversed(opnode.inputs): stack.append((s, False))
      else: opnode.__dict__[self.nm] = self.fxn(opnode)
    return x.__dict__[self.nm]

# **************** Expression Graph ****************
@dataclass(eq=False, slots=True) # NOTE: this should be frozen, but frozen is slower
class OpNode(TensorDSL):
  """
  OpNode structs (which Tensor's deusugar into) are vertices that form an
  expression graph G=(V,E) where V is a Set<Op> and E is a Set<(Op,Op)>

  OpNodes structs store state for the
    1. static specification of the function being applied to the expression graph
      a. specified function type of f (OpCode)
      b. resulting *image* of f: _ -> R^(d0xd1x...xdn) (Shape)
    2. dynamic evaluation of the function being applied to the expression graph
      a. arguments passed in (OpNode.inputs)
  """
  opcode: OpCode
  inputs: tuple[OpNode, ...]
  dtype: DType
  payload: Any=None
  # shape, storage (and it's device) are embedded in the IR as opcode's with payloads

  def __repr__(self): return pretty_print(self)
  def payloadstr(self): return f'({", ".join(map(str, self.payload))})' if self.opcode is OpCode.REDUCE_AXIS else repr(self.payload)
  # def tagstr(self): return f", tag={self.tag}" if self.tag is not None else ""

  # **************** Virtual/Logical Shape ****************
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
      case OpCode.UNIQUE | OpCode.DEVICE | OpCode.RANGE | OpCode.LOAD | OpCode.IF | OpCode.BARRIER | OpCode.CUSTOM | OpCode.CUSTOMI | OpCode.VECTORIZE | OpCode.VCONST | OpCode.GEP | OpCode.SPECIAL | OpCode.UNROLL | OpCode.CONTRACT: return None
      case OpCode.INDEX:
        if not isinstance(self.dtype, PtrDType):                                                                                                        return None # non pointer index doesn't have a shape
        elif self.inputs[0]._shape is None or len(self.inputs[1:]) == len(self.inputs[0].shape):                                                        return None # fully indexed doesn't have a shape. TODO: remove this
        else:                                                                                                                                           return self.inputs[0].shape[len(self.inputs[1:]):] # pointer index
      # constructor ops (which init the shape)
      case OpCode.CONST | OpCode.DEFINE_VAR | OpCode.BIND:                                                                                              return () if self._device is not None else None
      case OpCode.BUFFER:                                                                                                                               return (self.payload,)
      case OpCode.BUFFER_VIEW:                                                                                                                          return (self.payload[0],)
      case OpCode.BUFFERIZE:                                                                                                                            return tuple([int(r.vmax+1) for r in self.inputs[1:]])
      case OpCode.DEFINE_GLOBAL | OpCode.DEFINE_LOCAL | OpCode.DEFINE_REG:                                                                              return (self.ptrdtype.size,)
      case OpCode.REDUCE | OpCode.MSTACK | OpCode.MSELECT | OpCode.DETACH | OpCode.CONTIGUOUS | OpCode.CONTIGUOUS_BACKWARD | OpCode.AFTER | OpCode.END: return self.inputs[0]._shape # passthrough ops
      case OpCode.KERNEL:                                                                                                                               return self.payload.ast._shape # ops with custom handling
      case OpCode.BITCAST: # TODO: disallow shape changing bitcast
        ps = self.inputs[0]._shape
        if ps is None: return None
        if (output_sz:=self.dtype.itemsize) != (input_sz:=self.inputs[0].dtype.itemsize): return ps[:-1]+(ssimplify((ps[-1]*input_sz) // output_sz),)
        return ps
      case OpCode.RESHAPE: # TODO: disallow reshape from nothing. tested by TestOpenClip.test_multigpu_clip_score
        if self.inputs[0]._shape is None:                                                                                                               return self.movementopcode_payload

    # COMPUTE ops keep the shape the same. all inputs with shape must match
    if self.opcode in GroupedOpCode.Compute.union({OpCode.CAST, OpCode.COPY, OpCode.ASSIGN, OpCode.NOOP, OpCode.GROUP, OpCode.SINK, OpCode.ALLREDUCE, OpCode.STORE}):
      input_shapes = [x._shape for x in (self.inputs[:2] if self.opcode is OpCode.ASSIGN else self.inputs) if x._shape is not None] # TODO: remove this hack for 3 op assign
      if len(input_shapes) == 0: return None
      if not all_same(input_shapes): raise RuntimeError(f"shape mismatch at {self.opcode}: {input_shapes}")
      return input_shapes[0]
    # MOVEMENT ops change the shape. this is the logic from the old ShapeTracker NOTE: ssimplify is required because the shape needs to be canonical for broadcasting and same shape checking
    elif self.opcode in GroupedOpCode.Movement.union({OpCode.MULTI, OpCode.REDUCE_AXIS, OpCode.WMMA}):
      ps = self.inputs[0]._shape
      if ps is None and self.opcode is OpCode.WMMA: return None # TODO: WMMA is used for both axis WMMA and op WMMA. fix this and remove this hack. tested by BERT on AMD LLVM
      if ps is None: raise RuntimeError(f"movement op {self.opcode} requires shape")

      match self.opcode:
        case OpCode.RESHAPE:
          if not all(x >= 0 for x in self.movementopcode_payload):                                                                                        raise ValueError(f"shape can't contain negative numbers {self.movementopcode_payload}")
          if helpers.prod(ps) != helpers.prod(self.movementopcode_payload):                                                                               raise ValueError(f"bad reshape: {ps} -> {self.movementopcode_payload}")
          return self.movementopcode_payload
        case OpCode.EXPAND:
          foo = len(ps) != len(self.movementopcode_payload) or not all(s==ns or (s==1 and ns>=0) for s,ns in zip(ps, self.movementopcode_payload))
          if foo:                                                                                                                                         raise ValueError(f"bad expand: {ps} -> {self.movementopcode_payload}")
          return self.movementopcode_payload
        case OpCode.PERMUTE:
          foo = sorted(self.movementopcode_payload) != list(range(len(ps)))
          if foo:                                                                                                                                         raise ValueError(f"invalid permutation {self.movementopcode_payload} of len {len(ps)}")
          return tuple(ps[i] for i in self.movementopcode_payload)
        case OpCode.PAD:
          foo = len(ps) != len(self.movementopcode_payload) or not all(resolve(b>=0) and resolve(e>=0) for b,e in self.movementopcode_payload)                                # TODO: why do i need resolve here?
          if foo:                                                                                                                                         raise ValueError(f"invalid pad {self.movementopcode_payload}")
          return tuple(ssimplify(s+b+e) for s,(b,e) in zip(ps, self.movementopcode_payload))
        case OpCode.SHRINK:
          foo = len(ps) != len(self.movementopcode_payload) or not all(resolve(0<=b) and resolve(b<=e) and resolve(e<=s) for s,(b,e) in zip(ps, self.movementopcode_payload)) # TODO: why do i need resolve here?
          if foo:                                                                                                                                         raise ValueError(f"invalid shrink {self.movementopcode_payload} for {ps}")
          return tuple(ssimplify(e-s) for s,e in self.movementopcode_payload)
        case OpCode.FLIP:
          foo = len(ps) != len(self.movementopcode_payload) or not all(isinstance(x, bool) for x in self.movementopcode_payload)
          if foo:                                                                                                                                         raise ValueError(f"bad flip on {ps}, {self.movementopcode_payload}")
          return ps
        case OpCode.MULTI: return tuple(s*len(self.device) if a == self.axis else s for a,s in enumerate(ps))
        case OpCode.REDUCE_AXIS | OpCode.WMMA:
          axis_arg = self.payload[1] if self.opcode is OpCode.REDUCE_AXIS else self.payload[7]
          foo = not isinstance(axis_arg, tuple) or not all(isinstance(x, int) and x>=0 and x<len(ps) for x in axis_arg)
          if foo: raise ValueError(f"invalid type for axis: {axis_arg}")
          return tuple(1 if i in axis_arg else s for i,s in enumerate(ps))

    raise NotImplementedError(f"no shape handling for {self.opcode} with {self.dtype}") # all OpCodes must be explicitly handled

  # **************** Physical Storage (uses runtime/) ****************
  unique_num = itertools.count(0)
  @staticmethod
  def unique(payload:int|None=None):
    return OpNode(OpCode.UNIQUE, tuple(), dtypes.void, payload=next(OpNode.unique_num) if payload is None else payload)

  @staticmethod
  def new_buffer(device:str|tuple[str, ...], size:int, dtype:DType, num=None):
    inputs = (OpNode.unique(num), OpNode(OpCode.DEVICE, tuple(), dtypes.void, payload=device))
    return OpNode(OpCode.BUFFER, inputs, dtype, size)
  
  def copy_to_device(self: Self, device: str|tuple[str, ...]|OpNode):
    if DEBUG >= 1: print(f"OpNode.copy_to_device for opnode {self} to device {device}")
    device_opnode = OpNode(OpCode.DEVICE, (), dtypes.void, payload=device) if not isinstance(device, OpNode) else device
    return OpNode(OpCode.COPY, (self, device_opnode), self.dtype)

  @property
  def device(self) -> str|tuple[str, ...]:                                    return helpers.unwrap(self._device)
  @recursive_property
  def _device(self) -> str|tuple[str, ...] | None:
    if self.opcode is OpCode.DEVICE:                                          return self.payload
    elif self.opcode is OpCode.BUFFERIZE:                                     return self.payload.device
    elif self.opcode is OpCode.AFTER:                                         return self.inputs[0]._device
    elif self.opcode is OpCode.MSELECT:
      assert isinstance(self.inputs[0].device, tuple), "mselect must be on tuple device"
      return self.inputs[0].device[self.payload]
    elif self.opcode is OpCode.MSTACK:                                        return tuple(cast(str, x.device) for x in self.inputs)
    elif self.opcode in {OpCode.COPY, OpCode.BUFFER, OpCode.ALLREDUCE}:       return self.inputs[1].device
    else:
      for input in self.inputs:                                               # otherwise, recurse
        if input._device is not None:                                         return input._device
    return None

  @property
  def buffer(self) -> Buffer:
    from picograd.runtime.device import Buffer
    if self is not self.base: assert self.opcode is OpCode.RESHAPE, f"expected: OpCode.RESHAPE, actual: {self}"; return self.inputs[0].buffer
    assert self.opcode is OpCode.BUFFER, f"expected: OpCode.BUFFER, actual: {self.opcode}"  

    if (cached := buffers.get(self)) is None: buffers[self] = cached = Buffer(self.device, self.dtype.base, self.size).ref(1)
    return cached
  
  @property # NOTE: this is used by the JIT to determine which inputs we capture
  def realized(self) -> Buffer|MultiBuffer|None: return self.buffer if self.opcode in {OpCode.BUFFER, OpCode.MSTACK} and self.buffer.is_allocated() else None
  @property
  def is_realized(self) -> bool: return all(x.base.realized is not None for x in self.base.inputs) if self.base.op is OpCode.MULTI else self.base.realized is not None





  # **************** GraphBuilder Required Methods ****************
  """
  the graphbuilder overrides the semantics of the host language with a nonstandard interpretation (device acceleration of f(x), automatic differentiation of f'(x))
  with ComputeOpCodeBuilder._apply_compute_opcode() and MovementOpCodeBuilder._apply_movement_opcode()
  which acts as the embedded DSL's "parser" by coupling python dunder builtins to be aware of the corresponding IR OpCode

  *:  keep in mind that the semantics of these two methods are applying *ir op code*
      that is, to maintain parity in semantics with tinygrad (and a smooth pedagogical progression),
      the returned OpNode's are still un-{materialized/realized/evaluated}, and caller's (namely tensor.py)
      need to invoke .eval() on the OpNode for eager semantics.

  **: teenygrad follows tinygrad's bent towards object-oriented organization where containers are lazily initialied
      i.e Tensor, OpNode, and Buffer are all non-allocating/evalauting/materializing, and only do so on Tensor.evaluate(), OpNode.evaluate(), and a Buffer.allocate()
  """
  def _forward_computeop(self, opcode: OpCode, *inputs:OpNode) -> Self:
    output_dtype = (self, *inputs)[-1].dtype # use the last input's dtype 
    if opcode in {OpCode.CMPLT, OpCode.CMPNE, OpCode.CMPEQ}: output_dtype = dtypes.bool.vec(output_dtype.count) if output_dtype.count > 1 else dtypes.bool
    return OpNode(opcode, (self,)+inputs, output_dtype,)

  def _forward_movementop(self, opcode: OpCode, payload, same_shape_noop: bool=False) -> Self:
    """
    _apply_movement_opcode is more involved compared to _apply_compute_opcode.
    this is largely because movement opcode's (i.e OpCode.{RESHAPE/EXPAND/PAD/PERMUTE/FLIP/etc...})
    modify the *shape*, which is *logical/virtual* and needs to be mapped to *physical* memory.

    with the application of movement opcode's, there's a design decision to be made.
      1. following the numpy/torch model i.e torch's c10::TensorImpl/c10::StorageImpl like c++'s std::iterator/std::container
         where view operations (movement opcodes) are non-allocating and share the same underlying storage
         tinygrad followed this design decision with their ShapeTracker/LazyBuffer abstractions,
         which mapped logical nd-indices to physical 1d-indices with a *stack* of views via strides

      2. the alternative design decision is to *encode* and embed all shape/movement semantics for a given Tensor *within* the dsl's IR itself
         in order to enable __________ about the shapes with the RANGIFY/POSTOPT op codes, which decouples the *algorithm* with it's *layout/organization*
         and inspired by the dissertations of halide (https://dspace.mit.edu/handle/1721.1/89996) and tvm (https://arxiv.org/abs/1802.04799).
         conflating the algorithm with it's organization (i.e mapping logical shape to physical storage via strides)
         becomes problematic when you want to *vertically split* the shape for _____ optimizations.
         see: https://x.com/__tinygrad__/status/1964037572503752910
    
         so _apply_movement_opcode converts the *payload* (i.e python tuple) for the given movement *opcode* (i.e OpCode.{RESHAPE/EXPAND/PAD/PERMUTE/FLIP/etc...})
         *into* the embedded dsl's IR with OpNode's that have OpCode.{VECTORIZE/VCONST} which are subsequently used as input OpNode's to the originally specified movement OpNode.
         and any subsequent logic that needs to access the embedded payload will use .movementopcode_payload() and .gep(), which are aware of how to extract the payload from the IR
    """
    output_opnodes_inputs = OpNode._convert_movementopcode_payload_to_opnodeir_input(opcode, payload)
    if len(output_opnodes_inputs) == 0:                         output_opnode = OpNode(opcode, (self,), self.dtype, payload)
    else:                                                       output_opnode = OpNode(opcode, (self,) + helpers.normalize_shape(output_opnodes_inputs), self.dtype) # no .simplify() peephole on inputs

    if DEBUG >= 1: print("constructed movement opnode with opcode", output_opnode.opcode)
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

  def toposort(self, gate:Callable|None=None) -> dict[OpNode, None]:
    output: dict[OpNode, None] = {}
    stack: list[tuple[OpNode, bool]] = [(self, False)] # each stack entry is (node, visited_flag)

    while stack:
      opnode, visited = stack.pop()
      if opnode in output: continue
      if not visited:
        if gate is None or gate(opnode):
          stack.append((opnode, True))  # push node back on stack to process after its srcs
          for input in reversed(opnode.inputs): stack.append((input, False)) # push srcs on the stack
      else: output[opnode] = None # second time i'm seeing this node, add it to returned toposort
    return output

  @property
  def base(self) -> OpNode:
    if self.opcode in GroupedOpCode.Movement:       return self.inputs[0].base
    if self.opcode is OpCode.MULTI:                 return self.inputs[0].base  # MULTI is really a VIEW
    return self
  
  def gep(self, i:int) -> int: # like gep, but might return an integer
    match self.opcode:
      case OpCode.CONST:                            return self.payload # TODO: this won't hit on non-peepholed expressions because i'm not .simplfying() on _apply_movementopcode??
      case OpCode.VCONST:                           return self.payload[i]
      case OpCode.VECTORIZE:                        return self.inputs[i].sintify()
      case _: raise RuntimeError(f"no sgep on {self.op}")
  
  @property
  def movementopcode_payload(self):
    match self.opcode:
      case OpCode.RESHAPE | OpCode.EXPAND:          return tuple(self.inputs[1].gep(i) for i in range(self.inputs[1].dtype.count))
      case OpCode.PAD | OpCode.SHRINK:              return tuple((self.inputs[1].gep(i), self.inputs[2].gep(i)) for i in range(self.inputs[1].dtype.count))
      case OpCode.PERMUTE | OpCode.FLIP:            return self.arg
      case _:                                       raise RuntimeError(f"{self.op} is not a MovementOp")

  # **************** Peephole (evaluation) ****************
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

# **************** Expression Graph Linearizer (Scheduler) ****************
# Graph<OpNode> not runnable bc it still contains control/ordering nodes (AFTER, RANGE, BIND),
# no concrete buffer assignments, and no memory layout; device sharding and buffer views have to be expanded;
# and many steps are non-kernel operations like copies/encodes.

# Scheduling linearizes the graph, binds range values, applies multi-device splits, assigns/reuses buffers via
# the memory planner, and then lowers each step to a device-specific runner. Only after ExecItems exist can
# tinygrad optionally fuse batches into device graphs (see GraphRunner/JIT in tinygrad/engine/jit.py)

buffers: weakref.WeakKeyDictionary[OpNode, Buffer] = weakref.WeakKeyDictionary() # this maps BUFFER uops to their device Buffers