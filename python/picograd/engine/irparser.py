"""
this irparser includes teenygrad's intermediate representation and "parser".
since teenygrad is an domain specific language embedded within the host language of python,
the term "parser" is overloaded since teenygrad overrides the semantics of the host language with the GraphBuilder (there is no lexing, parsing, and typechecking),
a composition of ComputeOpCodeBuilder and MovementOpCodeBuilder which map operations to their ir opcodes.
the provided methods implemented on these two mixins are used by the sugared Tensor handle's graph/ir-builder logic.
"""

from __future__ import annotations
from typing import Self
from enum import Enum, IntEnum, auto

from picograd import helpers
from picograd.dtype import Const

# **************** Intermediate Representation ****************
class FastEnum(IntEnum): # wrapper around IntEnum that preserves Enum.__str__ and makes auto() unique across all FastEnum subclasses
  def __str__(self): return Enum.__str__(self)
  @staticmethod
  def _generate_next_value_(_, __, ___, last_values): return 1 + max([0, *last_values, *[max(c) for c in FastEnum.__subclasses__()]])

class OpCode(FastEnum):
  # ** 1 -- defines/special **
  DEFINE_GLOBAL = auto(); DEFINE_VAR = auto(); BIND = auto()                                                         # define GLOBAL/VAR are ptrs to outside the Kernel
  SPECIAL = auto()                                                                                                   # this is a RANGE for GPU dimensions, similar to symbolic shapes but not exactly
  DEFINE_LOCAL = auto(); DEFINE_REG = auto()                                                                         # define LOCAL/REG allocate things

  # ** 2 -- non op opnodes **
  NOOP = auto(); REWRITE_ERROR = auto()                                                                              # uops that aren't rendered
  SINK = auto(); AFTER = auto(); GROUP = auto()                                                                      # AFTER passes src[0] through and promises in the toposort that any consumers of the AFTER run after src[1:]
                                                                                                                     # GROUP is a NOOP that just merges things together
  GEP = auto(); VECTORIZE = auto()                                                                                   # vector creation / item selection

  # ** 3 -- MEMORY **
  INDEX = auto()                                                                                                     # INDEX is a BinaryOp similar to ADD, but it operates on pointers
  LOAD = auto(); STORE = auto()                                                                                      # load/store before math

  # ** 4 -- COMPUTE **
  WMMA = auto()                                                                                                      # tensor core math op, not elementwise

  CAST = auto(); BITCAST = auto(); EXP2 = auto(); LOG2 = auto(); SIN = auto()                                        # UnaryOps
  SQRT = auto(); RECIPROCAL = auto(); NEG = auto(); TRUNC = auto()

  ADD = auto(); MUL = auto(); SHL = auto(); SHR = auto(); IDIV = auto(); MAX = auto(); MOD = auto()                  # BinaryOps
  CMPLT = auto(); CMPNE = auto(); CMPEQ = auto()
  XOR = auto(); OR = auto(); AND = auto()
  THREEFRY = auto(); SUB = auto(); FDIV = auto(); POW = auto()

  WHERE = auto(); MULACC = auto()                                                                                    # TernaryOps

  # ** 5 -- control flow / consts / custom **
  BARRIER = auto(); RANGE = auto(); IF = auto(); END = auto(); ENDIF = auto()                                        # control flow ops
  VCONST = auto(); CONST = auto()                                                                                    # consts. VCONST is a vectorized const
  CUSTOM = auto(); CUSTOMI = auto()                                                                                  # CUSTOM/CUSTOMI are used to output strings into codegen. the I makes the string inline

  # ** 6 -- ops that don't exist in programs **
  UNIQUE = auto(); DEVICE = auto(); KERNEL = auto(); ASSIGN = auto()                                                 # tensor graph ops
  CONTIGUOUS = auto(); CONTIGUOUS_BACKWARD = auto(); DETACH = auto()                                                 # ops that adjust the behavior of the scheduler
  BUFFERIZE = auto(); COPY = auto(); BUFFER = auto(); BUFFER_VIEW = auto(); MSELECT = auto(); MSTACK = auto()        # buffer ops
  RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); FLIP = auto()                  # the core 6 movement ops! these only exist in the tensor graph
  MULTI = auto()                                                                                                     # MULTI is really a movement op
  REDUCE_AXIS = auto(); REDUCE = auto(); ALLREDUCE = auto()                                                          # reduce
  UNROLL = auto(); CONTRACT = auto(); CAT = auto(); PTRCAT = auto()                                                  # expander ops

class GroupedOpCode:
  Unary =        {OpCode.EXP2, OpCode.LOG2, OpCode.SIN, OpCode.SQRT, OpCode.RECIPROCAL, OpCode.NEG, OpCode.TRUNC}
  Binary =       {OpCode.ADD, OpCode.MUL, OpCode.IDIV, OpCode.MAX, OpCode.MOD, OpCode.CMPLT, OpCode.CMPNE, OpCode.CMPEQ,
                  OpCode.XOR, OpCode.SHL, OpCode.SHR, OpCode.OR, OpCode.AND, OpCode.THREEFRY, OpCode.SUB, OpCode.FDIV, OpCode.POW}
  Ternary =      {OpCode.WHERE, OpCode.MULACC}
  Compute =      set.union(Unary, Binary, Ternary)

  Elementwise =  set.union(Compute, {OpCode.CAST, OpCode.BITCAST}) # TODO: is BITCAST always Elementwise if it's shape changing?
  Defines =      {OpCode.DEFINE_GLOBAL, OpCode.DEFINE_LOCAL, OpCode.DEFINE_REG}
  Irreducible =  {OpCode.CONST, OpCode.DEFINE_VAR, OpCode.SPECIAL, OpCode.RANGE}
  Movement =     {OpCode.RESHAPE, OpCode.EXPAND, OpCode.PERMUTE, OpCode.PAD, OpCode.SHRINK, OpCode.FLIP}
  Buffer =       {OpCode.LOAD, OpCode.STORE, OpCode.CONST, OpCode.DEFINE_VAR}

  Commutative =  {OpCode.ADD, OpCode.MUL, OpCode.MAX, OpCode.CMPNE, OpCode.CMPEQ, OpCode.XOR, OpCode.AND, OpCode.OR} # BinaryOps that can be flipped
  Associative =  {OpCode.ADD, OpCode.MUL, OpCode.AND, OpCode.OR, OpCode.MAX}                                         # BinaryOps where f(f(a,b),c) = f(a,f(b,c))
  Idempotent =   {OpCode.OR, OpCode.AND, OpCode.MAX}                                                                 # BinaryOps where f(x,x)=x
  Comparison =   {OpCode.CMPLT, OpCode.CMPNE, OpCode.CMPEQ}                                                          # These can change the dtype to bool
  UnsafePad =    {OpCode.RECIPROCAL, OpCode.LOG2, OpCode.EXP2, OpCode.IDIV, OpCode.POW}                              # do not preserve f(0) = 0
  All =          set(OpCode)


# **************** GraphBuilder: ComputeOpCodeBuilder * MovementOpCodeBuilder ****************
"""
GraphBuilder (at the bottom of the file) is a ComputeOpCodeBuilder and MovementOpCodeBuilder which effectively
1. removes the repetition between sugared and desugared Tensor/Op
2. acts as the embedded DSL's "parser", by coupling python dunder builtins to be aware of the corresponding OpCode ir
"""

class ComputeOpCodeBuilder:
  # required
  def _apply_compute_opcode(self, ftype: OpCode, *inputs: Self) -> Self: raise NotImplementedError
  def const_like(self, b: Const) -> Self: raise NotImplementedError

  # provided
  def _apply_compute_binopcode(self, op: OpCode, x: Self | Const, reverse: bool) -> Self:
    return self.ufix(x)._apply_compute_opcode(op, self) if reverse else self._apply_compute_opcode(op, self.ufix(x))
  def ufix(self, x: Self | Const) -> Self:
    return self.const_like(x) if not isinstance(x, ComputeOpCodeBuilder) else x

  def neg(self):
    if (dtype := getattr(self, "dtype")) is None:
      raise TypeError(f"MathTraits __neg__ requires a dtype, {self=}")
    return self.logical_not() if dtype.scalar() == dtypes.bool else self * (-1)
  def add(self, x: Self | Const, reverse: bool = False):                                                     return self._apply_compute_binopcode(OpCode.ADD, x, reverse)
  def sub(self, x: Self | Const, reverse: bool = False):                                                     return self.ufix(x)._apply_compute_opcode(OpCode.ADD, -self) if reverse else self._apply_compute_opcode(OpCode.ADD, self.ufix(-x))
  def mul(self, x: Self | Const, reverse: bool = False):                                                     return self._apply_compute_binopcode(OpCode.MUL, x, reverse)
  def idiv(self, x: Self | Const, reverse: bool = False):                                                    return self._apply_compute_binopcode(OpCode.IDIV, x, reverse)
  def mod(self, x: Self | Const, reverse: bool = False):                                                     return self._apply_compute_binopcode(OpCode.MOD, x, reverse)
  def div(self, x: Self | Const, reverse: bool = False): return (self.ufix(x) * self._apply_compute_opcode(OpCode.RECIP)) if reverse else (self * self.ufix(x)._apply_compute_opcode(OpCode.RECIP))
  def recip(self):                                                                                           return self._apply_compute_opcode(OpCode.RECIP)
  def trunc(self):                                                                                           return self._apply_compute_opcode(OpCode.TRUNC)
  def sqrt(self):                                                                                            return self._apply_compute_opcode(OpCode.SQRT)
  def sin(self):                                                                                             return self._apply_compute_opcode(OpCode.SIN)
  def log2(self):                                                                                            return self._apply_compute_opcode(OpCode.LOG2)
  def exp2(self):                                                                                            return self._apply_compute_opcode(OpCode.EXP2)
  def pow(self, x: Self | Const):                                                                            return self._apply_compute_opcode(OpCode.POW, self.ufix(x))
  def maximum(self, x: Self | Const):                                                                        return self._apply_compute_opcode(OpCode.MAX, self.ufix(x))
  def minimum(self, x: Self | Const): return -(-self).maximum(-x)
  def threefry(self, seed: Self): return self._apply_compute_opcode(OpCode.THREEFRY, seed)
  def bitwise_and(self, x: Self | Const, reverse: bool = False): self._check_dtype();                        return self._apply_compute_binopcode(OpCode.AND, x, reverse)
  def bitwise_or(self, x: Self | Const, reverse: bool = False): self._check_dtype();                         return self._apply_compute_binopcode(OpCode.OR, x, reverse)
  def bitwise_xor(self, x: Self | Const, reverse: bool = False): self._check_dtype();                        return self._apply_compute_binopcode(OpCode.XOR, x, reverse)
  def lshift(self, x: Self | int, reverse: bool = False): return self._apply_compute_binopcode(OpCode.SHL, x, reverse)
  def rshift(self, x: Self | int, reverse: bool = False): return self._apply_compute_binopcode(OpCode.SHR, x, reverse)
  def where(self, x: Self | Const, y: Self | Const):
    if isinstance(x, type(self)):
      return self._apply_compute_opcode(OpCode.WHERE, x, x.ufix(y))
    if isinstance(y, type(self)):
      return self._apply_compute_opcode(OpCode.WHERE, y.ufix(x), y)
    raise RuntimeError("where needs at least one UOp arg")
  def logical_not(self): return self.ne(True)
  
  def __neg__(self):                                                                                         return self.neg()
  def __add__(self, x: Self | Const):                                                                        return self.add(x)
  def __radd__(self, x: Self | Const):                                                                       return self.add(x, True)
  def __sub__(self, x: Self | Const):                                                                        return self.sub(x)
  def __rsub__(self, x: Self | Const):                                                                       return self.sub(x, True)
  def __mul__(self, x: Self | Const):                                                                        return self.mul(x)
  def __rmul__(self, x: Self | Const):                                                                       return self.mul(x, True)
  def __pow__(self, x: Self | Const):                                                                        return self.pow(x)
  def __truediv__(self, x: Self | Const):                                                                    return self.div(x)
  def __rtruediv__(self, x: Self | Const):                                                                   return self.div(x, True)
  def __floordiv__(self, x: Self | Const):                                                                   return self.idiv(x)  # TODO: idiv is trunc div, not floordiv
  def __rfloordiv__(self, x: Self | Const):                                                                  return self.idiv(x, True)
  def __mod__(self, x: Self | Const):                                                                        return self.mod(x)
  def __rmod__(self, x: Self | Const):                                                                       return self.mod(x, True)
  
  def __lt__(self, x: Self | Const):                                                                         return self._apply_compute_opcode(OpCode.CMPLT, self.ufix(x))
  def __gt__(self, x: Self | Const):                                                                         return self.ufix(x)._apply_compute_opcode(OpCode.CMPLT, self)
  def __ge__(self, x: Self | Const):                                                                         return (self < x).logical_not()
  def __le__(self, x: Self | Const):                                                                         return (self > x).logical_not()
  def ne(self, x: Self | Const):                                                                             return self._apply_compute_opcode(OpCode.CMPNE, self.ufix(x))
  def eq(self, x: Self | Const):                                                                             return self.ne(x).logical_not()
  def __ne__(self, x: Self | Const):                                                                         return self.ne(x)  # type: ignore[override]
  # NOTE: __eq__ isn't overridden, and means the same thing as is b default

  def __and__(self, x: Self | Const):                                                                        return self.bitwise_and(x)
  def __or__(self, x: Self | Const):                                                                         return self.bitwise_or(x)
  def __xor__(self, x: Self | Const):                                                                        return self.bitwise_xor(x)
  def __rand__(self, x: Self | Const):                                                                       return self.bitwise_and(x, True)
  def __ror__(self, x: Self | Const):                                                                        return self.bitwise_or(x, True)
  def __rxor__(self, x: Self | Const):                                                                       return self.bitwise_xor(x, True)

  def __lshift__(self, x: Self | int):                                                                       return self.lshift(x)
  def __rshift__(self, x: Self | int):                                                                       return self.rshift(x)
  def __rlshift__(self, x: Self | int):                                                                      return self.lshift(x, True)
  def __rrshift__(self, x: Self | int):                                                                      return self.rshift(x, True)

  def _check_dtype(self):
    if (dtype := getattr(self, "dtype")) is not None:
      if isinstance(dtype, tuple):
        dtype = dtype[0]
      if not (dtypes.is_bool(dtype) or dtypes.is_int(dtype)):
        raise RuntimeError(f"{dtype} is not supported")

class MovementOpCodeBuilder:
  # required
  def _apply_movement_opcode(self, op: OpCode, payload) -> Self: raise NotImplementedError
  @property
  def shape(self) -> tuple[int, ...]: raise NotImplementedError
  
  # provided
  def expand(self) -> Self: raise NotImplementedError("todo")
  def reshape(self, shape: tuple[int, ...]) -> Self:
    output_shape = tuple([dim_i if dim_i is not None else self.shape[i] for i, dim_i in enumerate(helpers.normalize_shape(shape))])                      # handle None and args
    if inferred_dim_count := output_shape.count(-1) >= 1:                                                                                                # handle -1 inferred dims
      if inferred_dim_count > 1: raise RuntimeError(f"only one dimension can be inferred using -1, getting {output_shape}")
      else: output_shape = tuple([-helpers.prod(self.shape) // helpers.prod(output_shape) if dim == -1 else dim for dim in output_shape])
    if helpers.prod(self.shape) != helpers.prod(output_shape): raise ValueError(f"size mismatch, can't reshape ({self.shape}) -> ({output_shape})")      # guard
    
    output = self._apply_movement_opcode(OpCode.RESHAPE, payload=output_shape)                                                                           # apply
    return self if output.shape == self.shape else output                                                                                                # identity return

  def shrink(self) -> Self: raise NotImplementedError("todo")
  def permute(self) -> Self: raise NotImplementedError("todo")
  def flip(self) -> Self: raise NotImplementedError("todo")

  def view(self) -> Self: raise NotImplementedError("todo")
  def squeeze(self) -> Self: raise NotImplementedError("todo")
  def unsqueeze(self) -> Self: raise NotImplementedError("todo")

  def transpose(self) -> Self: raise NotImplementedError("todo")
  def flatten(self) -> Self: raise NotImplementedError("todo")
  def unflatten(self) -> Self: raise NotImplementedError("todo")

class GraphBuilder(ComputeOpCodeBuilder, MovementOpCodeBuilder):
  pass