from __future__ import annotations
from typing import Self
from enum import Enum, IntEnum, auto

from picograd.dtype import ConstType

# **************** Intermediate Representation ****************
class FastEnum(IntEnum): # wrapper around IntEnum that preserves Enum.__str__ and makes auto() unique across all FastEnum subclasses
  def __str__(self): return Enum.__str__(self)
  @staticmethod
  def _generate_next_value_(_, __, ___, last_values): return 1 + max([0, *last_values, *[max(c) for c in FastEnum.__subclasses__()]])

class OpCode(FastEnum):
  """
  the order of these OpCode controls the order of the toposort
  """
  # ** 1 -- defines/special **
  DEFINE_GLOBAL = auto(); DEFINE_VAR = auto(); BIND = auto()                                                  # define GLOBAL/VAR are ptrs to outside the Kernel
  SPECIAL = auto()                                                                                            # this is a RANGE for GPU dimensions, similar to symbolic shapes but not exactly
  DEFINE_LOCAL = auto(); DEFINE_REG = auto()                                                                  # define LOCAL/REG allocate things

  # ** 2 -- non op uops **
  NOOP = auto(); REWRITE_ERROR = auto()                                                                       # uops that aren't rendered
  SINK = auto(); AFTER = auto(); GROUP = auto()                                                               # AFTER passes src[0] through and promises in the toposort that any consumers of the AFTER run after src[1:]
                                                                                                              # GROUP is a NOOP that just merges things together
  GEP = auto(); VECTORIZE = auto()                                                                            # vector creation / item selection

  # ** 3 -- MEMORY **
  INDEX = auto()                                                                                              # INDEX is a BinaryOp similar to ADD, but it operates on pointers
  LOAD = auto(); STORE = auto()                                                                               # load/store before math

  # ** 4 -- COMPUTE **
  WMMA = auto()                                                                                               # tensor core math op, not elementwise

  CAST = auto(); BITCAST = auto(); EXP2 = auto(); LOG2 = auto(); SIN = auto()                                 # UnaryOps
  SQRT = auto(); RECIPROCAL = auto(); NEG = auto(); TRUNC = auto()

  ADD = auto(); MUL = auto(); SHL = auto(); SHR = auto(); IDIV = auto(); MAX = auto(); MOD = auto()           # BinaryOps
  CMPLT = auto(); CMPNE = auto(); CMPEQ = auto()
  XOR = auto(); OR = auto(); AND = auto()
  THREEFRY = auto(); SUB = auto(); FDIV = auto(); POW = auto()

  WHERE = auto(); MULACC = auto()                                                                             # TernaryOps

  # ** 5 -- control flow / consts / custom **
  BARRIER = auto(); RANGE = auto(); IF = auto(); END = auto(); ENDIF = auto()                                 # control flow ops
  VCONST = auto(); CONST = auto()                                                                             # consts. VCONST is a vectorized const
  CUSTOM = auto(); CUSTOMI = auto()                                                                           # CUSTOM/CUSTOMI are used to output strings into codegen. the I makes the string inline

  # ** 6 -- ops that don't exist in programs **
  UNIQUE = auto(); DEVICE = auto(); KERNEL = auto(); ASSIGN = auto()                                          # tensor graph ops
  CONTIGUOUS = auto(); CONTIGUOUS_BACKWARD = auto(); DETACH = auto()                                          # ops that adjust the behavior of the scheduler
  BUFFERIZE = auto(); COPY = auto(); BUFFER = auto(); BUFFER_VIEW = auto(); MSELECT = auto(); MSTACK = auto() # buffer ops
  RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); FLIP = auto()           # the core 6 movement ops! these only exist in the tensor graph
  MULTI = auto()                                                                                              # MULTI is really a movement op
  REDUCE_AXIS = auto(); REDUCE = auto(); ALLREDUCE = auto()                                                   # reduce
  UNROLL = auto(); CONTRACT = auto(); CAT = auto(); PTRCAT = auto()                                           # expander ops

# **************** OpMixin: ComputeMixin * MovementMixin ****************
"""
OpMixin (at the bottom of the file) is a ComputeMixin and MovementMixin which effectively
1. removes the repetition between sugared and desugared Tensor/Op
2. acts as the embedded DSL's "parser", by coupling python dunder builtins to be aware of the corresponding OpCode intermediate representation
the dunders call the provided mixins' methods, which in turn call .eval(), which is implemented by subclasses.
"""

class ComputeMixin:
  # required
  def eval(self, ftype: OpCode, *inputs: Self) -> Self: raise NotImplementedError
  def const_like(self, b: ConstType) -> Self: raise NotImplementedError

  # provided
  def _binop(self, op: OpCode, x: Self | ConstType, reverse: bool) -> Self:
    return self.ufix(x).eval(op, self) if reverse else self.eval(op, self.ufix(x))
  def ufix(self, x: Self | ConstType) -> Self:
    return self.const_like(x) if not isinstance(x, ComputeMixin) else x

  def neg(self):
    if (dtype := getattr(self, "dtype")) is None:
      raise TypeError(f"MathTraits __neg__ requires a dtype, {self=}")
    return self.logical_not() if dtype.scalar() == dtypes.bool else self * (-1)
  def add(self, x: Self | ConstType, reverse: bool = False): return self._binop(OpCode.ADD, x, reverse)
  def sub(self, x: Self | ConstType, reverse: bool = False): return self.ufix(x).eval(OpCode.ADD, -self) if reverse else self.eval(OpCode.ADD, self.ufix(-x))
  def mul(self, x: Self | ConstType, reverse: bool = False): return self._binop(OpCode.MUL, x, reverse)
  def idiv(self, x: Self | ConstType, reverse: bool = False): return self._binop(OpCode.IDIV, x, reverse)
  def mod(self, x: Self | ConstType, reverse: bool = False): return self._binop(OpCode.MOD, x, reverse)
  def div(self, x: Self | ConstType, reverse: bool = False): return (self.ufix(x) * self.eval(OpCode.RECIP)) if reverse else (self * self.ufix(x).eval(OpCode.RECIP))
  def recip(self): return self.eval(OpCode.RECIP)
  def trunc(self): return self.eval(OpCode.TRUNC)
  def sqrt(self): return self.eval(OpCode.SQRT)
  def sin(self): return self.eval(OpCode.SIN)
  def log2(self): return self.eval(OpCode.LOG2)
  def exp2(self): return self.eval(OpCode.EXP2)
  def pow(self, x: Self | ConstType): return self.eval(OpCode.POW, self.ufix(x))
  def maximum(self, x: Self | ConstType): return self.eval(OpCode.MAX, self.ufix(x))
  def minimum(self, x: Self | ConstType): return -(-self).maximum(-x)
  def threefry(self, seed: Self): return self.eval(OpCode.THREEFRY, seed)
  def bitwise_and(self, x: Self | ConstType, reverse: bool = False): self._check_dtype(); return self._binop(OpCode.AND, x, reverse)
  def bitwise_or(self, x: Self | ConstType, reverse: bool = False): self._check_dtype(); return self._binop(OpCode.OR, x, reverse)
  def bitwise_xor(self, x: Self | ConstType, reverse: bool = False): self._check_dtype(); return self._binop(OpCode.XOR, x, reverse)
  def lshift(self, x: Self | int, reverse: bool = False): return self._binop(OpCode.SHL, x, reverse)
  def rshift(self, x: Self | int, reverse: bool = False): return self._binop(OpCode.SHR, x, reverse)
  def where(self, x: Self | ConstType, y: Self | ConstType):
    if isinstance(x, type(self)):
      return self.eval(OpCode.WHERE, x, x.ufix(y))
    if isinstance(y, type(self)):
      return self.eval(OpCode.WHERE, y.ufix(x), y)
    raise RuntimeError("where needs at least one UOp arg")
  def logical_not(self): return self.ne(True)
  
  def __neg__(self): return self.neg()
  def __add__(self, x: Self | ConstType): return self.add(x)
  def __radd__(self, x: Self | ConstType): return self.add(x, True)
  def __sub__(self, x: Self | ConstType): return self.sub(x)
  def __rsub__(self, x: Self | ConstType): return self.sub(x, True)
  def __mul__(self, x: Self | ConstType): return self.mul(x)
  def __rmul__(self, x: Self | ConstType): return self.mul(x, True)
  def __pow__(self, x: Self | ConstType): return self.pow(x)
  def __truediv__(self, x: Self | ConstType): return self.div(x)
  def __rtruediv__(self, x: Self | ConstType): return self.div(x, True)
  def __floordiv__(self, x: Self | ConstType): return self.idiv(x)  # TODO: idiv is trunc div, not floordiv
  def __rfloordiv__(self, x: Self | ConstType): return self.idiv(x, True)
  def __mod__(self, x: Self | ConstType): return self.mod(x)
  def __rmod__(self, x: Self | ConstType): return self.mod(x, True)
  
  def __lt__(self, x: Self | ConstType): return self.eval(OpCode.CMPLT, self.ufix(x))
  def __gt__(self, x: Self | ConstType): return self.ufix(x).eval(OpCode.CMPLT, self)
  def __ge__(self, x: Self | ConstType): return (self < x).logical_not()
  def __le__(self, x: Self | ConstType): return (self > x).logical_not()
  def ne(self, x: Self | ConstType): return self.eval(OpCode.CMPNE, self.ufix(x))
  def eq(self, x: Self | ConstType): return self.ne(x).logical_not()
  def __ne__(self, x: Self | ConstType): return self.ne(x)  # type: ignore[override]
  # NOTE: __eq__ isn't overridden, and means the same thing as is b default

  def __and__(self, x: Self | ConstType): return self.bitwise_and(x)
  def __or__(self, x: Self | ConstType): return self.bitwise_or(x)
  def __xor__(self, x: Self | ConstType): return self.bitwise_xor(x)
  def __rand__(self, x: Self | ConstType): return self.bitwise_and(x, True)
  def __ror__(self, x: Self | ConstType): return self.bitwise_or(x, True)
  def __rxor__(self, x: Self | ConstType): return self.bitwise_xor(x, True)

  def __lshift__(self, x: Self | int): return self.lshift(x)
  def __rshift__(self, x: Self | int): return self.rshift(x)
  def __rlshift__(self, x: Self | int): return self.lshift(x, True)
  def __rrshift__(self, x: Self | int): return self.rshift(x, True)

  def _check_dtype(self):
    if (dtype := getattr(self, "dtype")) is not None:
      if isinstance(dtype, tuple):
        dtype = dtype[0]
      if not (dtypes.is_bool(dtype) or dtypes.is_int(dtype)):
        raise RuntimeError(f"{dtype} is not supported")

class MovementMixin:
  def expand(self) -> Self: raise NotImplementedError("todo")
  def reshape(self, shape) -> Self: raise NotImplementedError("todo")
  def shrink(self) -> Self: raise NotImplementedError("todo")
  def permute(self) -> Self: raise NotImplementedError("todo")
  def flip(self) -> Self: raise NotImplementedError("todo")

  def view(self) -> Self: raise NotImplementedError("todo")
  def squeeze(self) -> Self: raise NotImplementedError("todo")
  def unsqueeze(self) -> Self: raise NotImplementedError("todo")

  def transpose(self) -> Self: raise NotImplementedError("todo")
  def flatten(self) -> Self: raise NotImplementedError("todo")
  def unflatten(self) -> Self: raise NotImplementedError("todo")

class OpMixin(ComputeMixin, MovementMixin):
  pass