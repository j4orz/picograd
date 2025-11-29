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
  ...
  the order of these OpCode controls the order of the toposort
  """
  # ops that aren't rendered: noop, sink, unique, device, kernel, precast, rewrite_error, sentinel, after, group
  # buffer ops: copy, buffer, buffer_view, mselect, mstack, bufferize, contiguous, contiguous_backward
  # movement ops: these only exist in the tensor graph reshape, permute, expand, pad, shrink, flip, multi
  # def_global, def_local, def_reg, def_var, bind, special(range)
  # reduce_axis, reduce, allreduce
  # unroll, contract, gep, vectorize, cat, ptrcat

  # CAST = auto(); BITCAST = auto();
  # unaryops
  EXP2 = auto(); LOG2 = auto(); SIN = auto(); SQRT = auto(); RECIP = auto(); NEG = auto(); TRUNC = auto()
  # laod, store, assign, wmma, index

  MM = auto(); FA = auto() # TODO: order??
  # binary ops
  ADD = auto(); MUL = auto(); SHL = auto(); SHR = auto(); IDIV = auto(); MAX = auto(); MOD = auto()

  # CMPLT = auto(); CMPNE = auto(); CMPEQ = auto()
  # XOR = auto(); OR = auto(); AND = auto()
  # THREEFRY = auto(); SUB = auto(); FDIV = auto(); POW = auto()
  # where, mulacc
  # barrier, range, if, end, endif
  # vconst, const, custom, customi

# **************** OpMixin: ComputeMixin * MovementMixin ****************
"""
OpMixin (at the bottom of the file) is a ComputeMixin and MovementMixin provide
dunder builtins that call their non-dunder equivalents whom are coupled and aware of the corresponding OpCode.
these OpCodes are applied in their corresponding context (sugared Tensors, desugared Ops) by implementor-provided .alu() and .const_like().
this abstraction effectively removes the repetition happening between sugared and desugared Tensor/Op.
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
  def reciprocal(self): return self.eval(OpCode.RECIP)
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
  def expand(): raise NotImplementedError("todo")
  def reshape(): raise NotImplementedError("todo")
  def shrink(): raise NotImplementedError("todo")
  def permute(): raise NotImplementedError("todo")
  def flip(): raise NotImplementedError("todo")

  def view(): raise NotImplementedError("todo")
  def squeeze(): raise NotImplementedError("todo")
  def unsqueeze(): raise NotImplementedError("todo")

  def transpose(): raise NotImplementedError("todo")
  def flatten(): raise NotImplementedError("todo")
  def unflatten(): raise NotImplementedError("todo")

class OpMixin(ComputeMixin, MovementMixin):
  pass