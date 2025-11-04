from __future__ import annotations
from dataclasses import dataclass
from enum import auto, IntEnum, Enum
from picograd.mixins import ComputeMixin

@dataclass(eq=False, slots=True)
class UOp(ComputeMixin, MovementMixin, metaclass=UOpMetaClass):
  src:tuple[UOp, ...] = tuple(); op:Ops; dtype:DType = dtypes.void
  arg:Any = None; tag:Any = None

  @property
  def device(self) -> str|tuple[str, ...]: raise NotImplementedError("todo")
  @property
  def shape(self) -> tuple[sint, ...]: raise NotImplementedError("todo")














# wrapper around IntEnum that preserves Enum.__str__ and makes auto() unique across all FastEnum subclasses
class FastEnum(IntEnum):
  def __str__(self): return Enum.__str__(self)
  @staticmethod
  def _generate_next_value_(_, __, ___, last_values): return 1 + max([0, *last_values, *[max(c) for c in FastEnum.__subclasses__()]])

# the order of these Ops controls the order of the toposort
class OpCode(FastEnum):

  # high level ops <-- order???
  FA = auto(); MM = auto()

  # movement ops! these only exist in the tensor graph
  RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); FLIP = auto() # noqa: E702
  MULTI = auto()  # MULTI is really a movement op

  # reduce ops
  REDUCE_AXIS = auto(); REDUCE = auto(); ALLREDUCE = auto() # noqa: E702

  # unary ops
  CAST = auto(); BITCAST = auto(); EXP2 = auto(); LOG2 = auto(); SIN = auto(); SQRT = auto(); RECIP = auto(); NEG = auto(); TRUNC = auto() # noqa: E702

  # memory ops before math
  LOAD = auto(); STORE = auto() # noqa: E702
  ASSIGN = auto()  # TODO: ASSIGN is STORE, remove ASSIGN

  # tensor core math op, not elementwise
  WMMA = auto()

  # binary ops
  ADD = auto(); MUL = auto(); SHL = auto(); SHR = auto(); IDIV = auto(); MAX = auto(); MOD = auto() # noqa: E702
  CMPLT = auto(); CMPNE = auto(); CMPEQ = auto() # noqa: E702
  XOR = auto(); OR = auto(); AND = auto() # noqa: E702
  THREEFRY = auto(); SUB = auto(); FDIV = auto(); POW = auto() # noqa: E702