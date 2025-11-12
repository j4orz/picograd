from __future__ import annotations
from typing import Callable, Sequence
import math
from dataclasses import dataclass
from enum import auto, IntEnum, Enum
from picograd.dtype import DType, dtypes

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
  MOOOOOOSEEEEE
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