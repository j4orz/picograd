from __future__ import annotations
from typing import Final, Literal
from dataclasses import dataclass
from enum import Enum, auto

class AddrSpace(Enum):
  def __repr__(self): return str(self)
  GLOBAL = auto(); LOCAL = auto(); REG = auto()  # noqa: E702

# all DTypes should only be created once
class DTypeMetaClass(type):
  dcache: dict[tuple, DType] = {}
  def __call__(cls, *args, **kwargs):
    if (ret:=DTypeMetaClass.dcache.get(args, None)) is not None: return ret
    DTypeMetaClass.dcache[args] = ret = super().__call__(*args)
    return ret

@dataclass(frozen=True, eq=False)
class DType(metaclass=DTypeMetaClass):
  priority: int  # this determines when things get upcasted
  itemsize: int
  name: str
  fmt: FmtStr|None
  count: int
  _scalar: DType|None

@dataclass(frozen=True, eq=False)
class PtrDType(DType):
  _base: DType
  addrspace: AddrSpace
  v: int
  size: int = -1  # -1 is unlimited size