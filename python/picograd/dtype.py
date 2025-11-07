from __future__ import annotations
from typing import Literal
from dataclasses import dataclass

ConstType = float|int|bool
FmtStr = Literal['?', 'b', 'B', 'h', 'H', 'i', 'I', 'q', 'Q', 'e', 'f', 'd']

@dataclass(frozen=True, eq=False)
class DType(): #metaclass=DTypeMetaClass):
  priority: int  # this determines when things get upcasted
  itemsize: int
  name: str
  fmt: FmtStr|None
  count: int
  _scalar: DType|None

DTypeLike = str|DType
def to_dtype(dtype:DTypeLike) -> DType:  raise NotImplementedError("todo") #return dtype if isinstance(dtype, DType) else getattr(dtypes, dtype.lower())