from dataclasses import dataclass
from typing import TYPE_CHECKING
import pathlib

if TYPE_CHECKING: from picograd.sugar.tensor import Tensor
from picograd.engine import OpCode, OpNode

@dataclass(frozen=True)
class ExternalKernel:
  name: str; source: str|pathlib.Path; compiler: str = "nvcc"
  globals: tuple[int, int, int] = (1, 1, 1); locals: tuple[int, int, int] = (1, 1, 1)
  args: tuple[int, ...] = (); extra_args: tuple[str, ...] = ()
shared_mem: int = 0

# class Interpreter:
#   @staticmethod
#   def evaluate(t: Tensor) -> Tensor:
#     raise  NotImplementedError("todo")