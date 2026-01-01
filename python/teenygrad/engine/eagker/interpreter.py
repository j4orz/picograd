from dataclasses import dataclass
from typing import TYPE_CHECKING, List
import pathlib

if TYPE_CHECKING: from teenygrad.frontend.tensor import Tensor
from teenygrad.engine import OpCode, OpNode
from teenygrad.runtime.device import Device, Runtime

@dataclass(frozen=True)
class ExternalKernel:
  name: str; source: str|pathlib.Path; compiler: str = "nvcc"
  globals: tuple[int, int, int] = (1, 1, 1); locals: tuple[int, int, int] = (1, 1, 1)
  args: tuple[int, ...] = (); extra_args: tuple[str, ...] = ()
shared_mem: int = 0

class Interpreter:
  def __init__(self, device: str):
    self.device: Runtime = Device[device]

  @staticmethod
  def evaluate(schedule: List[OpNode]) -> List[Buffer]:
    i = 0
    while i < len(schedule):
      opnode = schedule[i]
      if opnode.opcode is OpCode.ADD:
        left, right = self.evaluate(opnode.inputs[0]), self.evaluate(opnode.inputs[0])
      else: raise NotImplementedError("todo")
