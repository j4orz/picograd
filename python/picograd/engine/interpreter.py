from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from picograd.sugar.tensor import Tensor

class Interpreter:
  @staticmethod
  def evaluate(t: Tensor) -> Tensor:
    raise  NotImplementedError("todo")