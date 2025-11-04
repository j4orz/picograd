from __future__ import annotations
import math, os
from typing import Callable
from picograd.engine.ir import UOp
from picograd.mixins import ComputeMixin
# from . import _pgrs

class Tensor(ComputeMixin): # , MovementMixin):
  """
  picograd's tensor frontend follows the numpy/torch tensor architecture
  see: (Dodson, Lewis 1985) https://dl.acm.org/doi/pdf/10.1145/1057935.1057937

  - Level  2:     (BCKWD)   provides the high level mathematical primitives of compilation, automatic differentiation, gradient descent
  - Level  0/1:   (FWD)     provides performance primitives that require the knowledge of microarchitecture to obtain peak theoretical throughput (FLOP/S)
  - Level -1:     (DATA)    provides the foundational multi-dimensional array data structure
  """
  __slots__ = "uop", "requires_grad", "grad"  # runtime data like device, shape, and dtype are deleted to uop, not tensor
  @property
  def device(self) -> str|tuple[str, ...]: return self.uop.device
  @property
  def shape(self) -> tuple[sint, ...]: return self.uop.shape
  @property
  def dtype(self) -> DType: return self.uop.dtype

  # ***** Tensor Compile, AD, GD (Level 2) *****
  def gradient(self, *targets:Tensor, gradient:Tensor|None=None, materialize_grads=False) -> list[Tensor]:
    assert gradient is not None or self.shape == tuple(), "when no gradient is provided, backward must be called on a scalar tensor"
    if not (self.is_floating_point() and all(t.is_floating_point() for t in targets)): raise RuntimeError("only float Tensors have gradient")
    if gradient is None: gradient = Tensor(1.0, dtype=self.dtype, device=self.device, requires_grad=False)
    target_uops = [x.uop for x in targets]
    grads = compute_gradient(self.uop, gradient.uop, set(target_uops))
    ret = []
    for x in target_uops:
      if (y:=grads.get(x)) is None:
        if materialize_grads: y = x.const_like(0)
        else: raise RuntimeError(f"{x}\n\nnot found in\n\n{self.uop}")
      ret.append(y)
    # create returned Tensors
    return [Tensor(u, device=t.device) for t,u in zip(targets, ret)]

  def backward(self, gradient:Tensor|None=None) -> Tensor:
    """
    Propagates the gradient of a tensor backwards through the computation graph.
    If the 'gradient' argument is not provided, the tensor must be a scalar, and the gradient is implicitly set to 1.0.
    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    t.sum().backward()
    print(t.grad.numpy())
    ```
    """
    all_uops = self.uop.toposort()
    tensors_need_grad: list[Tensor] = [t for tref in all_tensors if (t:=tref()) is not None and \
                                       t.uop in all_uops and t.requires_grad]
    # clear contexts
    for t,g in zip(tensors_need_grad, self.gradient(*tensors_need_grad, gradient=gradient, materialize_grads=True)):
      assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
      t.grad = g if t.grad is None else (t.grad + g)
    return self

  # ***** Tensor DNN, BLAS Operations (Level 1/0) *****
  def _apply_uop(self, f:Callable, *other:Tensor, extra_args=(), **kwargs) -> Tensor:
    out_uop: UOp = f(*[input.uop for input in (self,)+other], *extra_args, **kwargs)
    # if (metadata:=_METADATA.get()) is not None: all_metadata[new_uop] = (metadata,)
    needs_input_grad = [t.requires_grad for t in (self,)+other]
    requires_grad=True if any(needs_input_grad) else None if None in needs_input_grad else False
    out_tensor = Tensor(out_uop, device=out_uop.device, requires_grad=requires_grad)

    if os.getenv("EAGER", 0) == 1: # eager interpretation: materialize tensor
      # 1. allocate input and output bufs on device
      eval() # 2. dispatch op to eager interpreter
      return out_tensor
    elif os.getenv("GRAPH", 0) == 1: # lazy compilation: build the graph for .realize()
      # Pipeline: Tensor -> LazyData -> ScheduleItem -> ExecItem -> Codegen -> Source -> Binary -> Runtime  
      # -kernelize: graph rewrites
      # -schedule_with_vars: feeds graph to scheduler and memory planner
      # -realize: hands schedule to run_schedule
      return out_tensor

  def _apply_broadcasted_uop(self, f:Callable, other:Tensor|ConstType, reverse=False) -> Tensor:
    raise NotImplementedError("todo")
  
  def _binop(self, op, other, reverse): return self._apply_broadcasted_uop(lambda *u: UOp.alu(u[0], op, *u[1:]), other, reverse) # _binop is used by MathTrait

  # fa/mm
  def fa() -> Tensor:
    raise NotImplementedError("todo")
  
  def mm() -> Tensor:
    raise NotImplementedError("todo")

  # element ops
  # --activation
  def tanh(self) -> Tensor: return 2.0 * ((2.0 * self).sigmoid()) - 1.0
  def sigmoid(self) -> Tensor: return (1 + (self * (-1/math.log(2))).exp2()).reciprocal()

  # --unary
  def exp2(self) -> Tensor: return self._apply_uop(UOp.exp2) # self.cast(least_upper_float(self.dtype))._apply_uop(UOp.exp2)
  # reduce ops

  # movement ops
  # --high level: gather, cat, stack, repeat, split, chunk, unfold, squeeze, unsqueeze
  # --low level: view, reshape, expand, permute, flip, shrink, pad

  # ***** Tensor Constructors (Level -1) *****
  # --high level: .randn_like, randint, normal, uniform, scaled_uniform, kaiming_normal, randperm, multinomial

  # --low level:
  @staticmethod
  def empty(*shape, device:str|tuple[str, ...]|None=None, dtype:DTypeLike|None=None, **kwargs) -> Tensor:
    """
    Creates an empty tensor with the given shape.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.empty(2, 3)
    print(t.shape)
    ```
    """
    raise NotImplementedError("TODO")
  def empty_like(self, **kwargs) -> Tensor:
    """
    Creates an empty tensor with the same shape as `self`.
    If `dtype` is not specified, the dtype of `self` is used.
    """
    return Tensor.empty(self.shape, dtype=kwargs.pop("dtype", self.dtype), device=kwargs.pop("device", self.device), **kwargs)

  @staticmethod
  def rand(*shape, device:str|None=None, dtype:DTypeLike|None=None, contiguous:bool=True, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a uniform distribution over the interval `[0, 1)`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.rand(2, 3)
    print(t.numpy())
    ```
    """
    raise NotImplementedError("TODO")

  @staticmethod
  def full(shape:tuple[sint, ...], fill_value:ConstType, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with the given value.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.full((2, 3), 42).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.full((2, 3), False).numpy())
    ```
    """
    raise NotImplementedError("TODO")

  @staticmethod
  def zeros(*shape, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with zeros.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.zeros(2, 3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.zeros(2, 3, dtype=dtypes.int32).numpy())
    ```
    """
    raise NotImplementedError("TODO")

  @staticmethod
  def ones(*shape, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with ones.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(2, 3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(2, 3, dtype=dtypes.int32).numpy())
    ```
    """
    raise NotImplementedError("TODO")

  @staticmethod
  def arange(start, stop=None, step=1, **kwargs) -> Tensor:
    """
    Returns a 1-D tensor of size `ceil((stop - start) / step)` with values from `[start, stop)`, with spacing between values given by `step`.

    If `stop` is not specified, values are generated from `[0, start)` with the given `step`.

    If `stop` is specified, values are generated from `[start, stop)` with the given `step`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5, 10).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5, 10, 2).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5.5, 10, 2).numpy())
    ```
    """
    raise NotImplementedError("TODO")

  @staticmethod
  def linspace(start:int|float, stop:int|float, steps:int, **kwargs) -> Tensor:
    """
    Returns a 1-D tensor of `steps` evenly spaced values from `start` to `stop`, inclusive.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.linspace(0, 10, 5).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.linspace(-1, 1, 5).numpy())
    ```
    """
    raise NotImplementedError("TODO")

  @staticmethod
  def eye(n:int, m:int|None=None, **kwargs) -> Tensor:
    """
    Returns a 2-D tensor with `n` rows and `m` columns, with ones on the diagonal and zeros elsewhere.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.eye(3).numpy())
    ```

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.eye(2, 4).numpy())
    ```
    """
    raise NotImplementedError("TODO")

  # ***** Tensor Data (Level -1) *****  
  @property
  def ndim(self) -> int:
    """
    Returns the number of dimensions in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2], [3, 4]])
    print(t.ndim)
    ```
    """
    raise NotImplementedError("TODO")

  def numel(self) -> sint:
    """
    Returns the total number of elements in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(t.numel())
    ```
    """
    raise NotImplementedError("TODO")

  def element_size(self) -> int:
    """
    Returns the size in bytes of an individual element in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([5], dtype=dtypes.int16)
    print(t.element_size())
    ```
    """
    raise NotImplementedError("TODO")

  def nbytes(self) -> int:
    """
    Returns the total number of bytes of all elements in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([8, 9], dtype=dtypes.float)
    print(t.nbytes())
    ```
    """
    raise NotImplementedError("TODO")

  def is_floating_point(self) -> bool:
    """
    Returns `True` if the tensor contains floating point types, i.e. is one of `dtypes.float64`, `dtypes.float32`,
    `dtypes.float16`, `dtypes.bfloat16`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([8, 9], dtype=dtypes.float32)
    print(t.is_floating_point())
    ```
    """
    raise NotImplementedError("TODO")

  def size(self, dim:int|None=None) -> sint|tuple[sint, ...]:
    """
    Returns the size of the tensor. If `dim` is specified, return the length along dimension `dim`. Otherwise return the shape of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[4, 5, 6], [7, 8, 9]])
    print(t.size())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.size(dim=1))
    ```
    """
    raise NotImplementedError("TODO")

  def __init__(self):
    self.shape, self.stride = [], []

__all__ = ["Tensor"]