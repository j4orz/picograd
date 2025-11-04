from __future__ import annotations
import math, os
from typing import Callable
from picograd.uop import UOp
from picograd.mixins import ComputeMixin
# from . import _pgrs

class Tensor(ComputeMixin): # , MovementMixin):
  """
  picograd's tensor frontend follows the architecture of classic numerical computing like numpy
  for more details, see: (Dodson, Lewis 1985) https://dl.acm.org/doi/pdf/10.1145/1057935.1057937
  - Level  2:     (BCKWD)   provides the high level mathematical primitives of compilation, automatic differentiation, gradient descent
  - Level  0/1:   (FWD)     provides performance primitives that require the knowledge of microarchitecture to obtain peak theoretical throughput (FLOP/S)
  - Level -1:     (DATA)    provides the foundational multi-dimensional array data structure

  picograd has support for both interpretation and compilation, what is coloquially known as "eager" and "graph" mode.
  with EAGER=1, all methods use the picograd runtime directly and interpret Uop's eagerly
  with GRAPH=1, all methods lazily build up a global view of the neural network with a graph of uops, compiling on .realize()
  """

  # ***** Tensor Compile, AD, GD (Level 2) *****
  def gradient(self, *targets:Tensor, gradient:Tensor|None=None, materialize_grads=False) -> list[Tensor]:
    raise NotImplementedError("todo")

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
      in_bufs = [input._buffer for input in (self,)+other]
      shp, dtype = out_tensor.shape, out_tensor.dtype.base
      out_uop = UOp.new_buffer(dev, prod(shp), dtype).reshape(shp)
      out_buf = cast(Buffer, out_uop.base.buffer).allocate()
      eval() # 2. dispatch op to eager interpreter
      return out_tensor
    elif os.getenv("GRAPH", 0) == 1: # lazy compilation: build the graph for .realize()
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
  def empty(*shape, device:str|tuple[str, ...]|None=None, dtype:DTypeLike|None=None, **kwargs) -> Tensor: raise NotImplementedError("TODO")
  def empty_like(self, **kwargs) -> Tensor: return Tensor.empty(self.shape, dtype=kwargs.pop("dtype", self.dtype), device=kwargs.pop("device", self.device), **kwargs)
  @staticmethod
  def rand(*shape, device:str|None=None, dtype:DTypeLike|None=None, contiguous:bool=True, **kwargs) -> Tensor: raise NotImplementedError("TODO")
  @staticmethod
  def full(shape:tuple[sint, ...], fill_value:ConstType, **kwargs) -> Tensor: raise NotImplementedError("TODO")
  @staticmethod
  def zeros(*shape, **kwargs) -> Tensor: raise NotImplementedError("TODO")
  @staticmethod
  def ones(*shape, **kwargs) -> Tensor: raise NotImplementedError("TODO")
  @staticmethod
  def arange(start, stop=None, step=1, **kwargs) -> Tensor: raise NotImplementedError("TODO")
  @staticmethod
  def linspace(start:int|float, stop:int|float, steps:int, **kwargs) -> Tensor: raise NotImplementedError("TODO")
  @staticmethod
  def eye(n:int, m:int|None=None, **kwargs) -> Tensor: raise NotImplementedError("TODO")

  # ***** Tensor Data (Level -1) *****
  @property
  def device(self) -> str|tuple[str, ...]: return self.uop.device
  @property
  def shape(self) -> tuple[sint, ...]: return self.uop.shape
  @property
  def dtype(self) -> DType: return self.uop.dtype

  @property
  def ndim(self) -> int: raise NotImplementedError("TODO")
  def numel(self) -> sint: raise NotImplementedError("TODO")
  def element_size(self) -> int: raise NotImplementedError("TODO")
  def nbytes(self) -> int: raise NotImplementedError("TODO")
  def is_floating_point(self) -> bool: raise NotImplementedError("TODO")
  def size(self, dim:int|None=None) -> sint|tuple[sint, ...]: raise NotImplementedError("TODO")
  def __init__(self):
    self.shape, self.stride = [], []
  __slots__ = "uop", "requires_grad", "grad"  # runtime data like device, shape, and dtype are deleted to uop, not tensor

__all__ = ["Tensor"]
