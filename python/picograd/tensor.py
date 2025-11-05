from __future__ import annotations
import math, os
from typing import Callable
from picograd.engine import evaluator
from picograd.uop import UOp
from picograd.mixins import ComputeMixin
# from . import _pgrs

class Tensor(ComputeMixin): # , MovementMixin):
  """
  picograd follows (tf.1/pt1) which takes the numpy ndarray and adds
    - gpu acceleration to forward kernels .forward()
    - and automatic differentiation with backward kernels with .backward()
  like pytorch1, picograd's forward passes decompose (desugar) tensor methods into a more primitive set of UOps (from tinygrad)
                    and the backward pass is implemented with a dynamic tape (as opposed to a source to source transform like jax)
  
  picograd's forward passes follow the architecture of classic numerical computing
  which separate concerns into levels 1,2,3 like LAPACK/BLAS
    2. (DNN)   provides high level mathematical primitives (mathematician)
  0/1. (BLAS)  provides performance primitives (perf engineer)
   -1: (DATA)  provides the foundational multi-dimensional array data structure (compiler engineer)
      the semantics of k in level k has *three* meanings in of itself
        1. publish order of BLAS
        2. algorithmic complexity in naive implementation (n -> n^k)
        3. ptoential of data reuse, loop fusion, parallelism
        for more details refer to (Dodson, Lewis 1985) https://dl.acm.org/doi/pdf/10.1145/1057935.1057937
                         and https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Functionality

  picograd also follows (pt2/jax/tinygrad) which modify the tensor language implementation strategy
  from eager interpretation to just-in-time/lazy compilation. this is coloquially known as "GRAPH" mode.
  """

  def backward(self, gradient:Tensor|None=None) -> Tensor: raise NotImplementedError("todo")
  def _forward(self, f:Callable, *other:Tensor) -> Tensor: #extra_args=(), **kwargs)
    if os.getenv("EAGER") == 1:
      out_tensor = evaluator.eval_uop([self, other], out_uop)
      return out_tensor
    elif os.getenv("GRAPH") == 1: # lazy compilation: build the graph for .realize()
      # if (metadata:=_METADATA.get()) is not None: all_metadata[new_uop] = (metadata,)
      # needs_input_grad = [t.requires_grad for t in (self,)+other]
      # requires_grad=True if any(needs_input_grad) else None if None in needs_input_grad else False
      out_uop: UOp = f(*[input.uop for input in (self,)+other]) # *extra_args, **kwargs)
      out_tensor = Tensor(out_uop, device=out_uop.device) #, requires_grad=requires_grad)
      raise NotImplementedError("todo")
      # -kernelize: graph rewrites
      # -schedule_with_vars: feeds graph to scheduler and memory planner
      # -realize: hands schedule to run_schedule
      # return out_tensor

  def _apply_broadcasted_uop(self, f:Callable, other:Tensor|ConstType, reverse=False) -> Tensor: raise NotImplementedError("todo")  
  def _binop(self, op, other, reverse): return self._apply_broadcasted_uop(lambda *u: UOp.alu(u[0], op, *u[1:]), other, reverse) # _binop is used by MathTrait


  # ***** Tensor DNN (Level 2) *****
  def backward_numerical(self) -> Tensor: raise NotImplementedError("")
  def backward_symbolic(self) -> Tensor: raise NotImplementedError("")
  def backward_automatic(self) -> Tensor: raise NotImplementedError("")
  def fa() -> Tensor:
    raise NotImplementedError("todo")

  




  # ***** Tensor BLAS Operations (Level 1/0) *****
  def mm() -> Tensor:
    raise NotImplementedError("todo")

  # --activation
  def tanh(self) -> Tensor: return 2.0 * ((2.0 * self).sigmoid()) - 1.0
  def sigmoid(self) -> Tensor: return (1 + (self * (-1/math.log(2))).exp2()).reciprocal()

  # --unary
  def exp2(self) -> Tensor: return self._forward(UOp.exp2) # self.cast(least_upper_float(self.dtype))._apply_uop(UOp.exp2)

  # C. reduce ops

  # D. movement ops
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
  def data(self): raise NotImplementedError("todo")
  def item(self): raise NotImplementedError("todo")
  def to(self, device:str|tuple[str, ...]|None) -> Tensor: raise NotImplementedError("todo")

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
