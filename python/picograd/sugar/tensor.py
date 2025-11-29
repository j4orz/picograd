# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
#         and https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py
from __future__ import annotations
from typing import Callable, Self, cast
import math, weakref

from picograd.dtype import ConstType, DType
from picograd.engine import sint, Op, OpCode, OpMixin, evaluator
from picograd.helpers import EAGER, GRAPH

all_tensors: dict[weakref.ref[Tensor], None] = {}

class Tensor(OpMixin):
  """
  the Tensor class is ndarray domain specific language dual to the heterogenous Runtime
  all tensor methods funnel through ._forward(), whose semantics depend on whether eager mode or graph mode is respectively set with EAGER=1 or GRAPH=1
    - with EAGER=1, ._forward() will build up the expression graph (for backprop) but perform kernel dispatch and tensor materialization immediately 
    - with GRAPH=1, ._forward() will lazily build up the graph and user's must initiate computation with a .realize() barrier (like torch_xla.sync())
  """

  # ************ Tensor Data + Constructors ************
  __slots__ = "op", "requires_grad", "grad"
  @property
  def device(self) -> str|tuple[str, ...]: return self.op.device
  @property
  def shape(self) -> tuple[sint, ...]: return self.op.shape
  @property
  def dtype(self) -> DType: return self.op.dtype
  @property
  def ndim(self) -> int: raise NotImplementedError("TODO")
  def data(self): raise NotImplementedError("todo")
  def item(self): raise NotImplementedError("todo")
  def to(self, device:str|tuple[str, ...]|None) -> Self: raise NotImplementedError("todo")
  def numel(self) -> sint: raise NotImplementedError("TODO")
  def element_size(self) -> int: raise NotImplementedError("TODO")
  def nbytes(self) -> int: raise NotImplementedError("TODO")
  def is_floating_point(self) -> bool: raise NotImplementedError("TODO")
  def size(self, dim:int|None=None) -> sint|tuple[sint, ...]: raise NotImplementedError("TODO")

  def __init__(self):
    self.shape, self.stride = [], []
  # --high level: .randn_like, randint, normal, uniform, scaled_uniform, kaiming_normal, randperm, multinomial
  # --low level:
  @staticmethod
  def empty(*shape, device:str|tuple[str, ...]|None=None, dtype:DTypeLike|None=None, **kwargs) -> Self: raise NotImplementedError("TODO")
  def empty_like(self, **kwargs) -> Self: return Tensor.empty(self.shape, dtype=kwargs.pop("dtype", self.dtype), device=kwargs.pop("device", self.device), **kwargs)
  @staticmethod
  def rand(*shape, device:str|None=None, dtype:DTypeLike|None=None, contiguous:bool=True, **kwargs) -> Self: raise NotImplementedError("TODO")
  @staticmethod
  def full(shape:tuple[sint, ...], fill_value:ConstType, **kwargs) -> Self: raise NotImplementedError("TODO")
  @staticmethod
  def zeros(*shape, **kwargs) -> Self: raise NotImplementedError("TODO")
  @staticmethod
  def ones(*shape, **kwargs) -> Self: raise NotImplementedError("TODO")
  @staticmethod
  def arange(start, stop=None, step=1, **kwargs) -> Self: raise NotImplementedError("TODO")
  @staticmethod
  def linspace(start:int|float, stop:int|float, steps:int, **kwargs) -> Self: raise NotImplementedError("TODO")
  @staticmethod
  def eye(n:int, m:int|None=None, **kwargs) -> Self: raise NotImplementedError("TODO")

  # ************ Tensor Operations ************
  # f'(x) backward
  def backward(self, grad:Tensor|None=None) -> Self:
    """
    backward performs by collecting tensors, computing gradients with automatic differentiation, and updating said tensors.
    """
    # 1. collect all tensors that requires grad by topologically sorting the graph of uops and filter
    all_uops = self.op.toposort()
    tensors_require_grad: list[Tensor] = [t for tref in all_tensors if (t:=tref()) is not None and t.uop in all_uops and t.requires_grad]
    uops_require_grad = [t.uop for t in tensors_require_grad]
    assert grad is not None or self.shape == tuple(), "when no gradient is provided, backward must be called on a scalar tensor"
    if not (self.is_floating_point() and all(t.is_floating_point() for t in tensors_require_grad)): raise RuntimeError("only float Tensors have gradient")
    
    # 2. compute the gradient with a map of tensors to partials
    if grad is None: grad = Tensor(1.0, dtype=self.dtype, device=self.device, requires_grad=False) # base case is 1.0
    tens2grads = Tensor._automatically_differentiate(self.op, grad.uop, set(uops_require_grad)) # skipping materializing zerod grads for now
    grads = [Tensor(g, device=t.device) for t,g in zip(tens2grads.keys, tens2grads.values)] # initialize tensor grads on device
    
    # 3. update the tensors that require grad with the gradient's partials
    for t,g in zip(tensors_require_grad, grads):
      assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
      t.grad = g if t.grad is None else (t.grad + g) # accumulate if t.grad exists
    return self
    
  @staticmethod
  def _automatically_differentiate(root:Op, root_grad:Op, targets:set[Op]) -> dict[Op, Op]:
    """
    _differentiate backpropagates partials on a topologically sorted expression graph with the chain rule
    and produces the gradient in the form of a map of ops to their partials (which, in turn, are ops)
    """
    tens2grads = {root: root_grad}

    # 1. topological sort ordering is NP-complete?????????? <--------------------------- MOOOOOOOOOOOOOOOOOOSEEEEEE
    in_target_path: dict[Op, bool] = {}
    for u in root.toposort(): in_target_path[u] = any(x in targets or in_target_path[x] for x in u.inputs)
    dfs = list(root.toposort()) # lambda node: node.op not in {OpCode.DETACH, OpCode.ASSIGN} and in_target_path[node])) # don't flow through DETACH/ASSIGN or anything not in target path

    # 2. backpropagation with the chain rule
    for tensor in reversed(dfs):
      if tensor not in tens2grads: continue

      local_grads: tuple[Op|None, ...]|None = cast(tuple[Op, ...]|None, chain_rules.rewrite(tensor, ctx=tens2grads[tensor]))
      if local_grads is None: raise RuntimeError(f"failed to compute gradient for {tensor.op}\n\nin {str(tensor)[0:1000]}...")
      assert len(local_grads) == len(tensor.inputs), f"got {len(local_grads)} gradient, expected {len(tensor.inputs)}"

      for tensor,local_grad in zip(tensor.inputs, local_grads): # <--------------------- MOOOSE: why are we accumulating inside ad()? don't we do it in backward()??
        if local_grad is None: continue
        if tensor in tens2grads: tens2grads[tensor] = tens2grads[tensor] + local_grad # accumulate if tensor exists
        else: tens2grads[tensor] = local_grad # o/w initialize

        # if len(forward_metadata:=all_metadata.get(tensor, ())):
        #   backward_metadata = tuple(dataclasses.replace(x, backward=True) for x in forward_metadata)
        #   # we add the backward metadata to everything new in the graph
        #   for bw_uop in local_grad.toposort(lambda x: x not in (tensor, *tensor.src, tens2grads[tensor])):
        #     all_metadata[bw_uop] = all_metadata.get(bw_uop, ())+backward_metadata

    return tens2grads
  
  def forward(self, f:Callable, *other:Tensor) -> Self:
    """
    .forward() is the method in which all evaluations funnel through, regardless of whether the operation is either
      - sugar: directly calls .forward() or
      - primitive: indirectly calls .forward() by _binop override
    in either case, .forward() evaluates f(x) by calling f, implemented by Op.eval(), which in turn, launches device kernels.
    .forward() then wraps the new output Op in the expression graph with a Tensor handle
    """
    output_op: Op = f(*[t.uop for t in (self,)+other]) # <------------ compiler pipeline: if EAGER ... elif GRAPH ... else ...
    needs_input_grad = [t.requires_grad for t in (self,)+other]
    requires_grad=True if any(needs_input_grad) else None if None in needs_input_grad else False
    output_tensor = Tensor(output_op, device=output_op.device, requires_grad=requires_grad)
    return output_tensor
  
  def recip(self) -> Self: return self.forward(Op.recip)
  def sigmoid(self) -> Self: return (1 + (self * (-1/math.log(2))).exp2()).recip()
  def tanh(self) -> Self: return 2.0 * ((2.0 * self).sigmoid()) - 1.0
  def exp2(self) -> Self: return self.forward(Op.exp2)
  def log2(self) -> Self: return self.forward(Op.log2)
  def fa(self) -> Self: raise NotImplementedError("todo")
  def mm(self) -> Self: raise NotImplementedError("todo")
  # the builtin primitives (i.e add) which do not need to be desugared will call provided OpMixin._binop() to __________
  # which is overrided to ____________
  def _binop(self, op, x, reverse):
    return self._apply_broadcasted_uop(lambda *u: Op.alu(u[0], op, *u[1:]), x, reverse) # _binop is used by MathTrait
  def _apply_broadcasted_uop(self, fxn:Callable, x:Tensor|ConstType, reverse=False) -> Self:
    lhs,rhs = self._broadcasted(x, reverse)
    return lhs.forward(fxn, rhs)

  # reduce ops
  # movement ops
  # --high level: gather, cat, stack, repeat, split, chunk, unfold, squeeze, unsqueeze
  # --low level: view, reshape, expand, permute, flip, shrink, pad

__all__ = ["Tensor"]