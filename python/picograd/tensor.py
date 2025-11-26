# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from __future__ import annotations
import math, os, weakref
from typing import Callable, cast
from picograd.dtype import ConstType, DType, DTypeLike
from picograd.engine import evaluator
from picograd.op import sint, Op, OpCode, Pattern, PatternMatcher
# from picograd.mixins import ComputeMixin
# from . import _pgrs

all_tensors: dict[weakref.ref[Tensor], None] = {}

chain_rules = PatternMatcher([
  # (Pat(OpCode.CAST, name="ret"), lambda ctx, ret: (ctx.cast(ret.src[0].dtype),)),
  (Pattern(OpCode.RECIPROCAL, name="input"), lambda output_grad, input: (-output_grad * input * input,)),
  (Pattern(OpCode.SIN, name="input"), lambda output_grad, input: ((math.pi/2 - input.src[0]).sin() * output_grad,)),
  (Pattern(OpCode.LOG2, name="input"), lambda output_grad, input: (output_grad / (input.src[0] * math.log(2)),)),
  (Pattern(OpCode.EXP2, name="input"), lambda output_grad, input: (input * output_grad * math.log(2),)),
  (Pattern(OpCode.SQRT, name="input"), lambda output_grad, input: (output_grad / (input*2),)),
  # (Pat((OpCode.CMPLT, OpCode.CMPNE)), lambda: (None, None)),
  (Pattern(OpCode.ADD), lambda output_grad: (1.0*output_grad, 1.0*output_grad)),
  # (Pat(OpCode.POW, name="input", src=(Pat.var("b"), Pat.var("e"))), lambda output_grad, input, b, e:
  #   (output_grad * (b.eq(0)&e.eq(0)).where(e, e*b.pow(e-1)), output_grad * b.eq(0).where((e<0).where(input.const_like(-math.inf), 0), input*b.log2()*math.log(2.0)))),
  # (Pat(OpCode.MAX, src=(Pat.var("x"), Pat.var("y"))), lambda output_grad, x, y:
  #   ((x>y).where(output_grad, (x.eq(y)).where(output_grad * 0.5, 0)), (x<y).where(output_grad, (x.eq(y)).where(output_grad * 0.5, 0)))),
  (Pattern(OpCode.MUL, name="input"), lambda output_grad, input: (input.src[1]*output_grad, input.src[0]*output_grad)),
  # (Patttern(OpCode.WHERE, name="input"), lambda output_grad, input: (None, input.src[0].where(output_grad, output_grad.const_like(0)), input.src[0].where(output_grad.const_like(0), output_grad))),
  # (Patttern(OpCode.REDUCE_AXIS, name="input"), reduce_gradient),
  # (Patttern(OpCode.CONTIGUOUS), lambda output_grad: (output_grad,)),
  # (Patttern(OpCode.CONTIGUOUS_BACKWARD), lambda output_grad: (output_grad.contiguous(),)),
  # (Patttern(OpCode.RESHAPE, name="input"), lambda output_grad, input: (output_grad.reshape(input.src[0].shape), None)),
  # (Patttern(OpCode.EXPAND, name="input"), lambda output_grad, input: (output_grad.r(OpCode.ADD,tuple(i for i,(s,n) in enumerate(zip(input.src[0].shape, input.shape)) if s!=n)), None)),
  # (Patttern(OpCode.PAD, name="input"), lambda output_grad, input: (output_grad.shrink(tuple([(p[0], s+p[0]) for s,p in zip(input.src[0].shape, input.marg)])), None, None)),
  # (Patttern(OpCode.SHRINK, name="input"), lambda output_grad, input: (output_grad.pad(tuple([(p[0], s-p[1]) for s,p in zip(input.src[0].shape, input.marg)])), None, None)),
  # (Patttern(OpCode.PERMUTE, name="input"), lambda output_grad, input: (output_grad.permute(argsort(input.marg)),)),
  # (Patttern(OpCode.FLIP, name="input"), lambda output_grad, input: (output_grad.flip(input.marg),)),
  # (Patttern(OpCode.MULTI, name="input"), lambda output_grad, input: output_grad.shard(input.device, input.axis).src),
  # # NOTE: this is only correct when the KERNEL has a single output
  # (Patttern(OpCode.AFTER), lambda output_grad: (output_grad, output_grad)),
  # (Patttern(OpCode.KERNEL, name="k"), lambda output_grad, k: k.arg.grad_fxn(output_grad, k)),
  # # there's no gradient for bitcast
  # (Patttern(OpCode.BITCAST), lambda: (None,)),
])

class Tensor(): # todo: compute/movement mixins #(ComputeMixin): # , MovementMixin):
  """
  the Tensor class is ndarray domain specific language dual to the heterogenous Runtime
  all tensor methods funnel through ._forward(), whose semantics depend on whether eager mode or graph mode is respectively set with EAGER=1 or GRAPH=1
    - with EAGER=1, ._forward() will build up the expression graph (for backprop) but perform kernel dispatch and tensor materialization immediately 
    - with GRAPH=1, ._forward() will lazily build up the graph and user's must initiate computation with a .realize() barrier (like torch_xla.sync())
  """

  __slots__ = "uop", "requires_grad", "grad"
  def __init__(self):
    # MOOOSE: allocate *data* for evaluator to *operate* on
    self.shape, self.stride = [], []


  # ************ Tensor Data ************
  def data(self): raise NotImplementedError("todo")
  def item(self): raise NotImplementedError("todo")
  def to(self, device:str|tuple[str, ...]|None) -> Tensor: raise NotImplementedError("todo")
  def numel(self) -> sint: raise NotImplementedError("TODO")
  def element_size(self) -> int: raise NotImplementedError("TODO")
  def nbytes(self) -> int: raise NotImplementedError("TODO")
  def is_floating_point(self) -> bool: raise NotImplementedError("TODO")
  def size(self, dim:int|None=None) -> sint|tuple[sint, ...]: raise NotImplementedError("TODO")

  # runtime data like device, shape, and dtype are deleted to uop, not tensor
  @property
  def device(self) -> str|tuple[str, ...]: return self.uop.device
  @property
  def shape(self) -> tuple[sint, ...]: return self.uop.shape
  @property
  def dtype(self) -> DType: return self.uop.dtype
  @property
  def ndim(self) -> int: raise NotImplementedError("TODO")

  
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




  # ************ Tensor Forward and Backward ************
  def backward(self, grad:Tensor|None=None) -> Tensor:
    """
    backward performs by collecting tensors, computing gradients with automatic differentiation, and updating said tensors.
    """
    # 1. collect all tensors that requires grad by topologically sorting the graph of uops and filter
    all_uops = self.uop.toposort()
    tensors_require_grad: list[Tensor] = [t for tref in all_tensors if (t:=tref()) is not None and t.uop in all_uops and t.requires_grad]
    uops_require_grad = [t.uop for t in tensors_require_grad]
    assert grad is not None or self.shape == tuple(), "when no gradient is provided, backward must be called on a scalar tensor"
    if not (self.is_floating_point() and all(t.is_floating_point() for t in tensors_require_grad)): raise RuntimeError("only float Tensors have gradient")
    
    # 2. compute the gradient with a map of tensors to partials
    if grad is None: grad = Tensor(1.0, dtype=self.dtype, device=self.device, requires_grad=False) # base case is 1.0
    tens2grads = Tensor._automatically_differentiate(self.uop, grad.uop, set(uops_require_grad)) # skipping materializing zerod grads for now
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
    for u in root.toposort(): in_target_path[u] = any(x in targets or in_target_path[x] for x in u.src)
    dfs = list(root.toposort()) # lambda node: node.op not in {OpCode.DETACH, OpCode.ASSIGN} and in_target_path[node])) # don't flow through DETACH/ASSIGN or anything not in target path

    # 2. backpropagation with the chain rule
    for tensor in reversed(dfs):
      if tensor not in tens2grads: continue

      local_grads: tuple[Op|None, ...]|None = cast(tuple[Op, ...]|None, chain_rules.rewrite(tensor, ctx=tens2grads[tensor]))
      if local_grads is None: raise RuntimeError(f"failed to compute gradient for {tensor.op}\n\nin {str(tensor)[0:1000]}...")
      assert len(local_grads) == len(tensor.src), f"got {len(local_grads)} gradient, expected {len(tensor.src)}"

      for tensor,local_grad in zip(tensor.src, local_grads): # <--------------------- MOOOSE: why are we accumulating inside ad()? don't we do it in backward()??
        if local_grad is None: continue
        if tensor in tens2grads: tens2grads[tensor] = tens2grads[tensor] + local_grad # accumulate if tensor exists
        else: tens2grads[tensor] = local_grad # o/w initialize

        # if len(forward_metadata:=all_metadata.get(tensor, ())):
        #   backward_metadata = tuple(dataclasses.replace(x, backward=True) for x in forward_metadata)
        #   # we add the backward metadata to everything new in the graph
        #   for bw_uop in local_grad.toposort(lambda x: x not in (tensor, *tensor.src, tens2grads[tensor])):
        #     all_metadata[bw_uop] = all_metadata.get(bw_uop, ())+backward_metadata

    return tens2grads

  def _binop(self, op, x, reverse):
    return self._apply_broadcasted_uop(lambda *u: Op.alu(u[0], op, *u[1:]), x, reverse) # _binop is used by MathTrait
  def _apply_broadcasted_uop(self, fxn:Callable, x:Tensor|ConstType, reverse=False) -> Tensor:
    lhs,rhs = self._broadcasted(x, reverse)
    return lhs._apply_uop(fxn, rhs)
  def _forward(self, f:Callable, *other:Tensor) -> Tensor: #extra_args=(), **kwargs)
    if os.getenv("EAGER") == 1:
      out_tensor = evaluator.eval_uop([self, other], out_uop)
      return out_tensor
    elif os.getenv("GRAPH") == 1: # lazy compilation: build the graph for .realize()
      out_uop: Op = f(*[t.uop for t in (self,)+other])#, *extra_args, **kwargs)
      # if (metadata:=_METADATA.get()) is not None: all_metadata[out_uop] = (metadata,)
      needs_input_grad = [t.requires_grad for t in (self,)+other]
      return Tensor(out_uop, device=out_uop.device, requires_grad=True if any(needs_input_grad) else None if None in needs_input_grad else False)
    
      # defer execution and build up graph of ops for compilation later on .realize()
      # -kernelize: graph rewrites
      # -schedule_with_vars: feeds graph to scheduler and memory planner
      # -realize: hands schedule to run_schedule
      



  # ************ Tensor Operations ************
  def fa(self) -> Tensor: return self._forward(Op.FA)
  def mm(self) -> Tensor: return self._forward(Op.MM)
  # --activation
  def tanh(self) -> Tensor: return 2.0 * ((2.0 * self).sigmoid()) - 1.0
  def sigmoid(self) -> Tensor: return (1 + (self * (-1/math.log(2))).exp2()).reciprocal()
  # --unary
  def exp2(self) -> Tensor: return self._forward(Op.exp2) # self.cast(least_upper_float(self.dtype))._apply_uop(UOp.exp2)

  # reduce ops
  # movement ops
  # --high level: gather, cat, stack, repeat, split, chunk, unfold, squeeze, unsqueeze
  # --low level: view, reshape, expand, permute, flip, shrink, pad

__all__ = ["Tensor"]
