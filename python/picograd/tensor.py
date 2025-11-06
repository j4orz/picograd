from __future__ import annotations
import math, os
from typing import Callable
from picograd import op
from picograd.engine import evaluator
from picograd.op import Op, OpCode, PatternMatcher, Pat
from picograd.mixins import ComputeMixin
# from . import _pgrs

chain_rules = PatternMatcher([
  # (Pat(OpCode.CAST, name="ret"), lambda ctx, ret: (ctx.cast(ret.src[0].dtype),)),
  (Pat(OpCode.RECIPROCAL, name="input"), lambda output_grad, input: (-output_grad * input * input,)),
  (Pat(OpCode.SIN, name="input"), lambda output_grad, input: ((math.pi/2 - input.src[0]).sin() * output_grad,)),
  (Pat(OpCode.LOG2, name="input"), lambda output_grad, input: (output_grad / (input.src[0] * math.log(2)),)),
  (Pat(OpCode.EXP2, name="input"), lambda output_grad, input: (input * output_grad * math.log(2),)),
  (Pat(OpCode.SQRT, name="input"), lambda output_grad, input: (output_grad / (input*2),)),
  # (Pat((OpCode.CMPLT, OpCode.CMPNE)), lambda: (None, None)),
  (Pat(OpCode.ADD), lambda output_grad: (1.0*output_grad, 1.0*output_grad)),
  # (Pat(OpCode.POW, name="input", src=(Pat.var("b"), Pat.var("e"))), lambda output_grad, input, b, e:
  #   (output_grad * (b.eq(0)&e.eq(0)).where(e, e*b.pow(e-1)), output_grad * b.eq(0).where((e<0).where(input.const_like(-math.inf), 0), input*b.log2()*math.log(2.0)))),
  # (Pat(OpCode.MAX, src=(Pat.var("x"), Pat.var("y"))), lambda output_grad, x, y:
  #   ((x>y).where(output_grad, (x.eq(y)).where(output_grad * 0.5, 0)), (x<y).where(output_grad, (x.eq(y)).where(output_grad * 0.5, 0)))),
  (Pat(OpCode.MUL, name="input"), lambda output_grad, input: (input.src[1]*output_grad, input.src[0]*output_grad)),
  # (UPat(OpCode.WHERE, name="input"), lambda output_grad, input: (None, input.src[0].where(output_grad, output_grad.const_like(0)), input.src[0].where(output_grad.const_like(0), output_grad))),
  # (UPat(OpCode.REDUCE_AXIS, name="input"), reduce_gradient),
  # (UPat(OpCode.CONTIGUOUS), lambda output_grad: (output_grad,)),
  # (UPat(OpCode.CONTIGUOUS_BACKWARD), lambda output_grad: (output_grad.contiguous(),)),
  # (UPat(OpCode.RESHAPE, name="input"), lambda output_grad, input: (output_grad.reshape(input.src[0].shape), None)),
  # (UPat(OpCode.EXPAND, name="input"), lambda output_grad, input: (output_grad.r(OpCode.ADD,tuple(i for i,(s,n) in enumerate(zip(input.src[0].shape, input.shape)) if s!=n)), None)),
  # (UPat(OpCode.PAD, name="input"), lambda output_grad, input: (output_grad.shrink(tuple([(p[0], s+p[0]) for s,p in zip(input.src[0].shape, input.marg)])), None, None)),
  # (UPat(OpCode.SHRINK, name="input"), lambda output_grad, input: (output_grad.pad(tuple([(p[0], s-p[1]) for s,p in zip(input.src[0].shape, input.marg)])), None, None)),
  # (UPat(OpCode.PERMUTE, name="input"), lambda output_grad, input: (output_grad.permute(argsort(input.marg)),)),
  # (UPat(OpCode.FLIP, name="input"), lambda output_grad, input: (output_grad.flip(input.marg),)),
  # (UPat(OpCode.MULTI, name="input"), lambda output_grad, input: output_grad.shard(input.device, input.axis).src),
  # # NOTE: this is only correct when the KERNEL has a single output
  # (UPat(OpCode.AFTER), lambda output_grad: (output_grad, output_grad)),
  # (UPat(OpCode.KERNEL, name="k"), lambda output_grad, k: k.arg.grad_fxn(output_grad, k)),
  # # there's no gradient for bitcast
  # (UPat(OpCode.BITCAST), lambda: (None,)),
])

class Tensor(ComputeMixin): # , MovementMixin):
  """
  picograd follows (tf.1/pt1) which takes the numpy ndarray and adds
    - gpu acceleration to forward kernels with .forward()
    - and automatic differentiation with backward kernels with .backward()
  like pytorch1, picograd's forward passes decompose (desugar) tensor methods into a more primitive set of UOps (from tinygrad)
                    and the backward pass is implemented with a dynamic tape
                    (as opposed to a just-in-time source to source transform like jax)
  
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
  from eager interpretation to just-in-time/lazy compilation; this is coloquially known as "EAGER" and "GRAPH" mode, respectively
  all tensor methods funnel through ._forward(), whose semantics depend on whether eager mode is set with EAGER=1 or graph mode is set with GRAPH=1

  MOOSE
  with EAGER=1, ._forward() will dispatch a kernel immediately and save the graph...tape...
  with GRAPH=1, ._forward() will lazily build up the graph...tape...for a compilation on .realize()
  tensor with ComputeMixin and MovementMixin...

  TODO: strided layout(np/pt) vs scheduled layout(halide/tvm)
  """

  def backward(self, grad:Tensor|None=None) -> Tensor:
    """
    backward performs backpropagation by collecting tensors, computing gradients, and updating said tensors.
    the heavy lifting is done with Tensor.eval_grad()
    """
    # 1. collect all tensors that requires grad by topologically sorting the graph of uops and filter
    all_uops = self.uop.toposort() # < ---------------------MOOSE: assumes the graph is pointed to by self.uop
    tensors_require_grad: list[Tensor] = [t for tref in all_tensors if (t:=tref()) is not None and t.uop in all_uops and t.requires_grad]
    uops_require_grad = [t.uop for t in tensors_require_grad]
    assert grad is not None or self.shape == tuple(), "when no gradient is provided, backward must be called on a scalar tensor"
    if not (self.is_floating_point() and all(t.is_floating_point() for t in tensors_require_grad)): raise RuntimeError("only float Tensors have gradient")
    
    # 2. compute the gradient with a map of tensors to partials
    if grad is None: grad = Tensor(1.0, dtype=self.dtype, device=self.device, requires_grad=False) # base case is 1.0
    tens2grads = Tensor._eval_grad(self.uop, grad.uop, set(uops_require_grad)) # skipping materializing for now
    grads = [Tensor(g, device=t.device) for t,g in zip(tens2grads.keys, tens2grads.values)] # initialize tensor grads on device
    
    # 3. update the tensors that require grad with the gradient's partials
    for t,g in zip(tensors_require_grad, grads):
      assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
      t.grad = g if t.grad is None else (t.grad + g) # accumulate if t.grad exists
    return self      





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
  def exp2(self) -> Tensor: return self._forward(Op.exp2) # self.cast(least_upper_float(self.dtype))._apply_uop(UOp.exp2)

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
