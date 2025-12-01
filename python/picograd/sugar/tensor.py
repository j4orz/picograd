# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
#         and https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py
from __future__ import annotations
import functools
import operator
from typing import Callable, Iterable, Self, TypeVar, cast, get_args
import math, weakref, struct, pathlib

from picograd.engine import sint, OpNode, OpCode, OpMixin, evaluator
from picograd.helpers import EAGER, GRAPH
from picograd.runtime import Device
from picograd.dtype import ConstType, DType, DTypeLike, dtypes, truncate

all_tensors: dict[weakref.ref[Tensor], None] = {}
def fully_flatten(l):
  if hasattr(l, "__len__") and hasattr(l, "__getitem__") and not isinstance(l, str):
    if hasattr(l, "shape") and l.shape == (): return [l[()]]
    flattened = []
    for li in l: flattened.extend(fully_flatten(li))
    return flattened
  return [l]

def get_shape(x) -> tuple[int, ...]:
  # NOTE: str is special because __getitem__ on a str is still a str
  if not hasattr(x, "__len__") or not hasattr(x, "__getitem__") or isinstance(x, str) or (hasattr(x, "shape") and x.shape == ()): return ()
  if not all_same(subs:=[get_shape(xi) for xi in x]): raise ValueError(f"inhomogeneous shape from {x}")
  return (len(subs),) + (subs[0] if subs else ())

T = TypeVar("T")
U = TypeVar("U")
def prod(x:Iterable[T]) -> T|int: return functools.reduce(operator.mul, x, 1) # NOTE: it returns int 1 if x is empty regardless of the type of x

def argfix(*x):
  if x and x[0].__class__ in (tuple, list):
    if len(x) != 1: raise ValueError(f"bad arg {x}")
    return tuple(x[0])
  return x

class Tensor(OpMixin):
  """
  the Tensor class is a *sugared handle* to the expression graph of vertices V=Set<OpNode> and edges = Set<(OpNode,OpNode)>,
  which represents picograd's primitive understanding (intermediate representation) of the specified expression f(x)
  the data and functionality you expect to live on Tensor actually lives in OpNode because the Tensor class is actually a sugared handle
  """

  # ************ Tensor Data + Constructors ************
  __slots__ = "op", "grad", "requires_grad", 
  @property
  def device(self) -> str|tuple[str, ...]: return self.op.device
  @property
  def shape(self) -> tuple[sint, ...]: return self.op.shape
  @property
  def dtype(self) -> DType: return self.op.dtype
  @property
  def ndim(self) -> int: raise NotImplementedError("TODO")

  def data(self) -> memoryview:
    if 0 in self.shape: return memoryview(bytearray(0)).cast(self.dtype.base.fmt)
    assert all_int(self.shape), f"no data if shape is symbolic, {self.shape=}"
    return self._buffer().as_typed_buffer(self.shape)

  def item(self) -> ConstType:
    assert self.numel() == 1, "must have one element for item"
    return self.data()[(0,) * len(self.shape)]
  
  @staticmethod
  def zeros(*shape, **kwargs) -> Self: return Tensor.full(argfix(*shape), 0.0)
  @staticmethod
  def ones(*shape, **kwargs) -> Self: return Tensor.full(argfix(*shape), 1.0)
  @staticmethod
  def full(shape:tuple[sint, ...], fill_value: ConstType) -> Self:
    """
    full 
    """
    new_shape = (1, )*len(argfix(shape))
    return Tensor(fill_value, _force_unique=True).reshape(new_shape).expand(new_shape)
  
  def __init__(self,
               input:ConstType|bytes|list|tuple|OpNode|None,  # removed support for 'np.ndarray'|pathlib.Path|
               device:str|tuple|list|None=None, dtype:DTypeLike|None=None, requires_grad:bool|None=None, _force_unique:bool=False):
    """
    Tensor.__init__() constructs the OpNode for the Tensor's given device and dtype
    OpNode.const      symbolic const? ==> actually realize
    OpNode.buffer     fake realize  ==> actually realize
    """
    if device is None and isinstance(input, pathlib.Path): device = f"DISK:{input.resolve()}"  # keep it on the disk if device is None
    _dtype: DType|None = to_dtype(dtype) if dtype is not None else None
    _device: str|tuple[str, ...] = tuple(Device.canonicalize(x) for x in device) if isinstance(device, (tuple, list)) else Device.canonicalize(device)
    del device, dtype
    self.grad: Tensor|None = None                                 # tensors can have gradients if you have called .backward
    self.requires_grad: bool|None = requires_grad                 # NOTE: this can be in three states. False and None: no gradient, True: gradient. None (the default) will be updated to True if it's put in an optimizer

    if isinstance(input, OpNode):                   raise NotImplementedError("todo")
    elif input is None:                             opnode = OpNode.const(_dtype or dtypes.default_float, 0, _device, (), unique=_force_unique)                   # <= None
    elif isinstance(input, get_args(ConstType)):    opnode = OpNode.const(_dtype or dtypes.from_py(input), input, _device, (), unique=_force_unique)              # <= const_type
    elif isinstance(input, bytes):                  raise NotImplementedError("todo") # data = Tensor._frompy(data, dtypes.uint8 if _dtype is None else _dtype)   # <= bytes     
    elif isinstance(input, (list, tuple)):                                                                                                                        # <= list/tuple
      if _dtype is None:
        if (d := fully_flatten(input)) and all(isinstance(s, bool) for s in d): _dtype = dtypes.bool
        else:                                                                   _dtype = dtypes.default_int if d and all_int(d) else dtypes.default_float # NOTE: this works because all_int([True, False]) is True

      if _dtype in [dtypes.bfloat16, *dtypes.fp8s]: opnode = Tensor(Tensor._frompy(input, dtypes.float32), device=_device).cast(_dtype).uop
      else:                                         opnode = Tensor._frompy(input, _dtype)

    if not isinstance(input, OpNode): raise RuntimeError(f"can't create Tensor from {input!r} with type {type(input)}") # by this point it has to be a UOp
    if isinstance(_device, str): self.op: OpNode = opnode if input.device == _device else input.copy_to_device(_device) # data might be on a different device <---------- Tensor.op = opnode assignment
    all_tensors[weakref.ref(self)] = None                                                                               # add to all_tensors after construction succeeds
    return
  
  @staticmethod
  def _frompy(x:list|tuple|bytes, dtype:DType) -> OpNode:
    # if isinstance(x, bytes):
    #   output_opnode, data = OpNode.new_buffer("PYTHON", len(x)//dtype.itemsize, dtype), x
    # else:
    assert dtype.fmt is not None, f"{dtype=} has None fmt"
    output_opnode = OpNode.new_buffer("PYTHON", prod(shape:=get_shape(x)), dtype).reshape(shape)
    truncate_function = truncate[dtype]
    data = struct.pack(f"{output_opnode.size}{dtype.fmt}", *[truncate_function(dtypes.as_const(xi, dtype)) for xi in fully_flatten(x)])
    mv = memoryview(data if Device.DEFAULT != "PYTHON" else bytearray(data))
    output_opnode.storage.allocate(mv) # fake realize
    return output_opnode



  # < ------------------------------------------------- --high level: .randn_like, randint, normal, uniform, scaled_uniform, kaiming_normal, randperm, multinomial 





  # ************ Tensor Operations ************
  # f'(x) backward
  def backward(self, grad:Tensor|None=None) -> Self:
    """
    backward performs by collecting tensors, computing gradients with automatic differentiation, and updating said tensors.
    """
    # 1. collect all tensors that requires grad by topologically sorting the graph of uops and filter
    all_uops = self.op.toposort()
    tensors_require_grad: list[Tensor] = [t for tref in all_tensors if (t:=tref()) is not None and t.op in all_uops and t.requires_grad]
    uops_require_grad = [t.op for t in tensors_require_grad]
    assert grad is not None or self.shape == tuple(), "when no gradient is provided, backward must be called on a scalar tensor"
    if not (self.is_floating_point() and all(t.is_floating_point() for t in tensors_require_grad)): raise RuntimeError("only float Tensors have gradient")
    
    # 2. compute the gradient with a map of tensors to partials
    if grad is None: grad = Tensor(1.0, dtype=self.dtype, device=self.device, requires_grad=False) # base case is 1.0
    tens2grads = Tensor._automatically_differentiate(self.op, grad.op, set(uops_require_grad)) # skipping materializing zerod grads for now
    grads = [Tensor(g, device=t.device) for t,g in zip(tens2grads.keys, tens2grads.values)] # initialize tensor grads on device
    
    # 3. update the tensors that require grad with the gradient's partials
    for t,g in zip(tensors_require_grad, grads):
      assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
      t.grad = g if t.grad is None else (t.grad + g) # accumulate if t.grad exists
    return self
    
  @staticmethod
  def _automatically_differentiate(root:OpNode, root_grad:OpNode, targets:set[OpNode]) -> dict[OpNode, OpNode]:
    """
    _differentiate backpropagates partials on a topologically sorted expression graph with the chain rule
    and produces the gradient in the form of a map of ops to their partials (which, in turn, are ops)
    """
    tens2grads = {root: root_grad}

    # 1. topological sort ordering is NP-complete?????????? <--------------------------- MOOOOOOOOOOOOOOOOOOSEEEEEE
    in_target_path: dict[OpNode, bool] = {}
    for u in root.toposort(): in_target_path[u] = any(x in targets or in_target_path[x] for x in u.inputs)
    dfs = list(root.toposort()) # lambda node: node.op not in {OpCode.DETACH, OpCode.ASSIGN} and in_target_path[node])) # don't flow through DETACH/ASSIGN or anything not in target path

    # 2. backpropagation with the chain rule
    for tensor in reversed(dfs):
      if tensor not in tens2grads: continue

      local_grads: tuple[OpNode|None, ...]|None = cast(tuple[OpNode, ...]|None, chain_rules.rewrite(tensor, ctx=tens2grads[tensor]))
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
    output_opnode: OpNode = f(*[t.op for t in (self,)+other]) # <------------ compiler pipeline: if EAGER ... elif GRAPH ... else ...
    needs_input_grad = [t.requires_grad for t in (self,)+other]
    requires_grad=True if any(needs_input_grad) else None if None in needs_input_grad else False
    output_tensor = Tensor(output_opnode, device=output_opnode.device, requires_grad=requires_grad)
    return output_tensor

  def recip(self) -> Self: return self.forward(OpNode.recip)
  def sigmoid(self) -> Self: return (1 + (self * (-1/math.log(2))).exp2()).recip()
  def tanh(self) -> Self: return 2.0 * ((2.0 * self).sigmoid()) - 1.0
  def exp2(self) -> Self: return self.forward(OpNode.exp2)
  def log2(self) -> Self: return self.forward(OpNode.log2)
  def fa(self) -> Self: raise NotImplementedError("todo")
  def mm(self) -> Self: raise NotImplementedError("todo")
  # the builtin primitives (i.e add) which do not need to be desugared will call provided OpMixin._binop() to __________
  # which is overrided to ____________
  def _binop(self, op, x, reverse):
    return self._apply_broadcasted_uop(lambda *u: OpNode.alu(u[0], op, *u[1:]), x, reverse) # _binop is used by MathTrait
  def _apply_broadcasted_uop(self, fxn:Callable, x:Tensor|ConstType, reverse=False) -> Self:
    lhs,rhs = self._broadcasted(x, reverse)
    return lhs.forward(fxn, rhs)

  # reduce ops
  # movement ops
  # --high level: gather, cat, stack, repeat, split, chunk, unfold, squeeze, unsqueeze
  # --low level: view, reshape, expand, permute, flip, shrink, pad

__all__ = ["Tensor"]