# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
#         and https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py
from __future__ import annotations
from typing import Callable, Self, cast
import math, weakref

from picograd.dtype import ConstType, DType, dtypes
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
  
  def __init__(self,
               op:ConstType|bytes|list|tuple|UOp|'np.ndarray'|pathlib.Path|None,  # type: ignore [name-defined] # noqa: F821
               device:str|tuple|list|None=None,
               dtype:DTypeLike|None=None,
               requires_grad:bool|None=None,
               _force_unique:bool=False
              ):
    """

    """
    if device is None and isinstance(op, pathlib.Path): device = f"DISK:{op.resolve()}"  # keep it on the disk if device is None
    _dtype:DType|None = to_dtype(dtype) if dtype is not None else None
    _device:str|tuple[str, ...] = tuple(canonicalize_device(x) for x in device) if isinstance(device, (tuple, list)) else canonicalize_device(device)
    del device, dtype

    self.grad:Tensor|None = None # tensors can have gradients if you have called .backward
    self.requires_grad:bool|None = requires_grad # NOTE: this can be in three states. False and None: no gradient, True: gradient. None (the default) will be updated to True if it's put in an optimizer
    op = Tensor._construct_op(op) # create an Op
    if isinstance(_device, str): self.op: Op = op if op.device == _device else op.copy_to_device(_device) # data might be on a different device
    # elif isinstance(op.device, str): self.op = Tensor(op).shard(_device).uop # if device is a tuple, we should have/construct a MultiLazyBuffer
    # else: assert op.device == _device, f"MultiLazyBuffer device mismatch, {op.device} != {_device}"; self.op = op
    all_tensors[weakref.ref(self)] = None # add to all_tensors after construction succeeds

  @staticmethod
  def _construct_op(data, _device) -> Op: # removed support for numpy and pathlib.PATH (DISK)
    if isinstance(data, Op):
      assert _dtype is None or _dtype==data.dtype, f"dtype doesn't match ({_dtype} vs {data.dtype}), and casting isn't supported"
      # if data is dtype.index that means that this is a symbolic int and we need to lower it to something we can make a Tensor out of
      if data.dtype==dtypes.index: data = _index_to_concrete_int(data)
      if data.op is Ops.BIND:  # type: ignore  # mypy type narrowing is bugged here
        var, val = data.unbind()  # type: ignore
        # give the bound constant a device
        const = Op.const(var.dtype, val, _device, ())
        data = data.replace(src=(var.replace(src=const.src), const))  # type: ignore
    elif data is None: data = Op.const(_dtype or dtypes.default_float, 0, _device, (), unique=_force_unique)                             # None       =>
    elif isinstance(data, get_args(ConstType)): data = Op.const(_dtype or dtypes.from_py(data), data, _device, (), unique=_force_unique) # const_type =>
    elif isinstance(data, bytes): data = _frompy(data, dtypes.uint8 if _dtype is None else _dtype)                                       # bytes      =>
    elif isinstance(data, (list, tuple)):                                                                                                # list/tuple =>
      if _dtype is None:
        if (d := fully_flatten(data)) and all(isinstance(s, bool) for s in d): _dtype = dtypes.bool
        else: _dtype = dtypes.default_int if d and all_int(d) else dtypes.default_float  # NOTE: this works because all_int([True, False]) is True
      if _dtype in [dtypes.bfloat16, *dtypes.fp8s]: data = Tensor(_frompy(data, dtypes.float32), device=_device).cast(_dtype).uop
      else: data = _frompy(data, _dtype)

    if not isinstance(data, Op): raise RuntimeError(f"can't create Tensor from {data!r} with type {type(data)}") # by this point, it has to be a UOp

    return data

  @staticmethod
  def full(shape:tuple[sint, ...], fill_value:ConstType, **kwargs) -> Self:
    return Tensor(fill_value, _force_unique=True, **kwargs).reshape((1, )*len(new_shape := argfix(shape))).expand(new_shape)
  @staticmethod
  def zeros(*shape, **kwargs) -> Self: return Tensor.full(argfix(*shape), 0.0, **kwargs)
  @staticmethod
  def ones(*shape, **kwargs) -> Self: return Tensor.full(argfix(*shape), 1.0, **kwargs)

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
    output_op: Op = f(*[t.op for t in (self,)+other]) # <------------ compiler pipeline: if EAGER ... elif GRAPH ... else ...
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