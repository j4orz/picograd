# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
#         and https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py
from __future__ import annotations
from typing import Any, Callable, Self, Sequence, TypeGuard,  cast, get_args
import math, weakref, struct, pathlib

from picograd import helpers
from picograd.engine import OpCode, OpNode, GraphBuilder
from picograd.helpers import DEBUG, EAGER, GRAPH
from picograd.runtime import Device
from picograd.dtype import Const, DType, DTypeLike, dtypes
import picograd.dtype

all_tensors: dict[weakref.ref[Tensor], None] = {}
def all_same(items:tuple[T, ...]|list[T]): return all(x == items[0] for x in items)
def all_int(t: Sequence[Any]) -> TypeGuard[tuple[int, ...]]: return all(isinstance(s, int) for s in t)

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

class Tensor(GraphBuilder):
  """
  the Tensor class is a *sugared handle* to the expression graph of vertices V=Set<OpNode> and edges E=Set<(OpNode,OpNode)>,
  which represents picograd's primitive understanding (intermediate representation) of the specified expression f(x).
  the data and functionality you expect to live on Tensor actually lives in OpNode because the Tensor class is a sugared handle *to* the expression graph
  """

  # ************ Tensor Data + Constructors ************
  __slots__ = "opnode", "grad", "requires_grad", 
  @property
  def device(self) -> str|tuple[str, ...]: return self.opnode.device
  @property
  def shape(self) -> tuple[int, ...]: return self.opnode.shape
  @property
  def dtype(self) -> DType: return self.opnode.dtype
  @property
  def ndim(self) -> int: raise NotImplementedError("TODO")

  def data(self) -> memoryview:
    if 0 in self.shape: return memoryview(bytearray(0)).cast(self.dtype.base.fmt)
    assert all_int(self.shape), f"no data if shape is symbolic, {self.shape=}"
    return self._buffer().as_typed_buffer(self.shape)

  def item(self) -> Const:
    assert self.numel() == 1, "must have one element for item"
    return self.data()[(0,) * len(self.shape)]
  
  # high level: .randn_like, randint, normal, uniform, scaled_uniform, kaiming_normal, randperm, multinomial
  @staticmethod
  def zeros(*shape, **kwargs) -> Self: return Tensor.full(helpers.normalize_shape(*shape), 0.0)._evaluate()
  @staticmethod
  def ones(*shape, **kwargs) -> Self: return Tensor.full(helpers.normalize_shape(*shape), 1.0)._evaluate()  
  @staticmethod
  def full(shape: tuple[int, ...], fill: Const) -> Self:
    new_shape = (1, )*len(helpers.normalize_shape(shape))
    return Tensor(fill, force_unique=True).reshape(new_shape).expand(new_shape)._evaluate()
  
  def __init__(self, input: Const | bytes | list | tuple | None, # removed OpNode, np.ndarray, pathlib.Path support (for now).
               device: str |tuple | list | None=None, dtype: DTypeLike | None=None, requires_grad: bool | None=None, force_unique: bool=False): # kwargs
    """
    Tensor.__init__() constructs the OpNode for the Tensor's given device and dtype
    """
    if device is None and isinstance(input, pathlib.Path): device = f"DISK:{input.resolve()}"  # keep it on the disk if device is None
    if DEBUG >= 1: print("START Tensor.__init__() initializing tensor with dtype:", dtype, "on device:", device, "...")
    dtype: DType | None = picograd.dtype.to_dtype(dtype) if dtype is not None else None
    device: str | tuple[str, ...] = tuple(Device.canonicalize_device(x) for x in device) if isinstance(device, (tuple, list)) else Device.canonicalize_device(device)
    self.grad: Tensor | None = None                                                            # tensors can have gradients if you have called .backward
    self.requires_grad: bool | None = requires_grad                                            # NOTE: this can be in three states. False and None: no gradient, True: gradient. None (the default) will be updated to True if it's put in an optimizer
    self.opnode: OpNode = Tensor._input_to_opnode(input, device, dtype, force_unique)
    all_tensors[weakref.ref(self)] = None                                                      # add to all_tensors after construction succeeds
    if DEBUG >= 1: print("DONE Tensor.__init__() initializing tensor with dtype:", dtype, "on device:", device, "...")
    return
  
  @staticmethod
  def _input_to_opnode(input: Const|bytes|list|tuple|OpNode|None, device: str, dtype: DType|None, force_unique) -> OpNode:
    if DEBUG >= 1: print("START Tensor._input_to_opnode() constructing Tensor's OpNode...")
    if isinstance(input, OpNode):                                               raise NotImplementedError("todo")
    elif input is None:                                                         opnode = OpNode.const(dtype or dtypes.default_float, 0, device, (), unique=force_unique)
    elif isinstance(input, get_args(Const)):                                    opnode = OpNode.const(dtype or dtypes.from_py(input), input, device, (), unique=force_unique)
    elif isinstance(input, bytes):                                              opnode = Tensor._frompy(input, dtypes.uint8 if dtype is None else dtype)
    elif isinstance(input, (list, tuple)):
      if dtype is None:
        if (d := fully_flatten(input)) and all(isinstance(s, bool) for s in d): dtype = dtypes.bool
        else:                                                                   dtype = dtypes.default_int if d and all_int(d) else dtypes.default_float # NOTE: this works because all_int([True, False]) is True

      if dtype in [dtypes.bfloat16, *dtypes.fp8s]:                              opnode = Tensor(Tensor._frompy(input, dtypes.float32), device=device).cast(dtype).uop
      else:                                                                     opnode = Tensor._frompy(input, dtype)

    if not isinstance(opnode, OpNode): raise RuntimeError(f"can't create Tensor from {input!r} with type {type(input)}") # by this point it has to be a UOp
    if DEBUG >= 1: print("DONE Tensor._input_to_opnode() constructing Tensor's OpNode...")
    return opnode if opnode.device == device else opnode.copy_to_device(device)                    # data might be on a different device
  
  @staticmethod
  def _frompy(input:list|tuple|bytes, dtype:DType) -> OpNode:
    assert dtype.fmt is not None, f"{dtype=} has None fmt"
    if DEBUG >= 1: print("START _frompy(): constructing OpNode from python input", input)
    output_opnode = OpNode.new_buffer("HOST", helpers.prod(shape:=get_shape(input)), dtype) # BUFFER
    if DEBUG >= 1: print("_frompy() 1. created OpNode with OpCode.BUFFER-------------------------")
    output_opnode = output_opnode.reshape(shape)                                              # RESHAPE
    if DEBUG >= 1: print("_frompy() 2. applied OpCode.RESHAPE-------------------------")
    bytes = memoryview(struct.pack(f"{output_opnode.size}{dtype.fmt}", *[picograd.dtype.truncate[dtype](dtypes.as_const(xi, dtype)) for xi in fully_flatten(input)]))
    if DEBUG >= 1: print("_frompy() 3. allocating the buffer-------------------------")
    output_opnode.buffer.allocate(bytes) # fake realize. calling .storage.allocate() and passing bytes/memoryview as pre-allocated buf
    # todo: actually realize(evaluate/materialize)
    if DEBUG >= 1: print("DONE _frompy: constructing OpNode from python input", input)

    return output_opnode

  # **************** GraphBuilder Required Methods ****************
  def _apply_compute_opcode(self, ftype: OpCode, *inputs): raise NotImplementedError("todo")
  def _apply_movement_opcode(self, ftype: OpCode, *inputs): return self._forward(OpNode._apply_movement_opcode, extra_args=(op,), arg=arg)

  # ************ Tensor Operations ************
  def _evaluate(self, *other: Tensor) -> Self:
    """
    ._evaluate() is the method which all evaluations funnel through, regardless of whether the operation is either
      1. sugar: directly calls ._evaluate() or
      2. primitive: indirectly calls ._evaluate() by _binop override
    in either case, ._evaluate() evaluates f(x) by calling f, implemented by Op.eval(), which in turn, launches device kernels.
    ._evaluate() then wraps the new output Op in the expression graph with a Tensor handle
    """
    unrealized_tensors = [tensor for tensor in (self,)+other if not tensor.opnode.is_contiguous()]
    return self

  # f(x) forward
  def _forward(self, f: Callable, *other: Tensor) -> Self:
    """
    ._forward() is the internal graph(IR)-builder which constructs an OpNode
    that represents the computation of f applied to the provided Tensors.
    keep in mind that the Tensor returned is simply a *handle* to unevaluated, unmaterialized graph IR,
    which internal call-sites evaluate with ._evaluate() — teenygrad's .realize() equivalent.
    """
    output_opnode: OpNode = f(*[t.opnode for t in (self,)+other]) # <------------ compiler pipeline: if EAGER ... elif GRAPH ... else ...
    needs_input_grad = [t.requires_grad for t in (self,)+other]
    requires_grad=True if any(needs_input_grad) else None if None in needs_input_grad else False
    output_tensor = Tensor(output_opnode, device=output_opnode.device, requires_grad=requires_grad)
    return output_tensor
  
  # f'(x) backward
  def backward(self, grad: Tensor|None=None) -> Self:
    """
    backward performs by collecting tensors, computing gradients with automatic differentiation, and updating said tensors.
    """
    # 1. collect all tensors that requires grad by topologically sorting the graph of uops and filter
    all_uops = self.opnode.toposort()
    tensors_require_grad: list[Tensor] = [t for tref in all_tensors if (t:=tref()) is not None and t.opnode in all_uops and t.requires_grad]
    uops_require_grad = [t.opnode for t in tensors_require_grad]
    assert grad is not None or self.shape == tuple(), "when no gradient is provided, backward must be called on a scalar tensor"
    if not (self.is_floating_point() and all(t.is_floating_point() for t in tensors_require_grad)): raise RuntimeError("only float Tensors have gradient")
    
    # 2. compute the gradient with a map of tensors to partials
    if grad is None: grad = Tensor(1.0, dtype=self.dtype, device=self.device, requires_grad=False) # base case is 1.0
    tens2grads = Tensor._automatically_differentiate(self.opnode, grad.opnode, set(uops_require_grad)) # skipping materializing zerod grads for now
    grads = [Tensor(g, device=t.device) for t,g in zip(tens2grads.keys, tens2grads.values)] # initialize tensor grads on device
    
    # 3. update the tensors that require grad with the gradient's partials
    for t,g in zip(tensors_require_grad, grads):
      assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
      t.grad = g if t.grad is None else (t.grad + g) # accumulate if t.grad exists
    return self
    
  @staticmethod
  def _automatically_differentiate(root: OpNode, root_grad: OpNode, targets: set[OpNode]) -> dict[OpNode, OpNode]:
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




  # ************ Tensor Sugar ************
  def recip(self) -> Self: return self._forward(OpNode.recip)._evaluate()
  def sigmoid(self) -> Self: return (1 + (self * (-1/math.log(2))).exp2()).recip()._evaluate()
  def tanh(self) -> Self: return (2.0 * ((2.0 * self).sigmoid()) - 1.0)._evaluate()
  def exp2(self) -> Self: return self._forward(OpNode.exp2)._evaluate()
  def log2(self) -> Self: return self._forward(OpNode.log2)._evaluate()
  def fa(self) -> Self: raise NotImplementedError("todo")
  def mm(self) -> Self: raise NotImplementedError("todo")
  # the builtin primitives (i.e add) which do not need to be desugared will call provided OpMixin._binop() to __________
  # which is overrided to ____________
  def _apply_compute_binopcode(self, op, x, reverse):
    return self._apply_broadcasted_uop(lambda *u: OpNode.alu(u[0], op, *u[1:]), x, reverse) # _binop is used by MathTrait
  def _apply_broadcasted_uop(self, fxn:Callable, x:Tensor|Const, reverse=False) -> Self:
    lhs,rhs = self._broadcasted(x, reverse)
    return lhs.evaluate(fxn, rhs)

  # reduce ops
  # movement ops
  # --high level: gather, cat, stack, repeat, split, chunk, unfold, squeeze, unsqueeze
  # --low level: view, reshape, expand, permute, flip, shrink, pad

__all__ = ["Tensor"]