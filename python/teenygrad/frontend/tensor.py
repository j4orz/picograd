# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
#          and https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py
from __future__ import annotations
from typing import Any, Callable, Self, Sequence, TypeGuard, cast, get_args
import math, weakref, struct, pathlib

from teenygrad import helpers
from teenygrad.compiler import OpCode, OpNode, TensorDSL
from teenygrad.helpers import DEBUG, EAGER, GRAPH
from teenygrad.runtime import Device
from teenygrad.dtype import Const, DType, DTypeLike, dtypes
import teenygrad.dtype



import array

class InterpretedTensor:
  @staticmethod
  def arange(end: int) -> Self: return InterpretedTensor((end,), list(range(end)))
  @staticmethod
  def zeros(shape: tuple[int, ...]) -> Self:
    numel = math.prod(shape)
    tensor = InterpretedTensor((numel,), [0.0]*numel).reshape(shape)
    return tensor
  @staticmethod
  def ones(shape: tuple[int, ...]) -> Self:
    numel = math.prod(shape)
    tensor = InterpretedTensor((numel,), [1.0]*numel).reshape(shape)
    return tensor
  
  def __init__(self, shape: tuple[int, ...], storage: list[float]) -> Self:
    self.shape = shape
    self.stride = [math.prod(shape[i+1:]) for i in range(len(shape))] # math.prod([]) produces 1
    self.storage = storage

  def reshape(self, shape: tuple[int, ...]) -> Self:
    self.shape = shape
    self.stride = [math.prod(shape[i+1:]) for i in range(len(shape))] # math.prod([]) produces 1
    return self
  
  def __repr__(self) -> str:
    return f"InterpretedTensor({self.chunk(self.storage, self.shape)})"
  
  @property
  def numel(self): return math.prod(self.shape) # np (and thus jax) call this .size
  @property
  def ndim(self): return len(self.shape)

  @staticmethod
  def chunk(flat, shape):
    if len(shape) == 1: return flat[:shape[0]]
    size = len(flat) // shape[0]
    return [InterpretedTensor.chunk(flat[i*size:(i+1)*size], shape[1:]) for i in range(shape[0])]
  
  def __radd__(self, other: Self) -> Self: return self.__add__(other)
  def __add__(self, other: Self) -> Self:
    n, alpha = self.numel, 1
    x, y = array.array('f', self.storage), array.array('f', other.storage)
    teenygrad.rs.cpu.saxpy(n, alpha, x, y) # y=axpy
    return InterpretedTensor(self.shape, list(y))

  def __rmatmul__(self, other: Self) -> Self: return other.__matmul__(self) # GEMM does not commute: AB != BA
  def __matmul__(self, other: Self) -> Self:
    if other.ndim == 1: # gemv
      import sys
      sys.stdout.flush()
      m, n = self.shape[0], self.shape[1]
      alpha, beta = 1, 1
      a, x, y = array.array('f', self.storage), array.array('f', other.storage), array.array('f', [0.0]*m)
      teenygrad.rs.cpu_kernels.sgemv(m, n, alpha, beta, a, x, y)
      sys.stdout.flush()
      return InterpretedTensor((m,), list(y))
    elif other.ndim == 2: # gemm
      import sys
      import array
      sys.stdout.flush()
      m, n, p = self.shape[0], other.shape[1], self.shape[1]
      alpha, beta = 1, 1
      a, b, c = array.array('f', self.storage), array.array('f', other.storage), array.array('f', [0.0]*(m * n))
      teenygrad.rs.cpu.sgemm(m, n, p, alpha, beta, a, b, c)
      sys.stdout.flush()
      output_tensor = InterpretedTensor((m,n), list(c))
      return output_tensor      
    else:
      raise NotImplementedError("todo")

  


all_tensors: dict[weakref.ref[CompiledTensor], None] = {}
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

class CompiledTensor(TensorDSL):
  """
  the Tensor class is a *sugared handle* to the expression graph of vertices V=Set<OpNode> and edges E=Set<(OpNode,OpNode)>,
  which represents teenygrad's primitive understanding (intermediate representation) of the specified expression f(x).
  the data and functionality you expect to live on Tensor actually lives in OpNode because the Tensor class is a sugared handle *to* the expression graph.
  all methods that framework users apply to Tensors are implemented by calling GraphBuilder's methods, which, in turn, call
  ComputeOpCodeBuilder._apply_compute_opcode() and MovementOpCodeBuilder._apply_movement_opcode(), implemented by OpNode.
  *:  keep in mind that the semantics of these two methods are applying *ir op code*
      that is, to maintain parity in semantics with tinygrad (and a smooth pedagogical progression),
      the returned OpNode's are still un-{materialized/realized/evaluated}, and caller's (namely tensor.py)
      need to invoke .eval() on the OpNode for eager semantics.
  **: if you are wondering where and how teenygrad's shape, strides, and storage work (i.e movement operations), continue reading through OpNode's source.
      in short, shapes, strides, and storage is encoded *in* the IR!
  """

  # ************ Tensor Data + Constructors ************
  __slots__ = "shape", "stride", "storage", "_device" \
              "opnode", \
              "grad", "requires_grad"

  def __init__(self, input: Const|bytes|list|tuple|None, # removed OpNode, np.ndarray, pathlib.Path support (for now).
               device: str|tuple|list|None=None, dtype: DTypeLike|None=None, requires_grad: bool|None=None, force_unique: bool=False): # kwargs
    """
    Tensor.__init__() initializes state for self, which includes metadata for device, dtype, gradients, and most importantly, the handle to an OpNode
    """
    if device is None and isinstance(input, pathlib.Path): device = f"DISK:{input.resolve()}"  # keep it on the disk if device is None
    if DEBUG >= 1: print("START Tensor.__init__() initializing tensor with dtype:", dtype, "on device:", device, "...")
    self.grad: CompiledTensor | None = None                                                            # tensors can have gradients if you have called .backward
    self.requires_grad: bool | None = requires_grad                                            # NOTE: this can be in three states. False and None: no gradient, True: gradient. None (the default) will be updated to True if it's put in an optimizer
    device: str | tuple[str, ...] = tuple(Device._canonicalize_device(x) for x in device) if isinstance(device, (tuple, list)) else Device._canonicalize_device(device)
    dtype: DType | None = teenygrad.dtype.to_dtype(dtype) if dtype is not None else None

    if helpers.EAGER:
      self.shape: tuple[int] | None = []
      self.stride: tuple[int] | None = []
      self.storage: None = None
      self.offset: int = 0
      self._device, self._dtype = device, dtype

    elif helpers.GRAPH:
      self.opnode: OpNode = CompiledTensor._input_to_opnode(input, device, dtype, force_unique)
      print(f"DONE Tensor.__init__() initializing tensor with dtype: {dtype} on device {device} with input {input}\n")
      print(f"Tensor.opnode is now: {self.opnode}\n\n\n")
    else: raise NotImplementedError("todo")

    all_tensors[weakref.ref(self)] = None                                                      # add to all_tensors after construction succeeds
    return

  def data(self) -> memoryview:
    if 0 in self.shape: return memoryview(bytearray(0)).cast(self.dtype.base.fmt)
    assert all_int(self.shape), f"no data if shape is symbolic, {self.shape=}"
    return self._buffer().as_typed_buffer(self.shape)

  def item(self) -> Const:
    assert self.numel() == 1, "must have one element for item"
    return self.data()[(0,) * len(self.shape)]

  @property
  def device(self) -> str|tuple[str, ...]:
    if helpers.EAGER: return self._device
    elif helpers.GRAPH: return self.opnode.device

  @property
  def dtype(self) -> DType:
    if helpers.EAGER: return self._dtype
    elif helpers.GRAPH: return self.opnode.dtype
  @property
  def ndim(self) -> int: raise NotImplementedError("TODO")

  # high level: .randn_like, randint, normal, uniform, scaled_uniform, kaiming_normal, randperm, multinomial
  @staticmethod
  def zeros(*shape, **kwargs) -> Self: return CompiledTensor.full(helpers.normalize_shape(*shape), 0.0)._evaluate()
  @staticmethod
  def ones(*shape, **kwargs) -> Self: return CompiledTensor.full(helpers.normalize_shape(*shape), 1.0)._evaluate()  
  @staticmethod
  def full(shape: tuple[int, ...], fill: Const) -> Self:
    new_shape = (1, )*len(helpers.normalize_shape(shape))
    return CompiledTensor(fill, force_unique=True).reshape(new_shape).expand(new_shape)._evaluate()
  
  @staticmethod
  def _input_to_opnode(input: Const|bytes|list|tuple|OpNode|None, device: str, dtype: DType|None, force_unique) -> OpNode:
    if DEBUG >= 1: print("START Tensor._input_to_opnode() constructing Tensor's OpNode...")
    
    if input is None:                                                           opnode = OpNode.const(dtype or dtypes.default_float, 0, device, (), unique=force_unique)
    elif isinstance(input, get_args(Const)):                                    opnode = OpNode.const(dtype or dtypes.from_py(input), input, device, (), unique=force_unique)
    elif isinstance(input, bytes):                                              opnode = CompiledTensor._hostseq2dslopnode(input, dtypes.uint8 if dtype is None else dtype)
    elif isinstance(input, (list, tuple)):
      if dtype is None:
        if (d := fully_flatten(input)) and all(isinstance(s, bool) for s in d): dtype = dtypes.bool
        else:                                                                   dtype = dtypes.default_int if d and all_int(d) else dtypes.default_float # NOTE: this works because all_int([True, False]) is True

      if dtype in [dtypes.bfloat16, *dtypes.fp8s]:                              opnode = CompiledTensor(CompiledTensor._hostseq2dslopnode(input, dtypes.float32), device=device).cast(dtype).uop
      else:                                                                     opnode = CompiledTensor._hostseq2dslopnode(input, dtype)
      if DEBUG >= 1: print("DONE _input_to_opnode converting hostseq to dslopnode...")
    elif isinstance(input, OpNode):
      assert dtype is None or dtype==input.dtype, f"dtype doesn't match ({dtype} vs {input.dtype}), and casting isn't supported"
      # if input is dtype.index that means that this is a symbolic int and we need to lower it to something we can make a Tensor out of
      # if input.dtype==dtypes.index: input = _index_to_concrete_int(input)
      # if input.opcode is OpCode.BIND:
      opnode = input

    if not isinstance(opnode, OpNode): raise RuntimeError(f"can't create Tensor from {input!r} with type {type(input)}") # by this point it has to be a UOp

    if DEBUG >= 1: print(f"opnode's device is {opnode.device}, specified device is {device}")
    return opnode if opnode.device == device else opnode.copy_to_device(device)
  
  @staticmethod
  def _hostseq2dslopnode(input:list|tuple|bytes, dtype:DType) -> OpNode:
    """
    _host2d2l() takes in a python list, tuple, or bytearray and maps it into a teenygrad OpNode for the initialization call stack.
    the resulting OpNode with OpCode.BUFFER is allocated on device "HOST".
    callers must manually transfer the OpNode's backing storage with opnode.copy_to_device(device)
    """
    assert dtype.fmt is not None, f"{dtype=} has None fmt"
    output_opnode = OpNode.new_buffer("HOST", helpers.prod(shape:=get_shape(input)), dtype)
    output_opnode = output_opnode.reshape(shape)
    something = [teenygrad.dtype.truncate[dtype](dtypes.as_const(x, dtype)) for x in fully_flatten(input)]
    bytes = memoryview(struct.pack(f"{output_opnode.size}{dtype.fmt}", *something))
    output_opnode.buffer.allocate(bytes) # fake realize by passing an opaque_preallocation
    # todo: actually realize(evaluate/materialize)
    return output_opnode





  # ************ Tensor Sugar ************
  # the builtin primitives (i.e add) will call provided GraphBuilder._apply_compute_opcode
  def fa(self) -> Self: raise NotImplementedError("todo")
  def mm(self) -> Self: raise NotImplementedError("todo")
  def recip(self) -> Self: return self._forward_computeop(OpNode.recip)
  def sigmoid(self) -> Self: return (1 + (self * (-1/math.log(2))).exp2()).recip()
  def tanh(self) -> Self: return (2.0 * ((2.0 * self).sigmoid()) - 1.0)
  def exp2(self) -> Self: return self._forward_computeop(OpNode.exp2)
  def log2(self) -> Self: return self._forward_computeop(OpNode.log2)
  # reduce ops
  # movement ops high level: gather, cat, stack, repeat, split, chunk, unfold, squeeze, unsqueeze, movement ops low level: view, reshape, expand, permute, flip, shrink, pad

  def _evaluate(self) -> Self:
    """
    ._evaluate() is the method which all evaluations funnel through, regardless of whether the operation is either
      1. sugar: directly calls ._evaluate() or
      2. primitive: indirectly calls ._evaluate() by _binop override
    in either case, ._evaluate() evaluates f(x) by calling f, implemented by Op.eval(), which in turn, launches device kernels.
    ._evaluate() then wraps the new output Op in the expression graph with a Tensor handle
    """
    # unrealized_tensors = [tensor for tensor in (self,)+other if not tensor.opnode.is_contiguous()]
    print("evaluating graph ir..")
    return Interpreter.evaluate(self)
  
  def kernel_name(opcode: OpCode) -> str:
    if   opcode is OpCode.ADD:    return "add"
    elif opcode is OpCode.MUL:    return "mul"
    elif opcode is OpCode.MATMUL: return "matmul"
    else: raise NotImplementedError("todo")

  # **************** ComputeOpCodeBuilder/MovementOpCodeBuilder Methods ****************
  def _forward_computeop(self, opcode: OpCode, *inputs: Self) -> Self: ## required
    """
    ._forward() is the internal graph(IR)-builder which constructs a Tensor handle to OpNode IR
    after the IR for function f is applied to the expression graph with .forward(), internal callsites must evaluate with .evaluate()
    """
    if helpers.EAGER:
      runtime = Device[self.device]
      kernel_name, kernel_extension = kernel_name(opcode), kernel_extension(self.device)
      kernel_src = pathlib.Path(f"python/teenygrad/engine/eagker/{self.device}/{kernel_name}.{kernel_extension}")
      kernel_bin = runtime.compiler.compile(kernel_src.read_text())
      kernel_handle = runtime.kernel(kernel_name, kernel_bin)

      other, output, n = inputs[0], CompiledTensor(None, self.device), self.numel()
      globals, locals = ((n + 255) // 256, 1, 1), (256, 1, 1)
      kernel_handle(output.buf, self.buf, other.buf,
                    globals, locals, vals=(1024,))

      return output
    elif helpers.GRAPH:
      f = lambda *input_opnodes: input_opnodes[0]._apply_compute_opcode(opcode, *input_opnodes[1:])
      self.opnode = self._forward_computeop(f, *inputs)

      needs_input_grad = [t.requires_grad for t in (self,)+other]
      requires_grad = True if any(needs_input_grad) else None if None in needs_input_grad else False
      output_opnode: OpNode = f(*[t.opnode for t in (self,)+other])
      output = CompiledTensor(output_opnode, device=output_opnode.device, requires_grad=requires_grad)
      return output
    else:
      raise NotImplementedError("todo")

   # todo: other is Self. not Const or OpNodestorage  def _forward_computebinop(self, other: Self, opcode: OpCode, reverse): # overriding provided, so Tensor operations go through dtype and broacasting logic below
    print(f"self: {self}",)
    print(f"other: {other}",)
    import sys
    print("moose..")
    sys.stdout.flush()
    teenygrad.rs.cuda_smoke_test()
    sys.stdout.flush()
    print("deer..")

    return

    # if helpers.EAGER:
    #   runtime = Device[self.device]
    #   kernel_name, kernel_extension = kernel_name(opcode), kernel_extension(self.device)
    #   kernel_src = pathlib.Path(f"python/teenygrad/engine/eagker/{self.device}/{kernel_name}.{kernel_extension}")
    #   kernel_bin = runtime.compiler.compile(kernel_src.read_text())
    #   kernel_handle = runtime.kernel(kernel_name, kernel_bin)

    #   output, n = Tensor(None, self.device), self.numel()
    #   globals, locals = ((n + 255) // 256, 1, 1), (256, 1, 1)
    #   kernel_handle(output.buf, self.buf, other.buf,
    #                 globals, locals, vals=(1024,))

    #   return output
    # elif helpers.GRAPH:
    #   # 0. construct f, which when applied with y = f(x), produces the opnode y. the call is delegated to OpNode's _apply_compute_opcode
    #   f = lambda *input_opnodes: OpNode._forward_computeop(input_opnodes[0], opcode, *input_opnodes[1:])
    #   graph = self._forward_computeop(f, other); print("applied opnode to expression graph with tensor._forward()")
    #   schedule = graph.opnode.toposort(); print("linearized opnode graph into opnode schedule with opnode.toposort()")
    #   Interpreter.evaluate(schedule); print("evaluated schedule with Interpreter.evaluate()")
    #   return graph
    # else:
    #   raise NotImplementedError("todo")
  
    # 1. normalize other: Const|OpNodes -> Tensors
    # if not isinstance(other, Tensor):
    #   assert isinstance(other, (*get_args(Const), OpNode)), f"{type(other)=}, {other=}"
    #   if dtypes.is_float(self.dtype) or (dtypes.is_int(self.dtype) and isinstance(other, int)): other_dtype = self.dtype
    #   elif not isinstance(other, OpNode):                                                       other_dtype = dtypes.from_py(other)
    #   if isinstance(other, OpNode):                                                             other = Tensor.from_uop(other, device=self.device)
    #   else:                                                                                     other = Tensor(dtypes.as_const(other, other_dtype), self.device, other_dtype, requires_grad=False)

    # 2. normalize dtypes
    # match_dtype, mismatched_dtype = True, self.dtype != other.dtype
    # if match_dtype and mismatched_dtype:
    #   output_dtype = least_upper_dtype(self.dtype, other.dtype)
    #   self, other = self.cast(output_dtype), other.cast(output_dtype)

    # 3. reverse
    # if reverse: self, other = other, self
  
    # 4. broadcast NOTE: the backward cast is no-op in forward and uses sum_acc_dtype in the backward sum
    # broadcasted_shape = tuple(0 if 0 in nth_dim_sizes else smax(nth_dim_sizes) for nth_dim_sizes in zip(*_align_left(*[self.shape, other.shape])))
    
    # backward_cast = True
    # self, other = self.cast(sum_acc_dtype(self.dtype) if backward_cast else self.dtype)._broadcast_to(broadcasted_shape).cast(self.dtype), \
    #        other.cast(sum_acc_dtype(other.dtype) if backward_cast else other.dtype)._broadcast_to(broadcasted_shape).cast(other.dtype)    storage  
  # def _apply_movement_opcode(self, opcode: OpCode, *inputs):
  #   return self._forward(OpNode._apply_movement_opcode, extra_args=(opcode,), arg=arg)

  def _backward(self, grad: CompiledTensor|None=None) -> Self:
    """
    backward performs by collecting tensors, computing gradients with automatic differentiation, and updating said tensors.
    """
    # 1. collect all tensors that requires grad by topologically sorting the graph of uops and filter
    all_uops = self.opnode.toposort()
    tensors_require_grad: list[CompiledTensor] = [t for tref in all_tensors if (t:=tref()) is not None and t.opnode in all_uops and t.requires_grad]
    uops_require_grad = [t.opnode for t in tensors_require_grad]
    assert grad is not None or self.shape == tuple(), "when no gradient is provided, backward must be called on a scalar tensor"
    if not (self.is_floating_point() and all(t.is_floating_point() for t in tensors_require_grad)): raise RuntimeError("only float Tensors have gradient")
    
    # 2. compute the gradient with a map of tensors to partials
    if grad is None: grad = CompiledTensor(1.0, dtype=self.dtype, device=self.device, requires_grad=False) # base case is 1.0
    tens2grads = CompiledTensor._automatically_differentiate(self.opnode, grad.opnode, set(uops_require_grad)) # skipping materializing zerod grads for now
    grads = [CompiledTensor(g, device=t.device) for t,g in zip(tens2grads.keys, tens2grads.values)] # initialize tensor grads on device
    
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

    # 1. topological sort ordering is NP-complete??????????
    in_target_path: dict[OpNode, bool] = {}
    for u in root.toposort(): in_target_path[u] = any(x in targets or in_target_path[x] for x in u.inputs)
    dfs = list(root.toposort()) # lambda node: node.op not in {OpCode.DETACH, OpCode.ASSIGN} and in_target_path[node])) # don't flow through DETACH/ASSIGN or anything not in target path

    # 2. backpropagation with the chain rule
    for tensor in reversed(dfs):
      if tensor not in tens2grads: continue

      local_grads: tuple[OpNode|None, ...]|None = cast(tuple[OpNode, ...]|None, chain_rules.rewrite(tensor, ctx=tens2grads[tensor]))
      if local_grads is None: raise RuntimeError(f"failed to compute gradient for {tensor.op}\n\nin {str(tensor)[0:1000]}...")
      assert len(local_grads) == len(tensor.inputs), f"got {len(local_grads)} gradient, expected {len(tensor.inputs)}"

      for tensor,local_grad in zip(tensor.inputs, local_grads): # TODO: why are we accumulating inside ad()? don't we do it in backward()??
        if local_grad is None: continue
        if tensor in tens2grads: tens2grads[tensor] = tens2grads[tensor] + local_grad # accumulate if tensor exists
        else: tens2grads[tensor] = local_grad # o/w initialize

        # if len(forward_metadata:=all_metadata.get(tensor, ())):
        #   backward_metadata = tuple(dataclasses.replace(x, backward=True) for x in forward_metadata)
        #   # we add the backward metadata to everything new in the graph
        #   for bw_uop in local_grad.toposort(lambda x: x not in (tensor, *tensor.src, tens2grads[tensor])):
        #     all_metadata[bw_uop] = all_metadata.get(bw_uop, ())+backward_metadata

    return tens2grads

__all__ = ["CompiledTensor"]
