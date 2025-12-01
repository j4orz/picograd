from __future__ import annotations
from re import Pattern
from typing import Self
import ctypes
from dataclasses import dataclass

from picograd.helpers import DEBUG, MAX_BUFFER_SIZE
from picograd.engine.compiler import PatternMatcher
from picograd.engine.irparser import OpCode, OpMixin
from picograd.runtime.device import Buffer, Device
from picograd.dtype import ConstLike, DType, dtypes

# picograd to tinygrad bridge
# - removed buf_op and as_buf used by haldie/tvm schedule/rangify to map high level ops back to buffers
# - removed buf_target
# - rename OpMixin.alu() -> OpMixin.eval()
# - retrofit an eager interpreter in OpMixin.eval()

chain_rules = PatternMatcher([
  # (Pat(OpCode.CAST, name="ret"), lambda ctx, ret: (ctx.cast(ret.src[0].dtype),)),
  (Pattern(OpCode.RECIP, name="input"), lambda output_grad, input: (-output_grad * input * input,)),
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

def toposort(gate:Callable|None=None) -> dict[OpNode, None]:
  visited: dict[OpNode, None] = {}
  stack: list[tuple[OpNode, bool]] = [(self, False)] # each stack entry is (node, visited_flag)

  while stack:
    node, visited = stack.pop()
    if node in visited: continue
    if not visited:
      if gate is None or gate(node): # MOOSE gate?
        stack.append((node, True))  # push node back on stack to process after its srcs
        for s in reversed(node.inputs): stack.append((s, False)) # push srcs on the stack
    else: visited[node] = None # second time i'm seeing this node, add it to returned toposort
  return visited

# **************** Expression Graph ****************
@dataclass(eq=False, slots=True) # NOTE: this should be frozen, but frozen is slower
class OpNode(OpMixin):
  """
  GraphOp structs (which Tensor's deusugar into) are vertices which form an expression graph G=(V,E) where V is a Set<Op> and E is a Set<(Op,Op)>
  the name of the struct "Op" is somewhat of a misnomer because the structs store
  *state* for both the a. specified compute (OpCode) and b. the allocated memory (Buffer)
  so it's more accurate to conceptualize the struct of Op as both the function type f: _ -> _ and the evaluation of said function f(.)
  produced by the *functionality* of the dynamically "eager" interpreter and the just-in-time lazy "graph" compiler.

  the derivative f'(x) is the sum of path products on the expression graph, where factors in the product are local derivatives.
  selecting the optimal order to evaluate such path products (given that the operations represented by each vertex is *associative*) is NP-hard.
  since the functions f(x) that need to be differentiated in the field of machine learning are loss functions of the form f: R^n -> R which fan-out,
  the reverse direction is heuristically used with a reverse topological sort given that the time complexity is proportional to the number of outputs m which in this case is 1
  for many deeplearning workloads that are a series of matrix-matrix multiplications with a final matrix-vector multiplication, multiplying in the reverse direction results in [(v,e)->(v,e)^2]
  """
  inputs: tuple[OpNode, ...] = tuple()
  ftype: OpCode
  dtype: DType = dtypes.void
  storage: Buffer # make this Optional when adding the compiler pipeline

  @property
  def device(self) -> str|tuple[str, ...]: return unwrap(self._device)
  @property
  def size(self) -> int: return prod([int(x.vmax) if isinstance(x, OpNode) else x for x in self.shape])

  @staticmethod
  def new_buffer(device:str|tuple[str, ...], size:int, dtype:DType, num=None) -> Self:
    # return OpNode(OpCode.BUFFER, dtype, (OpNode.unique(num), OpNode(OpCode.DEVICE, arg=device)), size) <--- for now, picograd's retrofitted eager semantics mean all opnodes have their buffers eagerly materialized
    return Self()

  def const_like(self, b:ConstLike):
    return OpNode.const(self.dtype, b, device=self._device, shape=self._shape) # constants can optionally have a DEVICE source
  
  # **************** Compute ****************
  def eval(self, ftype: OpCode, *inputs:OpNode) -> Self: # required method by OpMixin
    """
    the evaluator overrides* the semantics of the host language with a nonstandard interpretation (device acceleration of f(x), automatic differentiation of f'(x))
    called by OpMixin.eval() which acts as the embedded DSL's "parser", by coupling python dunder builtins to be aware of the corresponding OpCode intermediate representation
    *: note that .eval is not a static method, that self is the Op that the OpCode ftype is operating on, to produce a new Self
    """
    out_dtype = (self, *inputs)[-1].dtype
    # if op in {OpCode.CMPLT, OpCode.CMPNE, OpCode.CMPEQ}: out_dtype = dtypes.bool.vec(out_dtype.count) if out_dtype.count > 1 else dtypes.bool
    # return Op((self,)+inputs, ftype, out_dtype,)
    match ftype:
      case OpCode.NEG: launch_neg(*inputs)
      case OpCode.ADD:
        # 1. memory: allocate and memcpy on device
        device = HIPDevice()
        a, b, c = [device.allocator.alloc(4), device.allocator.alloc(4), device.allocator.alloc(4)]
        device.allocator._copyin(a, memoryview(bytearray([2,0,0,0])))
        device.allocator._copyin(b, memoryview(bytearray([3,0,0,0])))
        # 2. compute: compile a kernel to a binary
        kernel = HIPCCCompiler().compile("__global__ void add(int *a, int *b, int *c) { int id = blockDim.x * blockIdx.x + threadIdx.x; if(id < N) c[id] = a[id] + b[id]; }")
        # 3. launch
        f = device.kernel("add", kernel)
        f(a, b, c) # HIPKernel

        print(val := device.allocator._as_buffer(c).cast("I").tolist()[0])
        assert val == 5 # check the data out
      case OpCode.MUL: raise NotImplementedError("todo")
      case OpCode.MM: raise NotImplementedError("todo")
      case OpCode.RECIP: raise NotImplementedError("todo")
      case OpCode.EXP2: raise NotImplementedError("todo")
      case OpCode.LOG2: raise NotImplementedError("todo")
      case OpCode.SIN: raise NotImplementedError("todo")
      case _: raise NotImplementedError(f"unsupported opcode {ftype!r}")
  













  
  def _device(self) -> str|tuple[str, ...]|None: # @recursive_property
    if self.op is OpCode.DEVICE: return self.arg
    if self.op is OpCode.BUFFERIZE: return self.arg.device
    if self.op is OpCode.AFTER: return self.inputs[0]._device
    if self.op is OpCode.MSELECT:
      assert isinstance(self.inputs[0].device, tuple), "mselect must be on tuple device"
      return self.inputs[0].device[self.arg]
    if self.op is OpCode.MSTACK: return tuple(cast(str, x.device) for x in self.inputs)
    if self.op in {OpCode.COPY, OpCode.BUFFER, OpCode.ALLREDUCE}: return self.inputs[1].device
    for x in self.inputs:
      if x._device is not None: return x._device
    return None

  @property
  def storage(self) -> Buffer|MultiBuffer:
    from tinygrad.device import Buffer, MultiBuffer
    if self is not self.base:
      assert self.op is OpCode.RESHAPE, f"can only be RESHAPE {self}"
      return self.inputs[0].storage
    if self.op is OpCode.MSELECT:
      ret = self.inputs[0].storage
      assert isinstance(ret, MultiBuffer)
      return ret.bufs[self.arg]
    if self.op is OpCode.MSTACK:
      ret = MultiBuffer.__new__(MultiBuffer)
      ret.bufs = [cast(Buffer, x.storage) for x in self.inputs]
      assert all_same([x.size for x in ret.bufs]) and all_same([x.dtype for x in ret.bufs]), "multibuffers mismatch buffers"
      return ret
    assert self.op is OpCode.BUFFER, f"must be BUFFER {self.op}"
    if (cret:=buffers.get(self)) is not None: return cret
    rdtype = self.dtype if isinstance(self.dtype, ImageDType) else self.dtype.base
    if isinstance(self.device, tuple): ret = MultiBuffer(self.device, self.size, rdtype).ref(1)
    else: ret = Buffer(self.device, self.size, rdtype).ref(1)
    buffers[self] = ret
    return ret