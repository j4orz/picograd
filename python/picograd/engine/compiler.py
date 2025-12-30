from dataclasses import dataclass
import math
from typing import Any

from picograd.dtype import DType
from .dslir import OpCode
from .opnode import OpNode

# -kernelize: graph rewrites
# -schedule_with_vars: feeds graph to scheduler and memory planner
# -realize: hands schedule to run_schedule

class PatternMatcher:
  """
  ...
  """
  def __init__(): raise NotImplementedError

class Pattern:
  """
  ...
  """
  __slots__ = ("op", "dtype", "arg", "name", "src")
  def __init__(self, op:OpCode|tuple[OpCode, ...]|set[OpCode]|None=None, dtype:DType|tuple[DType, ...]|None=None,
               src:tuple[Pattern, ...]|list[Pattern]|Pattern|None=None, arg:Any=None,
               name:str|None=None, allow_any_len:bool=False, custom_early_reject:set[OpCode]|None=None, location=None):
    raise NotImplementedError

# chain_rules = PatternMatcher([
#   # (Pat(OpCode.CAST, name="ret"), lambda ctx, ret: (ctx.cast(ret.src[0].dtype),)),
#   (Pattern(OpCode.RECIP, name="input"), lambda output_grad, input: (-output_grad * input * input,)),
#   (Pattern(OpCode.SIN, name="input"), lambda output_grad, input: ((math.pi/2 - input.src[0]).sin() * output_grad,)),
#   (Pattern(OpCode.LOG2, name="input"), lambda output_grad, input: (output_grad / (input.src[0] * math.log(2)),)),
#   (Pattern(OpCode.EXP2, name="input"), lambda output_grad, input: (input * output_grad * math.log(2),)),
#   (Pattern(OpCode.SQRT, name="input"), lambda output_grad, input: (output_grad / (input*2),)),
#   # (Pat((OpCode.CMPLT, OpCode.CMPNE)), lambda: (None, None)),
#   (Pattern(OpCode.ADD), lambda output_grad: (1.0*output_grad, 1.0*output_grad)),
#   # (Pat(OpCode.POW, name="input", src=(Pat.var("b"), Pat.var("e"))), lambda output_grad, input, b, e:
#   #   (output_grad * (b.eq(0)&e.eq(0)).where(e, e*b.pow(e-1)), output_grad * b.eq(0).where((e<0).where(input.const_like(-math.inf), 0), input*b.log2()*math.log(2.0)))),
#   # (Pat(OpCode.MAX, src=(Pat.var("x"), Pat.var("y"))), lambda output_grad, x, y:
#   #   ((x>y).where(output_grad, (x.eq(y)).where(output_grad * 0.5, 0)), (x<y).where(output_grad, (x.eq(y)).where(output_grad * 0.5, 0)))),
#   (Pattern(OpCode.MUL, name="input"), lambda output_grad, input: (input.src[1]*output_grad, input.src[0]*output_grad)),
#   # (Patttern(OpCode.WHERE, name="input"), lambda output_grad, input: (None, input.src[0].where(output_grad, output_grad.const_like(0)), input.src[0].where(output_grad.const_like(0), output_grad))),
#   # (Patttern(OpCode.REDUCE_AXIS, name="input"), reduce_gradient),
#   # (Patttern(OpCode.CONTIGUOUS), lambda output_grad: (output_grad,)),
#   # (Patttern(OpCode.CONTIGUOUS_BACKWARD), lambda output_grad: (output_grad.contiguous(),)),
#   # (Patttern(OpCode.RESHAPE, name="input"), lambda output_grad, input: (output_grad.reshape(input.src[0].shape), None)),
#   # (Patttern(OpCode.EXPAND, name="input"), lambda output_grad, input: (output_grad.r(OpCode.ADD,tuple(i for i,(s,n) in enumerate(zip(input.src[0].shape, input.shape)) if s!=n)), None)),
#   # (Patttern(OpCode.PAD, name="input"), lambda output_grad, input: (output_grad.shrink(tuple([(p[0], s+p[0]) for s,p in zip(input.src[0].shape, input.marg)])), None, None)),
#   # (Patttern(OpCode.SHRINK, name="input"), lambda output_grad, input: (output_grad.pad(tuple([(p[0], s-p[1]) for s,p in zip(input.src[0].shape, input.marg)])), None, None)),
#   # (Patttern(OpCode.PERMUTE, name="input"), lambda output_grad, input: (output_grad.permute(argsort(input.marg)),)),
#   # (Patttern(OpCode.FLIP, name="input"), lambda output_grad, input: (output_grad.flip(input.marg),)),
#   # (Patttern(OpCode.MULTI, name="input"), lambda output_grad, input: output_grad.shard(input.device, input.axis).src),
#   # # NOTE: this is only correct when the KERNEL has a single output
#   # (Patttern(OpCode.AFTER), lambda output_grad: (output_grad, output_grad)),
#   # (Patttern(OpCode.KERNEL, name="k"), lambda output_grad, k: k.arg.grad_fxn(output_grad, k)),
#   # # there's no gradient for bitcast
#   # (Patttern(OpCode.BITCAST), lambda: (None,)),
# ])

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

@dataclass(frozen=True)
class TensorCore: # D = A * B + C, A is (M x K), B is (K x N), C and D are (M x N)
  dims: tuple[int,int,int] # N, M, K
  threads: int # number of threads that construct the warp
  elements_per_thread: tuple[int, int, int] # elements per-thread to load/store from A/B/C
  dtype_in: DType # dtype for A and B
  dtype_out: DType # dtype for C and D
  opts: tuple[str, ...] # ordered tuple of "ux" or "lx" specifying kernel opts to perform. "ux" upcasts dim x and "lx" localizes dim x
  # (local_swizzle, upcast_swizzle, reduce_swizzle)
  # l<num> is the num axis of the locals, similar for u<num> and upcasts, r<num> and reduces
  swizzle: tuple[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]]

class Generator:
  """
  picograd follows tinygrad (and torch/xla and swift for tensorflow) with lazy graph capture, see (Suhan et al. https://arxiv.org/abs/2102.13267)
  and modifying the semantics of the programming model where users must explicitly materialize data with .realize(),
  as opposed to pt2 which maintains the eager programming model surface via graph capture at the host-language level (python bytecode interception)
  see (Ansel et al. https://docs.pytorch.org/assets/pytorch2-2.pdf)
  """
  device: str = ""
  suffix: str = ""
  # TODO: make this generic with a list of supported types
  supports_float4: bool = True
  has_local: bool = True
  has_threads: bool = False
  has_shared: bool = True
  # NOTE: these two should be in (x,y,z) order to match the max_sizes argument in get_grouped_dims
  global_max: tuple[int, ...]|None = (0x8FFFFFFF,) * (3) # TODO: Ops.SPECIAL int32 indexes right now
  local_max: tuple[int, ...]|None = (0x8FFFFFFF,) * (3) # TODO: Ops.SPECIAL int32 indexes right now
  shared_max: int = 32768
  tensor_cores: list[TensorCore] = []
  pre_matcher: PatternMatcher|None = None
  extra_matcher: PatternMatcher|None = None
  code_for_op: dict[OpCode, Callable] = {}

  def __reduce__(self): return self.__class__, ()
  def render(self, uops:list[OpNode]) -> str: raise NotImplementedError("needs a renderer")