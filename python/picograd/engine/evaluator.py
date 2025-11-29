from __future__ import annotations
import os
from typing import TYPE_CHECKING
# from picograd.device import Allocator
if TYPE_CHECKING: from picograd.sugar.tensor import Tensor
from picograd.engine import Op, OpCode, Pattern, PatternMatcher
# from . import _pgrs

# ************ f(x) ************  
def launch_add(x: Tensor, y: Tensor):
  raise NotImplementedError("")
  # out = cpu.allocator.alloc(4)
  # run_kernel("void kernel(int *out, int *a, int *b){ out[0]=a[0]+b[0]; }",
  #                   out, self.buf, other.buf)
  # raise NotImplementedError("")

def launch_mul(A: Tensor, B: Tensor): raise NotImplementedError("")
def launch_mm(A: Tensor, B: Tensor): raise NotImplementedError("")
  # if os.getenv("EAGER_NAIVE") == 1: # allocate/synchronize per op (no views)
  #   assert A.dtype == np.float32 and B.dtype == np.float32, "supports f32 only"
  #   assert A.ndim == 2 and B.ndim == 2, "expected 2D inputs"
  #   (M, K),  (K2, N) = (A.shape, B.shape)
  #   assert K == K2, f"shape mismatch: {A.shape} x {B.shape}"
  #   allocator = Allocator()

  #   _check(hip.hipSetDevice(0))
  #   bufA, bufB, bufC = allocator.alloc(A.nbytes), allocator.alloc(B.nbytes), allocator.alloc(M*N*4)
  #   allocator.to_device(bufA, A), allocator.to_device(bufB, B)

  #   grid, block  = ((N + 15)//16, (M + 15)//16, 1), (16, 16, 1)
  #   prog  = get_or_build_matmul()
  #   prog.launch([bufA, bufB, bufC], [M, N, K], grid, block)

  #   out = np.empty((M, N), dtype=np.float32)

  #   allocator.to_host(out, bufC)
  #   allocator.free(bufC); allocator.free(bufB); allocator.free(bufA)
  #   return out
  # elif os.getenv("EAGER_RUNTIME") == 1: # eager interpretation with runtime
  #   in_bufs = [input._buffer for input in (self,)+other] # 1. gather input bufs
  #   out_shp, out_dtype = out_tensor.shape, out_tensor.dtype.base
  #   out_uop = UOp.new_buffer(dev, prod(out_shp), out_dtype).reshape(out_shp) # 2. allocate output bufs
  #   out_buf = cast(Buffer, out_uop.base.buffer).allocate()
  #   eval_uop() # 2. dispatch op to eager interpreter
  #   return out_tensor

# ************ f'(x) ************  
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

def toposort(gate:Callable|None=None) -> dict[Op, None]:
  visited: dict[Op, None] = {}
  stack: list[tuple[Op, bool]] = [(self, False)] # each stack entry is (node, visited_flag)

  while stack:
    node, visited = stack.pop()
    if node in visited: continue
    if not visited:
      if gate is None or gate(node): # MOOSE gate?
        stack.append((node, True))  # push node back on stack to process after its srcs
        for s in reversed(node.inputs): stack.append((s, False)) # push srcs on the stack
    else: visited[node] = None # second time i'm seeing this node, add it to returned toposort
  return visited