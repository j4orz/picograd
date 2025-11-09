from __future__ import annotations
import os
from typing import TYPE_CHECKING
# from picograd.device import Allocator
from picograd.op import OpCode
if TYPE_CHECKING: from picograd.tensor import Tensor

def eval_uop(inputs, opcode) -> Tensor:
  """
  the eager evaluator is an embedded interpreter which override the semantics of the host language
  since inputs are values they need to be dynamically destructured
  TODO: dispatcher, registry?
  """
  match opcode:
    case OpCode.NEG: raise NotImplementedError("todo")
    case OpCode.ADD: launch_add(A, B)
    case OpCode.MUL: launch_mul(A, B)
    case OpCode.MM: launch_mm(A, B)
    case OpCode.RECIPROCAL: raise NotImplementedError("todo")
    case OpCode.EXP2: raise NotImplementedError("todo")
    case OpCode.LOG2: raise NotImplementedError("todo")
    case OpCode.SIN: raise NotImplementedError("todo")
    case _: raise NotImplementedError(f"unsupported opcode {opcode!r}")

def launch_add(A: Tensor, B: Tensor): raise NotImplementedError("")
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
