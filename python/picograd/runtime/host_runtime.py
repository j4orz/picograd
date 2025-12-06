# **************** Runtime: Host Allocators + Device Compilers ****************
from typing import Any
import itertools, time, base64, pickle

from picograd.dtype import DType
from picograd.engine.irparser import GroupedOpCode, OpCode
from picograd.runtime.device import Allocator, Compiler, Runtime

class HostDevice(Runtime):
  def __init__(self, device:str):
    super().__init__(device, HostAllocator(self), [(HostRenderer, HostCompiler)], HostKernel)

# **************** Host Memory Allocation ****************
class HostAllocator(Allocator['HostDevice']):
  def _alloc(self, size, options):             return memoryview(bytearray(size))
  def _copyin(self, dest, src:memoryview):     dest[:] = src
  def _copyout(self, dest:memoryview, src):    dest[:] = src

# **************** Device Kernel Compilation ****************
def _emulate(opcode: OpCode, values: dict[int, Any], pbufs: list[memoryview], pvals: list[int]):
  raise NotImplementedError("todo")

class HostKernel:
  def __init__(self, name:str, lib:bytes):
    self.uops: list[tuple[OpCode, DType, list[int], Any]] = pickle.loads(lib)
  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    st = time.perf_counter()
    warp = list(itertools.product(*[range(x) for x in local_size[::-1]]))
    warp_size = len(warp)
    void_ops = {OpCode.END, OpCode.BARRIER, OpCode.IF, OpCode.ENDIF, OpCode.SINK, OpCode.NOOP, OpCode.GROUP, OpCode.STORE}
    loop_ends: dict[int, int] = {srcs[1]:i for i, (uop, _, srcs, _) in enumerate(self.uops) if uop == OpCode.END}
    for idxs in itertools.product(*[range(x) for x in global_size[::-1]]):
      values: dict[int, Any] = {}
      pbufs: list[memoryview] = list(bufs)
      pvals: list[int] = list(vals)
      i = 0
      while i < len(self.uops):
        uop, dtype, srcs, arg = self.uops[i]
        src_values = [values[v] for v in srcs if self.uops[v][0] not in void_ops]
        src_dtypes = [self.uops[v][1] for v in srcs if self.uops[v][0] not in void_ops]
        _emulate()
    return time.perf_counter() - st

class HostCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)