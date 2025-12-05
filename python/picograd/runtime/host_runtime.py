# **************** Runtime: Host Allocators + Device Compilers ****************
from typing import Any
import itertools, time, base64, pickle

from python.picograd.dtype import DType
from python.picograd.engine.irparser import GroupedOpCode, OpCode
from python.picograd.runtime.device import Allocator, Compiler, Runtime

class HostDevice(Runtime):
  def __init__(self, device:str):
    super().__init__(device, HostAllocator(self), [(HostRenderer, HostCompiler)], HostKernel)

# **************** Host Memory Allocation ****************
class HostAllocator(Allocator['HostDevice']):
  def _alloc(self, size, options):             return memoryview(bytearray(size))
  def _copyin(self, dest, src:memoryview):     dest[:] = src
  def _copyout(self, dest:memoryview, src):    dest[:] = src

# **************** Device Kernel Compilation ****************
def _emulate(opcode: OpCode, values: dict[int, Any], pbufs: list[memoryview], pvals: list[int])
  if getenv("TRACE"): print(i, uop, dtype, arg, src_values, src_dtypes)
  if opcode is OpCode.END:
    i = srcs[1]
    return
  if opcode in (OpCode.BARRIER, OpCode.IF, OpCode.ENDIF, OpCode.SINK, OpCode.NOOP, OpCode.GROUP):
    # in the python emulator, the warp is always in sync
    i += 1
    return
  assert dtype is not None, f"{opcode} is missing a dtype"
  if opcode is OpCode.STORE:
    for j,val in enumerate(src_values[1] if src_dtypes[1].count > 1 else [src_values[1]]):
      for (m,o,g),v in zip(src_values[0], val):
        if g: _store(m, o+j, v, src_dtypes[1].scalar())
    i += 1
    return
  if opcode is OpCode.AFTER: values[i] = src_values[0]
  elif opcode in {OpCode.DEFINE_GLOBAL, OpCode.DEFINE_LOCAL, OpCode.DEFINE_REG}:
    assert isinstance(dtype, PtrDType), dtype
    storage_fmt = storage_fmt_for_dtype(dtype.base.scalar())
    if storage_fmt is None: raise RuntimeError(f"{dtype=} is not supported")
    if TYPE_CHECKING or sys.version_info < (3, 12): assert storage_fmt != "e"
    if opcode is OpCode.DEFINE_REG:
      # REGs are per thread
      values[i] = [memoryview(bytearray(dtype.size*dtype.itemsize)).cast(storage_fmt) for _ in range(warp_size)]
    else:
      buf = memoryview(bytearray(dtype.size*dtype.itemsize)) if opcode is not OpCode.DEFINE_GLOBAL else pbufs.pop(0)
      values[i] = [buf.cast(storage_fmt)] * warp_size
  elif opcode is OpCode.DEFINE_VAR:
    values[i] = [pvals.pop(0)] * warp_size
  elif opcode is OpCode.SPECIAL:
    if arg[0] == 'g': values[i] = [idxs[2-int(arg[-1])]] * warp_size
    elif arg[0] == 'l': values[i] = [x[2-int(arg[-1])] for x in warp]
  elif opcode is OpCode.CONST: values[i] = [arg] * warp_size
  elif opcode is OpCode.INDEX:
    ret:list = []
    if isinstance(src_dtypes[0], ImageDType):
      for m,ox,oy in zip(src_values[0], src_values[1][0], src_values[1][1]):
        if ox < 0 or ox >= src_dtypes[0].shape[1] or oy < 0 or oy >= src_dtypes[0].shape[0]: ret.append((m, None))
        else: ret.append((m, ox*4 + oy*src_dtypes[0].shape[1]*4))
    else:
      for m,o in zip(src_values[0], src_values[1]): ret.append((m,o))
    values[i] = [(m,o,g) for (m,o),g in zip(ret, src_values[2] if len(src_values) == 3 else [True]*len(ret))] # set the gate last
  elif opcode is OpCode.CAST and isinstance(dtype, PtrDType):
    values[i] = src_values[0]
  elif opcode is OpCode.RANGE:
    if i not in values: values[i] = [0] * warp_size
    else:
      for j in range(len(values[i])):
        values[i][j] += 1
    if values[i][0] == src_values[0][0]:
      del values[i]
      i = loop_ends[i] + 1
      return
  elif opcode is OpCode.VECTORIZE: values[i] = src_values
  elif opcode is OpCode.BITCAST:
    packed = struct.pack(str(warp_size) + storage_fmt_for_dtype(src_dtypes[0].scalar()),
                          *[to_storage_scalar(x, src_dtypes[0].scalar()) for x in src_values[0]])
    values[i] = list(struct.unpack(str(warp_size) +  storage_fmt_for_dtype(dtype.scalar()), packed))
    values[i] = [from_storage_scalar(x, dtype.scalar()) for x in values[i]]
  elif opcode is OpCode.CAST:
    values[i] = [truncate.get(dtype, lambda dt: dt)(dtypes.as_const(x, dtype)) for x in src_values[0]]
  elif opcode is OpCode.LOAD:
    if dtype.count > 1:
      values[i] = [load([src_values[i][j] if i != 0 and src_dtypes[i].count > 1 else src_values[i] \
                          for i in range(len(src_values))], j, dtype.scalar()) for j in range(dtype.count)]
    else:
      values[i] = load(src_values, 0, dtype)
  elif opcode is OpCode.GEP: values[i] = src_values[0][get_single_element(arg)]
  elif opcode is OpCode.WMMA:
    first_src_dtype = self.uops[srcs[0]][1]
    assert isinstance(first_src_dtype, DType) # mypy
    dims, dtype_in, device, threads = arg[1], first_src_dtype.scalar(), arg[4], arg[5]
    wmma_helper = functools.partial(generic_wmma_helper, src_values, warp_size)
    # TODO: refactor these to a shared TensorCoreLayout in kernel.py
    if device == "METAL":
      # A (2 elements on 32 threads): row major
      def a_b_elem(x, i, j, goff): return x[(i%2)][goff+(i//2)%2+(j%4)*2+(i//4)*8+(j//4)*16]
      # (i, j), C, D (2 elements on 32 threads): row major same as A/B
      def c_map(lane, elem): return (elem + ((lane%2)*2) + ((lane//8)%2)*4, ((lane//2)%4) + (lane//16)*4)
      values[i] = wmma_helper(32, 8, 2, 2, 2, a_b_elem, a_b_elem, c_map)
    elif device == "AMD" and threads == 64:
      def a_elem(x, k, row, goff): return x[k%(dims[2]//4)][goff + (k//(dims[2]//4))*16 + row]
      def b_elem(x, col, k, goff): return a_elem(x, k, col, goff)  # pylint: disable=arguments-out-of-order
      def c_map(lane, elem): return (lane%16, (lane//16)*4 + elem)
      values[i] = wmma_helper(64, dims[2], len(src_values[0]), len(src_values[1]), len(src_values[2]), a_elem, b_elem, c_map)
    elif device == "AMD" and len(src_values[0]) == 8: # RDNA4
      def a_elem(x, k, row, goff): return x[k - [0, 4, 4, 8][k//4]][goff + row + [0, 16, 0, 16][k//4]]
      def b_elem(x, col, k, goff): return a_elem(x, k, col, goff)
      def c_map(lane, elem): return (lane%16, (lane//16)*8 + elem)
      values[i] = wmma_helper(32, 16, 8, 8, 8, a_elem, b_elem, c_map)
    elif device == "AMD":
      # A (16 elements on 32 threads): col major, lane 16-32 == lane 0-15
      def a_elem(x, k, row, goff):
        assert x[k][goff+row] == x[k][goff+row+16], "warp elements not duplicated properly across lanes"
        return x[k][goff+row]
      # B (16 elements on 32 threads): row major, lane 16-32 == lane 0-15
      def b_elem(x, col, k, goff): return a_elem(x, k, col, goff)  # pylint: disable=arguments-out-of-order
      def c_map(lane, elem): return (lane%16, lane//16+elem*2) # (i, j), C, D (8 elements on 32 threads): row major
      values[i] = wmma_helper(32, 16, 16, 16, 8, a_elem, b_elem, c_map)
    elif device == "CUDA":
      # (col, row) given (lane, elem) for C & D (4 elements on 32 threads); shared by all tc shapes with M=16 N=8
      def c_map(lane, elem): return (elem%2 + (lane%4)*2, lane//4 + (elem//2)*8)

      if dims == (8,16,16):
        def a_elem(x, k, row, goff): return x[k%2 + (row//8)*2 + (k//8)*4][goff + (k//2)%4 + (row%8)*4]
        def b_elem(x, col, k, goff): return x[k%2 + (k//8)*2][goff + (k//2)%4 + col*4]
        values[i] = wmma_helper(32, 16, 8, 4, 4, a_elem, b_elem, c_map)

      elif dims == (8,16,32):
        def a_elem(x, k, row, goff): return x[k%4 + (row//8)*4 + (k//16)*8][goff + (k//4)%4 + (row%8)*4]
        def b_elem(x, col, k, goff): return x[k%4 + (k//16)*4][goff + (k//4)%4  + col*4]
        values[i] = wmma_helper(32, 32, 16, 8, 4, a_elem, b_elem, c_map)

      elif dims == (8,16,8) and dtype_in == dtypes.half:
        def a_elem(x, k, row, goff): return x[k%2 + (row//8)*2][goff + k//2 + (row%8)*4]
        def b_elem(x, col, k, goff): return x[k%2][goff + k//2 + col*4]
        values[i] = wmma_helper(32, 8, 4, 2, 4, a_elem, b_elem, c_map)

      elif dims == (8,16,8) and dtype_in == dtypes.float:
        def a_elem(x, k, row, goff): return x[(k//4)*2 + row//8][goff + k%4 + (row%8)*4]
        def b_elem(x, col, k, goff): return x[k//4][goff + k%4 + col*4]
        values[i] = wmma_helper(32, 8, 4, 2, 4, a_elem, b_elem, c_map)

      else: raise NotImplementedError(f"unimplemented tensor core {arg}")
    elif device == "INTEL":
      # A (16 elements on 8 threads)
      def a_elem(x, k, row, goff): return x[k%2+row*2][goff+k//2]
      # B (16 elements on 8 threads)
      def b_elem(x, col, k, goff): return x[k][goff+col]
      # C, D (8 elements on 8 threads)
      def c_map(lane, elem): return (lane, elem)
      values[i] = wmma_helper(8, 16, 16, 16, 8, a_elem, b_elem, c_map)
    elif device == "CPU":
      def elem(x, col, row, _): return x[col+row][0] # k is always 0
      def c_map(lane, elem): return (elem%16, elem//16)
      values[i] = wmma_helper(1, 1, 16, 16, 256, elem, elem, c_map)
    else: raise NotImplementedError(f"unimplemented tensor core {arg}")
  elif opcode in GroupedOpCode.ALU:
    assert all_same([len(x) for x in src_values]), f"{[len(x) for x in src_values]} doesn't match on {uop}"
    assert all_same([dtype] + src_dtypes) or uop in {*GroupOp.Comparison, OpCode.WHERE}, f"dtype mismatch on {uop}"
    values[i] = [exec_alu(uop, dtype, p) for p in zip(*src_values)]
  assert i in values, (uop, dtype, srcs, arg)
  i += 1

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