from __future__ import annotations
import re
import functools, ctypes, hashlib, pathlib, subprocess, tempfile
import gpuctypes.hip as cuda
from picograd.device import Allocator, BufferSpec, Compiler, CompilerPairT, Runtime
from picograd.helpers import DEBUG, colored, init_c_struct_t, init_c_var, mv_address, suppress_finalizing, system

# **************** Python/C Foreign Function Helpers  ****************
"""
...
"""
def mv_address(mv): return ctypes.addressof(ctypes.c_char.from_buffer(mv))
def unwrap_class_type(cls_t): return cls_t.func if isinstance(cls_t, functools.partial) else cls_t
def check(status):
  if status != 0: raise RuntimeError(f"HIP Error {status}, {ctypes.string_at(hip.hipGetErrorString(status)).decode()}")
def init_c_var(ctypes_var, init_cb): return (init_cb(ctypes_var), ctypes_var)[1]
def init_c_struct_t(fields: tuple[tuple[str, type[ctypes._SimpleCData]], ...]):
  class CStruct(ctypes.Structure):
    _pack_, _fields_ = 1, fields
  return CStruct
def encode_args(args, vals) -> tuple[ctypes.Structure, ctypes.Array]:
  c_args = init_c_struct_t(tuple([(f'f{i}', cuda.CUdeviceptr_v2) for i in range(len(args))] +
                                 [(f'v{i}', ctypes.c_int) for i in range(len(vals))]))(*args, *vals)
  vargs = (ctypes.c_void_p * 5)(ctypes.c_void_p(1), ctypes.cast(ctypes.byref(c_args), ctypes.c_void_p), ctypes.c_void_p(2),
                                ctypes.cast(ctypes.pointer(ctypes.c_size_t(ctypes.sizeof(c_args))), ctypes.c_void_p), ctypes.c_void_p(0))
  return c_args, vargs


# **************** Runtime: Host Allocators + Device Compilers ****************  
class CUDADevice(Runtime):
  devices: list[CUDADevice] = []
  peer_access = False

  def __init__(self, device:str):
    device_id = int(device.split(":")[1]) if ":" in device else 0
    check(cuda.cuInit(0))
    self.cu_device = init_c_var(cuda.CUdevice(), lambda x: check(cuda.cuDeviceGet(ctypes.byref(x), device_id)))
    self.context = init_c_var(cuda.CUcontext(), lambda x: check(cuda.cuCtxCreate_v2(ctypes.byref(x), 0, self.cu_device)))
    check(cuda.cuDeviceComputeCapability(ctypes.byref(major := ctypes.c_int()), ctypes.byref(minor := ctypes.c_int()), device_id))

    for dev in CUDADevice.devices:
      check(cuda.cuDeviceCanAccessPeer(ctypes.byref(val := ctypes.c_int()), self.cu_device, dev.cu_device))
      if val.value != 1: continue
      check(cuda.cuCtxSetCurrent(dev.context))
      check(cuda.cuCtxEnablePeerAccess(self.context, 0))
      check(cuda.cuCtxSetCurrent(self.context))
      check(cuda.cuCtxEnablePeerAccess(dev.context, 0))
      CUDADevice.peer_access = True

    self.arch = f"sm_{major.value}{minor.value}"
    self.pending_copyin: list[tuple[int, int, BufferSpec|None]] = []
    CUDADevice.devices.append(self)

    from tinygrad.runtime.graph.cuda import CUDAGraph
    compilers:list[CompilerPairT] = [(functools.partial(CUDARenderer, self.arch), functools.partial(CUDACompiler, self.arch)),
                                     (functools.partial(PTXRenderer, self.arch), functools.partial(PTXCompiler, self.arch)),
                                     (functools.partial(CUDARenderer, self.arch), functools.partial(NVCCCompiler, self.arch))]
    super().__init__(device, CUDAAllocator(self), compilers, functools.partial(CUDAKernel, self), None if MOCKGPU else CUDAGraph)

  def synchronize(self):
    check(cuda.cuCtxSetCurrent(self.context))
    check(cuda.cuCtxSynchronize())
    for opaque,sz,options in self.pending_copyin: self.allocator.free(opaque, sz, options)
    self.pending_copyin.clear()

  @staticmethod
  def synchronize_system():
    for d in CUDADevice.devices: d.synchronize()

# **************** Host Memory Allocation ****************
class CUDAAllocator(Allocator['CUDADevice']):
  def _alloc(self, size, options:BufferSpec):
    check(cuda.cuCtxSetCurrent(self.dev.context))
    if options.external_ptr: return cuda.CUdeviceptr_v2(options.external_ptr)
    if options.host: return init_c_var(ctypes.c_void_p(), lambda x: check(cuda.cuMemHostAlloc(ctypes.byref(x), size, 0x01)))
    return init_c_var(cuda.CUdeviceptr(), lambda x: check(cuda.cuMemAlloc_v2(ctypes.byref(x), size)))
  def _free(self, opaque, options:BufferSpec):
    try:
      if options.host: check(cuda.cuMemFreeHost(opaque))
      else: check(cuda.cuMemFree_v2(opaque))
    except (TypeError, AttributeError): pass
  def _copyin(self, dest, src:memoryview):
    check(cuda.cuCtxSetCurrent(self.dev.context))
    host_mem = self.alloc(len(src), BufferSpec(host=True))
    self.dev.pending_copyin.append((host_mem, len(src), BufferSpec(host=True)))
    ctypes.memmove(host_mem, mv_address(src), len(src))
    check(cuda.cuMemcpyHtoDAsync_v2(dest, host_mem, len(src), None))
  def _copyout(self, dest:memoryview, src):
    CUDADevice.synchronize_system()
    check(cuda.cuCtxSetCurrent(self.dev.context))
    check(cuda.cuMemcpyDtoH_v2(mv_address(dest), src, len(dest)))
  def _transfer(self, dest, src, sz:int, src_dev, dest_dev):
    check(cuda.cuCtxSetCurrent(src_dev.context))
    check(cuda.cuEventCreate(ctypes.byref(sync_event := cuda.CUevent()), 0))
    check(cuda.cuMemcpyDtoDAsync_v2(dest, src, sz, None))
    check(cuda.cuEventRecord(sync_event, None))
    check(cuda.cuCtxSetCurrent(dest_dev.context))
    check(cuda.cuStreamWaitEvent(None, sync_event, 0)) # sync the default stream on the dest dev
  def _offset(self, buf, size:int, offset:int): return cuda.CUdeviceptr_v2(buf.value + offset)


# **************** Device Kernel Compilation ****************
class CUDAKernel:
  def __init__(self, dev:CUDADevice, name:str, lib:bytes, smem:int=0):
    self.dev, self.name, self.lib, self.smem = dev, name, lib, smem
    if DEBUG >= 5: print("\n".join([f"{i+1:>3} {line}" for i, line in enumerate(pretty_ptx(lib.decode('utf-8')).split("\n"))]))

    check(cuda.cuCtxSetCurrent(self.dev.context))
    self.module = cuda.CUmodule()
    status = cuda.cuModuleLoadData(ctypes.byref(self.module), lib)
    if status != 0:
      del self.module
      raise RuntimeError(f"module load failed with status code {status}: {cuda.CUresult.get(status)}")
    check(cuda.cuModuleGetFunction(ctypes.byref(prg := cuda.CUfunction()), self.module, name.encode("utf-8")))
    self.prg = prg
    if self.smem > 0: check(cuda.cuFuncSetAttribute(self.prg, cuda.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, self.smem))

  @suppress_finalizing
  def __del__(self): check(cuda.cuModuleUnload(self.module))

  def __call__(self, *args, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    check(cuda.cuCtxSetCurrent(self.dev.context))
    if not hasattr(self, "vargs"):
      self.c_args, self.vargs = encode_args(args, vals)

      # HACK: For MOCKGPU send the args struct itself.
      if MOCKGPU: self.vargs = self.c_args # type: ignore[assignment]
    else:
      for i in range(len(args)): self.c_args.__setattr__(f'f{i}', args[i])
      for i in range(len(vals)): self.c_args.__setattr__(f'v{i}', vals[i])
    return cu_time_execution(lambda: check(cuda.cuLaunchKernel(self.prg, *global_size, *local_size, self.smem, None, None, self.vargs)), enable=wait)
  
# TODO: (picograd in process nvrtc for jit)
# class CUDARTCCompiler(Compiler):
# class NVCompiler(CUDARTCCompiler):

class NVCCCompiler(Compiler):
  def __init__(self, arch:str, extra_options:list[str]=[]):
    self.arch, self.extra_options = arch, extra_options
    super().__init__(f"compile_nvcc_{self.arch}_{hashlib.sha256(' '.join(extra_options).encode()).hexdigest()[:8]}")
  def compile(self, src:str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".cu") as srcf, tempfile.NamedTemporaryFile(suffix=".ptx") as libf:
      srcf.write(src.encode())
      srcf.flush()
      subprocess.run(["nvcc", f"-arch={self.arch}", "-ptx", "-o", libf.name, srcf.name] + self.extra_options,
                 check=True)
      return libf.read()
  def disassemble(self, lib:bytes): cuda_disassemble(lib, self.arch)

# TODO: (picograd ptx compiler)
# class PTXCompiler(Compiler):
# class NVPTXCompiler(PTXCompiler):

def cuda_disassemble(lib:bytes, arch:str):
  try:
    fn = (pathlib.Path(tempfile.gettempdir()) / f"tinycuda_{hashlib.md5(lib).hexdigest()}").as_posix()
    with open(fn, "wb") as f: f.write(lib)
    subprocess.run(["ptxas", f"-arch={arch}", "-o", fn, fn], check=False, stderr=subprocess.DEVNULL) # optional ptx -> sass step for CUDA=1
    print(system(f'nvdisasm {fn}'))
  except Exception as e: print("Failed to generate SASS", str(e), "Make sure your PATH contains ptxas/nvdisasm binary of compatible version.")

def pretty_ptx(s):
  # all expressions match `<valid_before><expr><valid_after>` and replace it with `<valid_before>color(<expr>)<valid_after>`
  s = re.sub(r'([!@<\[\s,\+\-;\n])((?:[_%$][\w%\$_]+(?:\.[xyz])?\:?)|(?:buf\d+))([<>\]\s,\+\-;\n\)])',
             lambda m:m[1]+colored(m[2], "blue")+m[3], s, flags=re.M) # identifiers
  s = re.sub(r'(.)((?:b|s|u|f)(?:8|16|32|64)|pred)([\.\s])', lambda m:m[1]+colored(m[2], "green")+m[3], s, flags=re.M) # types
  s = re.sub(r'^(\s*)([\w]+)(.*?;$)', lambda m:m[1]+colored(m[2], "yellow")+m[3], s, flags=re.M) # instructions
  s = re.sub(r'([<>\[\]\s,\+\-;])((?:0[fF][0-9a-fA-F]{8})|(?:[0-9]+)|(?:0[xX][0-9a-fA-F]+))([<>\[\]\s,\+\-;])',
             lambda m:m[1]+colored(m[2], "yellow")+m[3], s, flags=re.M) # numbers
  s = re.sub(r'(\.)(param|reg|global)', lambda m:m[1]+colored(m[2], "magenta"), s, flags=re.M) # space
  s = re.sub(r'(\.)(version|target|address_size|visible|entry)', lambda m:m[1]+colored(m[2], "magenta"), s, flags=re.M) # derivatives
  return s

def cuda_disassemble(lib:bytes, arch:str):
  try:
    fn = (pathlib.Path(tempfile.gettempdir()) / f"tinycuda_{hashlib.md5(lib).hexdigest()}").as_posix()
    with open(fn, "wb") as f: f.write(lib)
    subprocess.run(["ptxas", f"-arch={arch}", "-o", fn, fn], check=False, stderr=subprocess.DEVNULL) # optional ptx -> sass step for CUDA=1
    print(system(f'nvdisasm {fn}'))
  except Exception as e: print("Failed to generate SASS", str(e), "Make sure your PATH contains ptxas/nvdisasm binary of compatible version.")