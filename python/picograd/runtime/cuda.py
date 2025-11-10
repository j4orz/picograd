import functools

from picograd.device import Compiler, Device
import gpuctypes.cuda as cuda

def check(status):
  if status != 0:
    error = ctypes.string_at(init_c_var(ctypes.POINTER(ctypes.c_char)(), lambda x: cuda.cuGetErrorString(status, ctypes.byref(x)))).decode()
    raise RuntimeError(f"CUDA Error {status}, {error}")

# def cu_time_execution(cb, enable=False) -> float|None:
#   if not enable: return cb()
#   evs = [init_c_var(cuda.CUevent(), lambda x: cuda.cuEventCreate(ctypes.byref(x), 0)) for _ in range(2)]
#   cuda.cuEventRecord(evs[0], None)
#   cb()
#   cuda.cuEventRecord(evs[1], None)
#   check(cuda.cuEventSynchronize(evs[1]))
#   cuda.cuEventElapsedTime(ctypes.byref(ret := ctypes.c_float()), evs[0], evs[1])
#   for ev in evs: cuda.cuEventDestroy_v2(ev)
#   return ret.value * 1e-3

class CUDADevice(Device):
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
    super().__init__(device, CUDAAllocator(self), compilers, functools.partial(CUDAProgram, self), None if MOCKGPU else CUDAGraph)

  def synchronize(self):
    check(cuda.cuCtxSetCurrent(self.context))
    check(cuda.cuCtxSynchronize())
    for opaque,sz,options in self.pending_copyin: self.allocator.free(opaque, sz, options)
    self.pending_copyin.clear()

  @staticmethod
  def synchronize_system():
    for d in CUDADevice.devices: d.synchronize()

# **************** Compilers ****************

class CUDACompiler(Compiler):
  def __init__(self, arch:str, cache_key:str="cuda"):
    self.arch, self.compile_options = arch, [f'--gpu-architecture={arch}']
    self.compile_options += [f"-I{CUDA_PATH}/include"] if CUDA_PATH else ["-I/usr/local/cuda/include", "-I/usr/include", "-I/opt/cuda/include"]
    nvrtc_check(nvrtc.nvrtcVersion((nvrtcMajor := ctypes.c_int()), (nvrtcMinor := ctypes.c_int())))
    if (nvrtcMajor.value, nvrtcMinor.value) >= (12, 4): self.compile_options.append("--minimal")
    super().__init__(f"compile_{cache_key}_{self.arch}")
  def _compile_program(self, src:str, nvrtc_get_content:Callable, nvrtc_get_size:Callable) -> bytes:
    nvrtc_check(nvrtc.nvrtcCreateProgram(ctypes.byref(prog := nvrtc.nvrtcProgram()), src.encode(), "<null>".encode(), 0, None, None))
    nvrtc_check(nvrtc.nvrtcCompileProgram(prog, len(self.compile_options), to_char_p_p([o.encode() for o in self.compile_options])), prog)
    data = _get_bytes(prog, nvrtc_get_content, nvrtc_get_size, nvrtc_check)
    nvrtc_check(nvrtc.nvrtcDestroyProgram(ctypes.byref(prog)))
    return data
  def compile(self, src:str) -> bytes: return self._compile_program(src, nvrtc.nvrtcGetPTX, nvrtc.nvrtcGetPTXSize)
  def disassemble(self, lib:bytes): cuda_disassemble(lib, self.arch)

class NVCompiler(CUDACompiler):
  def __init__(self, arch:str): super().__init__(arch, cache_key="nv")
  def compile(self, src:str) -> bytes: return self._compile_program(src, nvrtc.nvrtcGetCUBIN, nvrtc.nvrtcGetCUBINSize)

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

class PTXCompiler(Compiler):
  def __init__(self, arch:str, cache_key="ptx"):
    self.arch = arch
    super().__init__(f"compile_{cache_key}_{self.arch}")
  def compile(self, src:str) -> bytes:
    return src.replace("TARGET", self.arch).replace("VERSION", "8.7" if (ver:=int(self.arch[3:]))>=120 else ("7.8" if ver>=89 else "7.5")).encode()
  def disassemble(self, lib:bytes): cuda_disassemble(lib, self.arch)
class NVPTXCompiler(PTXCompiler):
  def __init__(self, arch:str):
    nvrtc_check(nvrtc.nvJitLinkVersion(ctypes.byref(ctypes.c_uint()), ctypes.byref(ctypes.c_uint())))
    super().__init__(arch, cache_key="nv_ptx")
  def compile(self, src:str) -> bytes:
    jitlink_check(nvrtc.nvJitLinkCreate(handle := nvrtc.nvJitLinkHandle(), 1, to_char_p_p([f'-arch={self.arch}'.encode()])), handle)
    jitlink_check(nvrtc.nvJitLinkAddData(handle, nvrtc.NVJITLINK_INPUT_PTX, ptxsrc:=super().compile(src), len(ptxsrc), "<null>".encode()), handle)
    jitlink_check(nvrtc.nvJitLinkComplete(handle), handle)
    data = _get_bytes(handle, nvrtc.nvJitLinkGetLinkedCubin, nvrtc.nvJitLinkGetLinkedCubinSize, jitlink_check)
    jitlink_check(nvrtc.nvJitLinkDestroy(handle))
    return data

# **************** Host Allocator, Device Kernels ****************
class CUDAAllocator(LRUAllocator['CUDADevice']):
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

class CUDAProgram:
  def __init__(self, dev:CUDADevice, name:str, lib:bytes, smem:int=0):
    self.dev, self.name, self.lib, self.smem = dev, name, lib, smem
    if DEBUG >= 5: print("\n".join([f"{i+1:>3} {line}" for i, line in enumerate(pretty_ptx(lib.decode('utf-8')).split("\n"))]))

    check(cuda.cuCtxSetCurrent(self.dev.context))
    self.module = cuda.CUmodule()
    status = cuda.cuModuleLoadData(ctypes.byref(self.module), lib)
    if status != 0:
      del self.module
      raise RuntimeError(f"module load failed with status code {status}: {cuda.cudaError_enum__enumvalues[status]}")
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
