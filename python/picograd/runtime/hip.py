import ctypes
import functools
import gpuctypes.hip as hip
from picograd.device import BufferSpec, CompileError, Compiler, Device, LRUAllocator
from picograd.runtime.cpu import LLVMCompiler

def check(status):
  if status != 0: raise RuntimeError(f"HIP Error {status}, {ctypes.string_at(hip.hipGetErrorString(status)).decode()}")

class HIPDevice(Device):
  def __init__(self, device:str=""):
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.arch = init_c_var(hip.hipDeviceProp_t(), lambda x: check(hip.hipGetDeviceProperties(x, self.device_id))).gcnArchName.decode()
    self.time_event_st, self.time_event_en = [init_c_var(hip.hipEvent_t(), lambda x: hip.hipEventCreate(ctypes.byref(x), 0)) for _ in range(2)]

    compilers = [(functools.partial(HIPRenderer, self.arch), functools.partial(HIPCompiler, self.arch))]
    super().__init__(device, HIPAllocator(self), compilers, functools.partial(HIPKernel, self))
  def synchronize(self):
    check(hip.hipSetDevice(self.device_id))
    check(hip.hipDeviceSynchronize())

# **************** Host Allocator, Device Kernels ****************
class HIPAllocator(LRUAllocator[HIPDevice]):
  def _alloc(self, size:int, options:BufferSpec):
    check(hip.hipSetDevice(self.dev.device_id))
    return init_c_var(hip.hipDeviceptr_t(), lambda x: check(hip.hipMalloc(ctypes.byref(x), size)))
  def _free(self, opaque, options:BufferSpec): check(hip.hipFree(opaque))
  def _copyin(self, dest, src: memoryview):
    check(hip.hipSetDevice(self.dev.device_id))
    check(hip.hipMemcpy(dest, mv_address(src), len(src), hip.hipMemcpyHostToDevice))
  def _copyout(self, dest:memoryview, src):
    self.dev.synchronize()
    check(hip.hipMemcpy(mv_address(dest), src, len(dest), hip.hipMemcpyDeviceToHost))

class HIPKernel:
  def __init__(self, dev:HIPDevice, name:str, lib:bytes):
    self.dev, self.name, self.lib = dev, name, lib
    check(hip.hipSetDevice(self.dev.device_id))
    self.module = init_c_var(hip.hipModule_t(), lambda x: check(hip.hipModuleLoadData(ctypes.byref(x), lib)))
    self.prg = init_c_var(hip.hipFunction_t(), lambda x: check(hip.hipModuleGetFunction(ctypes.byref(x), self.module, name.encode("utf-8"))))

  def __call__(self, *args, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    check(hip.hipSetDevice(self.dev.device_id))
    if not hasattr(self, "vargs"):
      self.c_args = init_c_struct_t(tuple([(f'f{i}', hip.hipDeviceptr_t) for i in range(len(args))] + [(f'v{i}', ctypes.c_int) for i in range(len(vals))]))(*args, *vals)
      self.vargs = (ctypes.c_void_p * 5)(1, ctypes.cast(ctypes.byref(self.c_args), ctypes.c_void_p), 2, ctypes.cast(ctypes.pointer(ctypes.c_size_t(ctypes.sizeof(self.c_args))), ctypes.c_void_p), 3)

    for i in range(len(args)): self.c_args.__setattr__(f'f{i}', args[i])
    for i in range(len(vals)): self.c_args.__setattr__(f'v{i}', vals[i])
    if wait: check(hip.hipEventRecord(self.dev.time_event_st, None))
    check(hip.hipModuleLaunchKernel(self.prg, *global_size, *local_size, 0, None, None, self.vargs))

    if wait:
      check(hip.hipEventRecord(self.dev.time_event_en, None))
      check(hip.hipEventSynchronize(self.dev.time_event_en))
      check(hip.hipEventElapsedTime(ctypes.byref(ret := ctypes.c_floaşt()), self.dev.time_event_st, self.dev.time_event_en))
      return ret.value * 1e-3
    
  def __del__(self):
    if hasattr(self, 'module'): check(hip.hipModuleUnload(self.module))

# **************** Compilers ****************
class HIPCompiler(Compiler):
  def __init__(self, arch:str):
    self.arch = arch
    super().__init__(f"compile_hip_{self.arch}")
  def compile(self, src:str) -> bytes:
    try: return compile_hip(src, self.arch, src.split('\n', 1)[0].strip() == '.text')
    except RuntimeError as e: raise CompileError(e) from e
  def disassemble(self, lib:bytes): amdgpu_disassemble(lib)

class AMDLLVMCompiler(LLVMCompiler):
  jit = False
  target_arch = "AMDGPU"
  def __init__(self, arch: str):
    self.arch = arch
    super().__init__(self.arch, "+cumode")
  def __reduce__(self): return (AMDLLVMCompiler, (self.arch,))
  def compile(self, src:str) -> bytes:
    try: return super().compile(src)
    except RuntimeError as e:
      if "undefined value '@llvm.amdgcn." in str(e): raise CompileError(str(e) + "AMD with LLVM backend requires LLVM >= 18") from e
      raise CompileError(e) from e
  def disassemble(self, lib:bytes): amdgpu_disassemble(lib)

# AMD_COMGR_SAVE_TEMPS=1 AMD_COMGR_REDIRECT_LOGS=stdout AMD_COMGR_EMIT_VERBOSE_LOGS=1
def compile_hip(prg:str, arch="gfx1100", asm=False) -> bytes:
  check(comgr.amd_comgr_create_action_info(ctypes.byref(action_info := comgr.amd_comgr_action_info_t())))
  check(comgr.amd_comgr_action_info_set_language(action_info, comgr.AMD_COMGR_LANGUAGE_HIP))
  check(comgr.amd_comgr_action_info_set_isa_name(action_info, b"amdgcn-amd-amdhsa--" + arch.encode()))
  check(comgr.amd_comgr_action_info_set_logging(action_info, True))

  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_src := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_bc := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_reloc := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_exec := comgr.amd_comgr_data_set_t())))

  check(comgr.amd_comgr_create_data(comgr.AMD_COMGR_DATA_KIND_SOURCE, ctypes.byref(data_src := comgr.amd_comgr_data_t())))
  check(comgr.amd_comgr_set_data(data_src, len(rprg := prg.encode()), rprg))

  if asm:
    check(comgr.amd_comgr_set_data_name(data_src, b"<null>.s"))
    check(comgr.amd_comgr_data_set_add(data_set_src, data_src))
    status = comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE, action_info, data_set_src, data_set_reloc)
    if status != 0:
      print(_get_comgr_data(data_set_reloc, comgr.AMD_COMGR_DATA_KIND_LOG).decode())
      raise RuntimeError("assemble failed")
  else:
    check(comgr.amd_comgr_set_data_name(data_src, b"<null>"))
    check(comgr.amd_comgr_data_set_add(data_set_src, data_src))
    # -include hiprtc_runtime.h was removed
    options = [
      "-O3", "-mcumode", "--hip-version=6.0.32830", "-DHIP_VERSION_MAJOR=6", "-DHIP_VERSION_MINOR=0", "-DHIP_VERSION_PATCH=32830",
      "-D__HIPCC_RTC__", "-std=c++14", "-nogpuinc", "-Wno-gnu-line-marker", "-Wno-missing-prototypes", f"--offload-arch={arch}",
      "-I/opt/rocm/include", "-Xclang -disable-llvm-passes", "-Xclang -aux-triple", "-Xclang x86_64-unknown-linux-gnu"]
    check(set_options(action_info, ' '.join(options).encode()))
    status = comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC, action_info, data_set_src, data_set_bc)
    if status != 0:
      print(_get_comgr_data(data_set_bc, comgr.AMD_COMGR_DATA_KIND_LOG).decode())
      raise RuntimeError("compile failed")
    check(set_options(action_info, b"-O3 -mllvm -amdgpu-internalize-symbols"))
    check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, action_info, data_set_bc, data_set_reloc))

  check(set_options(action_info, b""))
  check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, action_info, data_set_reloc, data_set_exec))
  ret = _get_comgr_data(data_set_exec, comgr.AMD_COMGR_DATA_KIND_EXECUTABLE)
  check(comgr.amd_comgr_release_data(data_src))
  for x in [data_set_src, data_set_bc, data_set_reloc, data_set_exec]: check(comgr.amd_comgr_destroy_data_set(x))
  check(comgr.amd_comgr_destroy_action_info(action_info))
  return ret