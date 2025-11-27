import functools, ctypes, pathlib, hashlib, tempfile, subprocess
import gpuctypes.hip as hip
from picograd.helpers import DEBUG, OSX, system
from picograd.device import Allocator, BufferSpec, CompileError, Compiler, LRUAllocator, Runtime
# from picograd.runtime.cpu import LLVMCompiler

# **************** Python/C Foreign Function Helpers  ****************
"""
HIPDevice, HIPAllocator, and HIPKernel's uses the C/C++ hipamd runtime through foreign function library gpuctypes.hip
all hipamd function calls are wrapped with check(_) to assert success (see: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/error_codes.html#basic-runtime-errors)
and functions calls whose arguments need to be passed by reference are wrapped with f_by_ref
- ret_init_c_struct_t(_,_): ...
"""
def check(status):
  if status != 0: raise RuntimeError(f"HIP Error {status}, {ctypes.string_at(hip.hipGetErrorString(status)).decode()}")
def f_by_ref(ref, f): return (f(ref), ref)[1]
def mv_address(mv): return ctypes.addressof(ctypes.c_char.from_buffer(mv))

def init_c_struct_t(fields: tuple[tuple[str, type[ctypes._SimpleCData]], ...]):
  class CStruct(ctypes.Structure):
    _pack_, _fields_ = 1, fields
  return CStruct


# **************** Runtime: Host Allocators + Device Compilers ****************
class HIPDevice(Runtime):
  """
  picograd's HIPDevice(Runtime) is a thin python/c foreign function shim (this file is ~100loc)
  over vendor provided and implemented `hipamd` runtime and `hipcc` compiler.
  picograd's hip runtime stands in contrast to custom implemented tinygrad hardware command queue runtimes
  enabling features like egpu over usb, a valuable feature to applications such as openpilot.

  1. hip runtime api (accessed through tinygrad/gpuctypes, generated via trolldbois/ctypeslib)
      a. hip runtime documentation https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api.html
      b. hip runtime reference/headers https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api_reference.html https://github.com/ROCm/rocm-systems/blob/develop/projects/hip/include/hip/hip_runtime_api.h
      c. hip runtime source (hipamd), on nv devices, the hip runtime itself is a shim (hipother) over cuda runtime/driver apis https://github.com/ROCm/rocm-systems/blob/develop/projects/clr/README-doc.md https://github.com/ROCm/hipother
      d. hsa runtime (driven by kernel drivers "rocr runtime") https://github.com/ROCm/rocm-systems/tree/develop/projects/rocr-runtime
  2. the hipcc compiler driver (which in turn, calls clang or nvcc)
      a. hipcc documentation https://rocm.docs.amd.com/projects/HIPCC/en/latest/index.html
      b. hipcc source https://github.com/ROCm/llvm-project/tree/amd-staging/amd/hipcc
  """
  def __init__(self, device:str=""):
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    if DEBUG >= 1: print("initializing hipdevice(runtime) for device", self.device_id)
    self.arch = f_by_ref(hip.hipDeviceProp_t(), lambda x: check(hip.hipGetDeviceProperties(x, self.device_id))).gcnArchName.decode()
    # TODO (picograd profiling) self.time_event_st, self.time_event_en = [init_c_var(hip.hipEvent_t(), lambda x: hip.hipEventCreate(ctypes.byref(x), 0)) for _ in range(2)]

    compilers = [(functools.partial(HIPRenderer, self.arch), functools.partial(HIPCCCompiler, self.arch))] # MOOSE: renderer, fusion compiler pipeline
    super().__init__(device, HIPAllocator(self), compilers, functools.partial(HIPKernel, self))

  def synchronize(self):
    if DEBUG >= 1: print("synchronizing device", self.device_id)
    check(hip.hipSetDevice(self.device_id))
    check(hip.hipDeviceSynchronize())

# **************** Host Memory Allocation ****************
class HIPAllocator(Allocator[HIPDevice]):
  """
  ...
  """
  def _alloc(self, size:int, options:BufferSpec):
    check(hip.hipSetDevice(self.dev.device_id))
    return f_by_ref(hip.hipDeviceptr_t(), lambda x: check(hip.hipMalloc(ctypes.byref(x), size)))
  def _free(self, opaque, options:BufferSpec): check(hip.hipFree(opaque))
  def _copyin(self, dest, src: memoryview):
    check(hip.hipSetDevice(self.dev.device_id))
    check(hip.hipMemcpy(dest, mv_address(src), len(src), hip.hipMemcpyHostToDevice))
  def _copyout(self, dest:memoryview, src):
    self.dev.synchronize()
    check(hip.hipMemcpy(mv_address(dest), src, len(dest), hip.hipMemcpyDeviceToHost))

# **************** Device Kernel Compilation ****************
class HIPKernel:
  """
  ...
  """
  def __init__(self, dev:HIPDevice, name:str, lib:bytes):
    self.dev, self.name, self.lib = dev, name, lib
    check(hip.hipSetDevice(self.dev.device_id))
    self.module = f_by_ref(hip.hipModule_t(), lambda x: check(hip.hipModuleLoadData(ctypes.byref(x), lib)))
    self.prg = f_by_ref(hip.hipFunction_t(), lambda x: check(hip.hipModuleGetFunction(ctypes.byref(x), self.module, name.encode("utf-8"))))

  def __call__(self, *args, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    check(hip.hipSetDevice(self.dev.device_id))
    if not hasattr(self, "vargs"):
      self.c_args = init_c_struct_t(tuple([(f'f{i}', hip.hipDeviceptr_t) for i in range(len(args))] + [(f'v{i}', ctypes.c_int) for i in range(len(vals))]))(*args, *vals)
      self.vargs = (ctypes.c_void_p * 5)(1, ctypes.cast(ctypes.byref(self.c_args), ctypes.c_void_p), 2, ctypes.cast(ctypes.pointer(ctypes.c_size_t(ctypes.sizeof(self.c_args))), ctypes.c_void_p), 3)

    for i in range(len(args)): self.c_args.__setattr__(f'f{i}', args[i])
    for i in range(len(vals)): self.c_args.__setattr__(f'v{i}', vals[i])
    # if wait: check(hip.hipEventRecord(self.dev.time_event_st, None))
    check(hip.hipModuleLaunchKernel(self.prg, *global_size, *local_size, 0, None, None, self.vargs))

    # if wait:
    #   check(hip.hipEventRecord(self.dev.time_event_en, None))
    #   check(hip.hipEventSynchronize(self.dev.time_event_en))
    #   check(hip.hipEventElapsedTime(ctypes.byref(ret := ctypes.c_floaşt()), self.dev.time_event_st, self.dev.time_event_en))
    #   return ret.value * 1e-3
    
  def __del__(self):
    if hasattr(self, 'module'): check(hip.hipModuleUnload(self.module))

# TODO: (picograd in process comgr for jit)
# class HIPCOMGRCompiler(Compiler):

class HIPCCCompiler(Compiler):
  """
  ...
  """
  def __init__(self, arch:str, extra_options:list[str]=[]):
    self.arch, self.extra_options = arch, extra_options
    super().__init__(f"compile_hipcc_{self.arch}_{hashlib.sha256(' '.join(extra_options).encode()).hexdigest()[:8]}")
  def compile(self, src:str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".cpp") as srcf, tempfile.NamedTemporaryFile(suffix=".bc") as bcf:
      with tempfile.NamedTemporaryFile(suffix=".hsaco") as libf:
        srcf.write(src.encode())
        srcf.flush()

        subprocess.run(["hipcc", "-c", "-emit-llvm", "--cuda-device-only", "-O3", "-mcumode",
                        f"--offload-arch={self.arch}", "-I/opt/rocm/include/hip", "-o", bcf.name, srcf.name] + self.extra_options, check=True)
        subprocess.run(["hipcc", "-target", "amdgcn-amd-amdhsa", f"-mcpu={self.arch}",
                        "-O3", "-mllvm", "-amdgpu-internalize-symbols", "-c", "-o", libf.name, bcf.name] + self.extra_options, check=True)

        return pathlib.Path(libf.name).read_bytes()
  def disassemble(self, lib:bytes): amdgpu_disassemble(lib)

# TODO (picograd amdllvmcompiler) class AMDLLVMCompiler(LLVMCompiler):
def amdgpu_disassemble(lib:bytes):
  asm = system(f"{'/opt/homebrew/opt/llvm/bin/llvm-objdump' if OSX else '/opt/rocm/llvm/bin/llvm-objdump'} -d -", input=lib).splitlines()
  while asm and ("s_nop 0" in asm[-1] or "s_code_end" in asm[-1]): asm.pop()
  print("\n".join(asm))