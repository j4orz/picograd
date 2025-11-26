# inspired by https://github.com/tinygrad/tinygrad/blob/master/docs/abstractions2.py
print("******** first, the runtime ***********")

from picograd.runtime.hip_runtime import HIPDevice, HIPCCCompiler, HIPKernel

device = HIPDevice()

# 1. memory: allocate and memcpy on device
a, b, c = [device.allocator.alloc(4), device.allocator.alloc(4), device.allocator.alloc(4)]
device.allocator._copyin(a, memoryview(bytearray([2,0,0,0])))
device.allocator._copyin(b, memoryview(bytearray([3,0,0,0])))

# 2. compute: compile a kernel to a binary
kernel = HIPCCCompiler().compile("__global__ void add(int *a, int *b, int *c) { int id = blockDim.x * blockIdx.x + threadIdx.x; if(id < N) c[id] = a[id] + b[id]; }")

 # 3. launch: create a runtime and launch the kernel
f = device.runtime("add", kernel)
f(a, b, c)

print(val := device.allocator._as_buffer(c).cast("I").tolist()[0])
assert val == 5 # check the data out