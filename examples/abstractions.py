# inspired by https://github.com/tinygrad/tinygrad/blob/master/docs/abstractions2.py
print("******** first, the runtime with it's memory and compute management  ***********")
from picograd.runtime.hip_runtime import HIPDevice, HIPCCCompiler, HIPKernel
device = HIPDevice()

# 1. memory: allocate and memcpy on device
a, b, c = [device.allocator.alloc(4), device.allocator.alloc(4), device.allocator.alloc(4)]
device.allocator._copyin(a, memoryview(bytearray([2,0,0,0])))
device.allocator._copyin(b, memoryview(bytearray([3,0,0,0])))

# 2. compute: compile a kernel to a binary
kernel = HIPCCCompiler().compile("__global__ void add(int *a, int *b, int *c) { int id = blockDim.x * blockIdx.x + threadIdx.x; if(id < N) c[id] = a[id] + b[id]; }")

 # 3. launch
f = device.kernel("add", kernel)
f(a, b, c) # HIPKernel

print(val := device.allocator._as_buffer(c).cast("I").tolist()[0])
assert val == 5 # check the data out

print("******** second, the expression graph  ***********")
DEVICE = "HIP"

print("******** third, a sugared tensor  ***********")

from picograd import Tensor
from picograd.dtype import dtypes

x = Tensor([2], dtype=dtypes.int32, device=DEVICE)
y = Tensor([3], dtype=dtypes.int32, device=DEVICE)
z = x + y

# check the data out
print(val:=z.item())
assert val == 5
