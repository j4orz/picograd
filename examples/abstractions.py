# inspired by https://github.com/tinygrad/tinygrad/blob/master/docs/abstractions2.py
# picograd is a deep learning framework that includes 1. a tensor DSL and a 2. heterogenous runtime
# 1. a "Device" that provides a runtime (buffer management, compilation, and running programs)
# 2. a "Op" that uses device runtimes and specifies compute in an abstract intermediate representation
# 3. a "Tensor" that provides array programming language sugar on top of Op with automatic differentiation ".backward()"

print("******** first, the runtime ***********")

from tinygrad.runtime.ops_hip import HIPDevice, HIPCompiler, HIPProgram

hip = HIPDevice()

# 1. allocate some buffers
out = hip.allocator.alloc(4)
a = hip.allocator.alloc(4)
b = hip.allocator.alloc(4)

# 2. load in some values (little endian)
hip.allocator._copyin(a, memoryview(bytearray([2,0,0,0])))
hip.allocator._copyin(b, memoryview(bytearray([3,0,0,0])))

# 3. compile a program to a binary
kernel = HIPCompiler().compile("void add(int *out, int *a, int *b) { out[0] = a[0] + b[0]; }")

 # 4. create a runtime for the program
f = hip.runtime("add", kernel)
f(out, a, b) # 5. run the program

print(val := hip.allocator._as_buffer(out).cast("I").tolist()[0])
assert val == 5 # check the data out

# print("******** second, the ir ***********")
# DEVICE = "CPU" # NOTE: you can change this! CL, HIP, CUDA

# import struct
# from picograd.dtype import dtypes
# from picograd.device import Buffer, Device
# from picograd.op import Op, OpCode

# # allocate some buffers + load in values
# out = Buffer(DEVICE, 1, dtypes.int32).allocate()
# a = Buffer(DEVICE, 1, dtypes.int32).allocate().copyin(memoryview(bytearray(struct.pack("I", 2))))
# b = Buffer(DEVICE, 1, dtypes.int32).allocate().copyin(memoryview(bytearray(struct.pack("I", 3))))
# # NOTE: a._buf is the same as the return from cpu.allocator.alloc

# # describe the computation (picograd eager specifies computation with specific eager kernels?)
# idx = Op.const(dtypes.index, 0)
# buf_1 = Op(OpCode.DEFINE_GLOBAL, dtypes.int32.ptr(), (), 1)
# buf_2 = Op(OpCode.DEFINE_GLOBAL, dtypes.int32.ptr(), (), 2)
# alu = buf_1.index(idx) + buf_2.index(idx)
# output_buf = Op(OpCode.DEFINE_GLOBAL, dtypes.int32.ptr(), (), 0)
# st_0 = Op(OpCode.STORE, dtypes.void, (output_buf.index(idx), alu))
# s = Op(OpCode.SINK, dtypes.void, (st_0,))

# print("******** third, the language ***********")
# from picograd import Tensor
# a, b = Tensor([2], dtype=dtypes.int32, device=DEVICE), Tensor([3], dtype=dtypes.int32, device=DEVICE)
# out = a + b

# print(val:=out.item()) # check the data out
# assert val == 5