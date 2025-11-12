# inspired by https://github.com/tinygrad/tinygrad/blob/master/docs/abstractions2.py
# picograd is a tensor library, and as a tensor library it has multiple parts
# 1. a "Device" that provides a runtime (buffer management, compilation, and running programs)
# 2. a "Op" that uses device runtimes and specifies compute in an abstract intermediate representation
# 3. a "Tensor" that provides array programming language sugar on top of Op with automatic differentiation ".backward()"

from picograd.runtime.cpu import ClangJITCompiler, CPUDevice, CPUProgram

print("******** first, the runtime ***********")
cpu = CPUDevice()

out, a, b = cpu.allocator.alloc(4), cpu.allocator.alloc(4), cpu.allocator.alloc(4) # allocate some buffers
cpu.allocator._copyin(a, memoryview(bytearray([2,0,0,0]))) # load in some values (little endian)
cpu.allocator._copyin(b, memoryview(bytearray([3,0,0,0])))

kernel = "void add(int *out, int *a, int *b) { out[0] = a[0] + b[0]; }"
lib = ClangJITCompiler().compile() # compile a program to a binary
fxn = cpu.runtime("add", lib) # create a runtime for the program
fxn(out, a, b) # run the program

print(val := cpu.allocator._as_buffer(out).cast("I").tolist()[0]) # check the data out
assert val == 5

print("******** second, the ir ***********")
DEVICE = "CPU" # NOTE: you can change this! CL, HIP, CUDA

import struct
from picograd.dtype import dtypes
from picograd.device import Buffer, Device
from picograd.op import Op, OpCode

# allocate some buffers + load in values
out = Buffer(DEVICE, 1, dtypes.int32).allocate()
a = Buffer(DEVICE, 1, dtypes.int32).allocate().copyin(memoryview(bytearray(struct.pack("I", 2))))
b = Buffer(DEVICE, 1, dtypes.int32).allocate().copyin(memoryview(bytearray(struct.pack("I", 3))))
# NOTE: a._buf is the same as the return from cpu.allocator.alloc

# describe the computation (picograd eager specifies computation with specific eager kernels?)
idx = Op.const(dtypes.index, 0)
buf_1 = Op(OpCode.DEFINE_GLOBAL, dtypes.int32.ptr(), (), 1)
buf_2 = Op(OpCode.DEFINE_GLOBAL, dtypes.int32.ptr(), (), 2)
alu = buf_1.index(idx) + buf_2.index(idx)
output_buf = Op(OpCode.DEFINE_GLOBAL, dtypes.int32.ptr(), (), 0)
st_0 = Op(OpCode.STORE, dtypes.void, (output_buf.index(idx), alu))
s = Op(OpCode.SINK, dtypes.void, (st_0,))

print("******** third, the language ***********")
from picograd import Tensor
a, b = Tensor([2], dtype=dtypes.int32, device=DEVICE), Tensor([3], dtype=dtypes.int32, device=DEVICE)
out = a + b

print(val:=out.item()) # check the data out
assert val == 5