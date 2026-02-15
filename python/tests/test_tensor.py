import numpy as np
from teenygrad import InterpretedTensor

def test_add():
  a, b = InterpretedTensor.ones((3, 4)), InterpretedTensor.ones((3, 4))
  a_np, b_np = np.ones((3, 4)), np.ones((3, 4))
  c, c_np = a + b, a_np + b_np

  assert c.shape == (3, 4)
  assert c.storage == [float(x) for x in c_np.flatten()]

def test_gemm():
  a, b = InterpretedTensor.arange(12).reshape((3,4)), InterpretedTensor.arange(20).reshape((4,5))
  a_np, b_np = np.arange(12.0).reshape((3,4)), np.arange(20.0).reshape((4,5))
  c, c_np = a @ b, a_np @ b_np

  assert c.shape == (3, 5)
  assert c.storage == [float(x) for x in c_np.flatten()]