import numpy as np
import torch
from teenygrad import InterpretedTensor

def test_add():
  a, b = InterpretedTensor.ones((3, 4)), InterpretedTensor.ones((3, 4))
  a_np, b_np = np.ones((3, 4)), np.ones((3, 4))
  c, c_np = a + b, a_np+b_np

  assert c.shape == (3, 4)
  assert c.storage == [float(x) for x in c_np.flatten()]

def test_gemm():
  a, b = InterpretedTensor.arange(12).reshape((3,4)), InterpretedTensor.arange(20).reshape((4,5))
  a_np, b_np = np.arange(12.0).reshape((3,4)), np.arange(20.0).reshape((4,5))
  c, c_np = a @ b, a_np @ b_np

  assert c.shape == (3, 5)
  assert c.storage == [float(x) for x in c_np.flatten()]

def test_backward_scalar():
  x_pt = torch.tensor(3.0, requires_grad=True)
  y_pt = x_pt*x_pt
  y_pt.backward()

  x = InterpretedTensor((1,), [3.0], requires_grad=True)
  y = x * x
  y.backward()

  # f:R->R       f':R->R
  # f(x)=x^2 ==> f'(x)=2x
  #   x =3   ==> f'(x)=6
  assert x.grad.storage == [x_pt.grad.item()]