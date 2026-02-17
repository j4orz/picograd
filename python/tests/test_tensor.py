import numpy as np
import torch
from teenygrad import InterpretedTensor

def test_add():
  a, b = InterpretedTensor.ones((3, 4)), InterpretedTensor.ones((3, 4))
  a_np, b_np = np.ones((3, 4)), np.ones((3, 4))
  c, c_np = a + b, a_np+b_np

  assert c.shape == (3, 4)
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

def test_gemm_forward():
  a, b = InterpretedTensor.arange(12).reshape((3,4)), InterpretedTensor.arange(20).reshape((4,5))
  a_np, b_np = np.arange(12.0).reshape((3,4)), np.arange(20.0).reshape((4,5))
  c, c_np = a @ b, a_np @ b_np

  assert c.shape == (3, 5)
  assert c.storage == [float(x) for x in c_np.flatten()]

def test_tanh():
  x, x_np = InterpretedTensor.arange(12).reshape((3,4)), np.arange(12.0, dtype=np.float32).reshape((3,4))
  y, y_np = x.tanh(), np.tanh(x_np)

  assert y.storage == [float(x) for x in y_np.flatten()]

def test_gemm_backward():
  a_pt, b_pt = torch.arange(12.0).reshape(3,4).requires_grad_(True), torch.arange(20.0).reshape(4,5).requires_grad_(True)
  c_pt = a_pt @ b_pt # f: R^n->R^m, f(x):= GEMM(x)
  l = c_pt.sum() # f○g: R^n->R, g(x):= sum(x)
  l.backward() # .backward() evaluates grad, and requires output y∈R

  a, b = InterpretedTensor.arange(12, requires_grad=True).reshape((3,4)), InterpretedTensor.arange(20, requires_grad=True).reshape((4,5))
  c = a @ b
  c.backward()

  assert a.grad.storage == [float(x) for x in a_pt.grad.flatten()] # dL/dA = dL/dC @ B^T
  assert b.grad.storage == [float(x) for x in b_pt.grad.flatten()] # dL/dB = A^T @ dL/dC

def test_linear_forward():
  torch.manual_seed(42)
  n,m,b = 4, 3, 2

  linear_NMpt = torch.nn.Linear(n,m,bias=False)
  x_BNpt = torch.randn(b,n)
  y_BMpt = linear_NMpt(x_BNpt)

  from teenygrad.frontend.nn import Linear
  weight = InterpretedTensor((m,n), linear_NMpt.weight.detach().numpy().flatten().tolist())
  linear_NM = Linear(n,m,weight=weight, bias=False)
  x_BN = InterpretedTensor((b,n), x_BNpt.detach().numpy().flatten().tolist())
  y_BM = linear_NM(x_BN)

  np.testing.assert_allclose(y_BM.storage, [float(v) for v in y_BMpt.detach().numpy().flatten()], rtol=1e-5, atol=1e-5)

def test_tanh_backward():
  x_pt = torch.arange(12.0, dtype=torch.float32).reshape(3,4).requires_grad_(True)
  y_pt = x_pt.tanh()
  y_pt.sum().backward()

  x = InterpretedTensor.arange(12, requires_grad=True).reshape((3,4))
  y = x.tanh()
  y.backward()

  np.testing.assert_allclose(x.grad.storage, [float(v) for v in x_pt.grad.flatten()], rtol=1e-5, atol=1e-6) # d/dx tanh(x) = 1 - tanh(x)^2