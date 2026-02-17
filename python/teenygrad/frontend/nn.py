from .tensor import InterpretedTensor

class Linear:
  def __init__(self, n: int, m: int, weight=None, bias=True):
    self.n, self.m = n, m
    self.W_MN = weight if weight is not None else InterpretedTensor((m, n), [0.0]*(m*n))
    self.b_M = InterpretedTensor((m,), [0.0]*m) if bias else None

  def __call__(self, x_BN: InterpretedTensor):
    y_M = x_BN @ self.W_MN.T
    if self.b_M is not None: y_M = y_M + self.b_M
    return y_M