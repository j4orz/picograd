from .tensor import Tensor

class Linear:
  def __init__(self, n, m, bias=True):
    self.W_nm = Tensor.randn((n, m), generator=g) * (5/3)/n**0.5 # kaiming init (He et al. 2015)
    self.b_m = Tensor.zeros(m) if bias else None

  def __call__(self, x_n):
    self.y_m = x_n @ self.W_nm
    if self.b_m is not None: self.y_m += self.b_m
    return self.y_m

  def parameters(self):
    return [self.W_nm] + ([] if self.b_m is None else [self.b_m])

__all__ = ["Linear", "BatchNorm"]