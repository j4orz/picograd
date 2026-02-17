from .tensor import InterpretedTensor

class Linear:
  def __init__(self, din: int, dout: int, weight=None, bias=True):
    self.din, self.dout = din, dout
    self.weight = weight if weight is not None else InterpretedTensor((dout, din), [0.0]*(dout*din))
    self.bias = InterpretedTensor((dout,), [0.0]*dout) if bias else None

  def __call__(self, x):
    output = x @ self.weight.T
    if self.bias is not None: output = output + self.bias
    return output