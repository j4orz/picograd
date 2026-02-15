from teenygrad import InterpretedTensor
a, b = InterpretedTensor.arange(12).reshape((3,4)), InterpretedTensor.arange(20).reshape((4,5))
c = a@b
print(c)

import torch
from teenygrad import InterpretedTensor
x_pt = torch.tensor(3.0, requires_grad=True)
y_pt = x_pt*x_pt
y_pt.backward()

# f:R->R       f':R->R
# f(x)=x^2 ==> f'(x)=2x
#   x =3   ==> f'(x)=6
print(x_pt.grad.item())

x = InterpretedTensor((1,), [3.0], requires_grad=True) #, requires_grad=True)
y = x * x
print(y)
y.backward()
print(x.grad)