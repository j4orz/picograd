"""
picograd is a teaching deep learning framework that bridges micograd to tinygrad which consists of a
1. sugar: domain specific ndarray language (python `Tensor`) that provides automatic differentiation, optimizers, and neural network layers
2. engine: an interpreter and compiler pipeline for an abstract compute language (python `Op`) that the tensor frontend desugars/lowers to,
    which uses heterogenous runtimes (c/c++ `Device(Runtime)`)
    the exact decomposition and intermediate representation is taken directly from tinygrad's RISC-y opset of element ops, reduce ops, and movement ops.

    a. interpreter pipeline "pt1's age of researcH":
      when evaluating the model, a graph of those decomposed ops will be traced dynamically at runtime (as opposed to just-in-time source to source transform like autograd/jax),
      which then serves as the data structure for automatic differentiation to apply backpropagation and route gradients throughout the graph
    b. compiler pipeline "pt2's age of scaling":
      which modifies the language implementation strategy from eager interpretation to just-in-time/lazy compilation
      in order to obtain a global view of the computational graph, and to apply optimizations; the primary one being fusion.
    
    gpu accelerated kernels (cuda c/hip c)
3. runtime: ...

NOTE: a lot of the complexity (and thus lines of code) are actually located in the Tensor and Runtimes,
      which reflect the M:N problem whose responsibility is behooved on language implementations to solve.
      these lines of code aren't core "business/application logic" (have nothing to do with automatic differentiation nor accelerated kernels),
      but are necessary "infrastructure/framework" logic i.e infrastructure, tools, tests.
      i.e Tensor construction logic has to support all device/dtype configurations, and Runtime has to interface with CUDA/HIP Runtimes and Compilers
"""

print("""
      ⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️
      importing picograd: the bridge from micrograd to tinygrad
      ⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️⛩️
      """)
from .sugar import nn, optim
from .sugar.tensor import Tensor
__all__ = ["Tensor"]

from importlib import import_module as _import_module
_pgrs = _import_module("picograd._pgrs")